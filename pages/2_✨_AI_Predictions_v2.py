import streamlit as st
import pandas as pd
import numpy as np
import fastf1
import os  # <--- Added this to fix the error
from utils import setup_app, load_data, load_model, get_schedule, get_weekend_status
from datetime import datetime, timedelta

setup_app()

# --- 1. SETUP & LOADERS ---
TRACK_CONFIG = {
    'Monaco Grand Prix': 'Street', 'Singapore Grand Prix': 'Street', 
    'Azerbaijan Grand Prix': 'Street', 'Saudi Arabian Grand Prix': 'Street',
    'Las Vegas Grand Prix': 'Street', 'Miami Grand Prix': 'Street',
    'Australian Grand Prix': 'Street', 'Canadian Grand Prix': 'Hybrid',
    'Italian Grand Prix': 'Power', 'Belgian Grand Prix': 'Power'
}

@st.cache_resource
def load_race_artifacts():
    try:
        import pickle
        with open('data/race_model.pkl', 'rb') as f:
            return pickle.load(f)
    except: return None

data_store = load_data()
quali_artifacts = load_model()
race_artifacts = load_race_artifacts()

df_history = data_store['history']
df_train = data_store.get('training', pd.DataFrame())

# --- FIX: Safe File Loading ---
if os.path.exists('data/race_training_data.parquet'):
    df_race_train = pd.read_parquet('data/race_training_data.parquet')
else:
    df_race_train = pd.DataFrame()

st.title("📊 F1 Intelligence Hub")

if df_history.empty:
    st.error("No Data found. Please run etl_process.py")
    st.stop()

# --- TABS ---
tab1, tab2 = st.tabs(["🚀 Next Race Predictor", "🧪 Scenario Simulator"])

# ==============================================================================
# TAB 1: WEEKEND COMMAND CENTER
# ==============================================================================
with tab1:
    st.header("🏁 Race Weekend Command Center")
    
    # Refresh Button
    if st.button("🔄 Refresh Live Data", key="refresh_live"):
        st.cache_data.clear()
        st.rerun()

    now = pd.Timestamp.now()
    year = now.year
    
    # --- A. DETERMINE NEXT RACE ---
    try:
        schedule = get_schedule(year)
        if not schedule.empty and schedule['Session5Date'].dt.tz is not None:
            now = now.tz_localize(schedule['Session5Date'].dt.tz)
            
        future_races = schedule[schedule['Session5Date'] > now]
        
        if future_races.empty:
            st.success("Season Complete! See you next year.")
            st.stop()
            
        next_race = future_races.iloc[0]
        st.subheader(f"📍 {next_race['EventName']} ({year})")
        
        # Check for Sprint
        is_sprint = (next_race['EventFormat'] == 'sprint')
        if is_sprint:
            st.caption("⚡ SPRINT WEEKEND DETECTED")

        # --- B. STATUS CHECKER ---
        status_dict, _ = get_weekend_status(year, next_race['RoundNumber'])
        
        # Layout Status
        if is_sprint:
            sessions_ordered = ["FP1", "Sprint Qualifying", "Sprint", "Qualifying", "Race"]
        else:
            sessions_ordered = ["FP1", "FP2", "FP3", "Qualifying", "Race"]

        cols = st.columns(len(sessions_ordered))
        for i, sess_key in enumerate(sessions_ordered):
            # Map UI names to Status Dict keys
            map_key = sess_key
            if sess_key == "Qualifying": map_key = "Quali"
            if sess_key == "Sprint Qualifying": map_key = "Sprint Shootout" 
            
            with cols[i]:
                if map_key in status_dict:
                    s = status_dict[map_key]['state']
                    d = status_dict[map_key]['date']
                    if s == 'Complete':
                        st.success(f"**{sess_key}**\n\n✅ Done")
                    else:
                        time_str = d.strftime('%a %H:%M')
                        if (d - now) < timedelta(days=1):
                            st.warning(f"**{sess_key}**\n\n⏳ {time_str}") 
                        else:
                            st.info(f"**{sess_key}**\n\n📅 {time_str}")

        st.divider()
        
        # --- C. DETERMINE MODE (QUALI vs RACE) ---
        quali_status = status_dict.get('Quali', {}).get('state', 'upcoming')
        race_mode = (quali_status == 'Complete')
        
        # ------------------------------------------------------------------
        # MODE 1: QUALIFYING PREDICTOR (Before Quali is done)
        # ------------------------------------------------------------------
        if not race_mode:
            st.subheader("🔮 Qualifying Predictor")
            
            # 1. Determine Input Session (FP3 usually, FP1 for Sprint)
            input_session = 'FP1' if is_sprint else 'FP3'
            input_status = status_dict.get(input_session, {}).get('state', 'upcoming')
            
            st.info(f"Using **{input_session}** data to predict Qualifying grid.")
            
            # 2. Get Data (Live or Estimate)
            fp_data = {}
            if input_status == 'Complete':
                try:
                    with st.spinner(f"Fetching {input_session} results..."):
                        sess = fastf1.get_session(year, next_race['RoundNumber'], input_session)
                        sess.load(telemetry=False, messages=False)
                        fp_data = sess.results.set_index('Abbreviation')['Position'].to_dict()
                    st.success(f"✅ Loaded Live {input_session} Speeds")
                except:
                    st.warning(f"⚠️ Could not load {input_session}. Using historical estimates.")
            
            # 3. Build Input Table
            if not df_train.empty:
                latest_year = df_train['Year'].max()
                recent_races = df_train[df_train['Year'] == latest_year]
                active_drivers = sorted(recent_races['Driver'].unique())
                
                driver_stats = []
                for d in active_drivers:
                    d_hist = recent_races[recent_races['Driver'] == d]
                    team = d_hist['TeamName'].iloc[-1] if not d_hist.empty else "Unknown"
                    
                    if d in fp_data:
                        est_pos = fp_data[d]
                    else:
                        est_pos = d_hist['FP_Pos'].mean() if not d_hist.empty else 10
                    
                    driver_stats.append({'Driver': d, 'Team': team, 'FP_Rank': int(est_pos)})
                
                input_df = pd.DataFrame(driver_stats).sort_values('FP_Rank')
                
                # 4. Display/Edit Table
                st.write(f"### 1️⃣ {input_session} Standings (Input)")
                if input_status == 'Complete':
                    # Locked view if session is done
                    st.dataframe(input_df, hide_index=True, use_container_width=True)
                    edited_df = input_df 
                else:
                    # Editable view if session hasn't happened
                    edited_df = st.data_editor(
                        input_df, 
                        column_config={"FP_Rank": st.column_config.NumberColumn("Est. Rank", min_value=1, max_value=20)},
                        disabled=["Driver", "Team"],
                        hide_index=True,
                        use_container_width=True
                    )
                
                # 5. Run Quali Model
                if st.button("🚀 Predict Qualifying Grid", type="primary"):
                    if quali_artifacts:
                        results = []
                        temp_df = edited_df.copy()
                        temp_df['Proxy_Gap'] = temp_df['FP_Rank'] * 0.1 
                        team_gap_avgs = temp_df.groupby('Team')['Proxy_Gap'].mean()
                        
                        # Maps
                        driver_map = quali_artifacts['driver_map']
                        team_map = quali_artifacts['team_map']
                        le_track = quali_artifacts['le_track']
                        le_type = quali_artifacts['le_type']
                        
                        t_name = next_race['EventName']
                        t_type = TRACK_CONFIG.get(t_name, 'Circuit')
                        try: t_type_code = le_type.transform([t_type])[0]
                        except: t_type_code = 0

                        progress = st.progress(0)
                        for idx, row in temp_df.iterrows():
                            d_name = row['Driver']
                            team = row['Team']
                            fp_pos = row['FP_Rank']
                            
                            fp_gap = fp_pos * 0.1
                            t_avg = team_gap_avgs.get(team, 0.5)
                            delta = fp_gap - t_avg
                            
                            d_past = df_train[df_train['Driver'] == d_name]
                            form = d_past['Quali_Pos'].tail(3).median() if not d_past.empty else 10
                            d_tr = df_train[(df_train['Driver'] == d_name) & (df_train['EventName'] == t_name)]
                            tr_avg = d_tr['Quali_Pos'].mean() if not d_tr.empty else 10
                            
                            d_rate = driver_map.get(d_name, 10.0)
                            t_rate = team_map.get(team, 10.0)
                            
                            feats = np.array([[fp_pos, fp_gap, delta, form, tr_avg, d_rate, t_rate, t_type_code]])
                            pred = quali_artifacts['model'].predict(feats)[0]
                            results.append({'Driver': d_name, 'Team': team, 'Predicted Grid': max(1, min(20, pred))})
                            progress.progress((idx+1)/len(temp_df))
                            
                        res_df = pd.DataFrame(results).sort_values('Predicted Grid')
                        res_df['Grid Position'] = range(1, len(res_df)+1)
                        st.success("Qualifying Prediction Complete")
                        st.dataframe(res_df[['Grid Position', 'Driver', 'Team', 'Predicted Grid']], hide_index=True, use_container_width=True)
                    else:
                        st.error("Quali Model missing.")

        # ------------------------------------------------------------------
        # MODE 2: RACE PREDICTOR (After Quali is done)
        # ------------------------------------------------------------------
        else:
            st.subheader("🏎️ Race Strategy Predictor")
            
            # 1. Show Previous Sessions (Locked/Reference)
            with st.expander("📂 View Practice & Qualifying Data (Reference)"):
                st.info("Qualifying is complete. This data is locked for reference.")
                # If we had the data stored in session_state, we could show it here.
                # For now, we assume users just want the Race prediction.

            # 2. Track Risk Analysis
            chaos_map = race_artifacts.get('chaos_map', {}) if race_artifacts else {}
            chaos_score = chaos_map.get(next_race['EventName'], 0.15)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Track Type", TRACK_CONFIG.get(next_race['EventName'], 'Circuit'))
            c2.metric("DNF Probability", f"{chaos_score:.0%}")
            
            if chaos_score > 0.20: c3.error("⚠️ High Chaos Risk")
            elif chaos_score > 0.10: c3.warning("⚠️ Moderate Risk")
            else: c3.success("✅ Low Chaos Risk")
            
            # 3. Inputs: Starting Grid
            st.write("### 🚦 Starting Grid & Conditions")
            col_rain, col_fetch = st.columns([1, 2])
            is_wet = col_rain.toggle("🌧️ Wet Race?", value=False)
            
            grid_df = pd.DataFrame()
            if col_fetch.button("🔄 Reload Official Grid"):
                try:
                    with st.spinner("Fetching Official Grid..."):
                        qs = fastf1.get_session(year, next_race['RoundNumber'], 'Q')
                        qs.load(telemetry=False, messages=False)
                        if 'Position' in qs.results.columns:
                            grid_df = qs.results[['Abbreviation', 'TeamName', 'Position']].copy()
                            grid_df.columns = ['Driver', 'Team', 'Grid']
                            grid_df['Grid'] = pd.to_numeric(grid_df['Grid'], errors='coerce').fillna(20)
                            st.success("Loaded Official Grid!")
                except: st.warning("Grid fetch failed.")
            
            # Fallback Grid
            if grid_df.empty:
                active_drivers = sorted(df_train['Driver'].unique())
                grid_data = [{'Driver': d, 'Team': 'Unknown', 'Grid': i+1} for i, d in enumerate(active_drivers[:20])]
                grid_df = pd.DataFrame(grid_data)
                
            # Editable Grid (Locked if fetched, but usually good to keep editable for corrections)
            edited_grid = st.data_editor(
                grid_df,
                column_config={
                    "Grid": st.column_config.NumberColumn("Start Pos", min_value=1, max_value=20),
                    "Driver": st.column_config.TextColumn("Driver", disabled=True)
                },
                hide_index=True,
                use_container_width=True
            )
            
            # 4. Run Race Model
            if st.button("🏁 Predict Race Result", type="primary"):
                if race_artifacts:
                    results = []
                    team_grids = edited_grid.groupby('Team')['Grid'].mean()
                    
                    # Track Stats
                    if not df_race_train.empty:
                        tr_stats = df_race_train[df_race_train['EventName'] == next_race['EventName']]
                        if not tr_stats.empty:
                            overtake_diff = tr_stats.iloc[0]['Overtaking_Difficulty']
                        else:
                            overtake_diff = 0.7
                    else: overtake_diff = 0.7
                    
                    progress = st.progress(0)
                    for idx, row in edited_grid.iterrows():
                        d_name = row['Driver']
                        team = row['Team']
                        grid = row['Grid']
                        
                        # Features
                        t_avg = team_grids.get(team, 10)
                        tm_delta = grid - t_avg
                        
                        # History
                        d_hist = df_race_train[df_race_train['Driver'] == d_name].sort_values(['Year', 'RoundNumber']) if not df_race_train.empty else pd.DataFrame()
                        form = d_hist['Finish_Pos'].tail(3).median() if not d_hist.empty else 10
                        
                        d_tr = df_race_train[(df_race_train['Driver'] == d_name) & (df_race_train['EventName'] == next_race['EventName'])] if not df_race_train.empty else pd.DataFrame()
                        tr_avg = d_tr['Finish_Pos'].mean() if not d_tr.empty else 10
                        
                        # Ratings
                        d_rate = race_artifacts['driver_map'].get(d_name, 10.0)
                        t_rate = race_artifacts['team_map'].get(team, 10.0)
                        
                        # New V3 Features
                        recov = grid - t_rate
                        lock = grid * overtake_diff
                        
                        # Track Code
                        tr_type = TRACK_CONFIG.get(next_race['EventName'], 'Circuit')
                        try: tr_code = race_artifacts['le_type'].transform([tr_type])[0]
                        except: tr_code = 0
                        
                        # Predict
                        feats = np.array([[
                            grid, recov, lock, tm_delta, form, tr_avg, overtake_diff, 
                            chaos_score, 1 if is_wet else 0, d_rate, t_rate, tr_code
                        ]])
                        
                        pred = race_artifacts['model'].predict(feats)[0]
                        pred_pos = max(1, min(20, pred))
                        
                        results.append({'Driver': d_name, 'Team': team, 'Start': grid, 'Predicted Finish': pred_pos, 'Net': grid - pred_pos})
                        progress.progress((idx+1)/len(edited_grid))
                        
                    res_df = pd.DataFrame(results).sort_values('Predicted Finish')
                    res_df['Predicted Finish'] = res_df['Predicted Finish'].round(1)
                    
                    st.success("Race Strategy Analysis Complete")
                    st.dataframe(
                        res_df[['Predicted Finish', 'Driver', 'Team', 'Start', 'Net']], 
                        column_config={"Net": st.column_config.NumberColumn("Gain/Loss", format="%+d")},
                        hide_index=True, 
                        use_container_width=True
                    )

                else:
                    st.error("Race Model missing.")

    except Exception as e:
        st.error(f"System Error: {e}")

# ==============================================================================
# TAB 2: SCENARIO SIMULATOR
# ==============================================================================
with tab2:
    st.header("🧪 Scenario Simulator")
    
    sim_mode = st.radio("Simulation Mode", ["Qualifying", "Race"], horizontal=True)
    
    if sim_mode == "Qualifying":
        # Keep existing Quali Sim logic
        if not df_train.empty:
            c1, c2 = st.columns(2)
            with c1:
                drv = st.selectbox("Driver", sorted(df_train['Driver'].unique()), key="q_drv")
                trk = st.selectbox("Track", sorted(df_train['EventName'].unique()), key="q_trk")
            with c2:
                fp = st.slider("Practice Rank", 1, 20, 5, key="q_fp")
                tm_fp = st.slider("Teammate Rank", 1, 20, 5, key="q_tm")
                
            if st.button("Simulate Quali"):
                if quali_artifacts:
                    d_past = df_train[df_train['Driver'] == drv]
                    team = d_past['TeamName'].iloc[-1] if not d_past.empty else "Unknown"
                    form = d_past['Quali_Pos'].tail(3).median() if not d_past.empty else 10
                    tr_hist = df_train[(df_train['Driver'] == drv) & (df_train['EventName'] == trk)]
                    tr_avg = tr_hist['Quali_Pos'].mean() if not tr_hist.empty else 10
                    
                    fp_gap = fp * 0.1
                    tm_gap = tm_fp * 0.1
                    delta = fp_gap - ((fp_gap+tm_gap)/2)
                    
                    d_rate = quali_artifacts['driver_map'].get(drv, 10.0)
                    t_rate = quali_artifacts['team_map'].get(team, 10.0)
                    t_type = TRACK_CONFIG.get(trk, 'Circuit')
                    try: t_code = quali_artifacts['le_type'].transform([t_type])[0]
                    except: t_code = 0
                    
                    feats = np.array([[fp, fp_gap, delta, form, tr_avg, d_rate, t_rate, t_code]])
                    pred = quali_artifacts['model'].predict(feats)[0]
                    st.metric("Predicted Grid", f"P{int(pred)}")
    
    else: # Race Sim
        if not df_race_train.empty:
            c1, c2 = st.columns(2)
            with c1:
                drv = st.selectbox("Driver", sorted(df_race_train['Driver'].unique()), key="r_drv")
                trk = st.selectbox("Track", sorted(df_race_train['EventName'].unique()), key="r_trk")
            with c2:
                grid = st.slider("Grid Position", 1, 20, 10, key="r_grid")
                wet = st.checkbox("Wet Race?", key="r_wet")
                
            if st.button("Simulate Race"):
                if race_artifacts:
                    d_past = df_race_train[df_race_train['Driver'] == drv]
                    team = d_past['TeamName'].iloc[-1] if not d_past.empty else "Unknown"
                    
                    tr_stats = df_race_train[df_race_train['EventName'] == trk]
                    if not tr_stats.empty:
                        overtake = tr_stats.iloc[0]['Overtaking_Difficulty']
                        chaos = tr_stats.iloc[0]['Chaos_Factor']
                    else:
                        overtake, chaos = 0.7, 0.15
                    
                    d_rate = race_artifacts['driver_map'].get(drv, 10.0)
                    t_rate = race_artifacts['team_map'].get(team, 10.0)
                    
                    recov = grid - t_rate
                    lock = grid * overtake
                    tm_delta = 0 
                    
                    form = d_past['Finish_Pos'].tail(3).median() if not d_past.empty else 10
                    tr_hist = df_race_train[(df_race_train['Driver'] == drv) & (df_race_train['EventName'] == trk)]
                    tr_avg = tr_hist['Finish_Pos'].mean() if not tr_hist.empty else 10
                    
                    t_type = TRACK_CONFIG.get(trk, 'Circuit')
                    try: t_code = race_artifacts['le_type'].transform([t_type])[0]
                    except: t_code = 0
                    
                    feats = np.array([[
                        grid, recov, lock, tm_delta, form, tr_avg, overtake, chaos, 1 if wet else 0, d_rate, t_rate, t_code
                    ]])
                    
                    pred = race_artifacts['model'].predict(feats)[0]
                    st.metric("Predicted Finish", f"P{int(pred)}")
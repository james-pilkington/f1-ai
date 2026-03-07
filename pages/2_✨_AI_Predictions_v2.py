import streamlit as st
import pandas as pd
import numpy as np
import fastf1
import os
from utils import setup_app, load_data, load_model, get_schedule, get_weekend_status
from datetime import datetime, timedelta

setup_app()

# --- CONFIG: TRACK INTELLIGENCE ---
# Must match the training config exactly for encoding
TRACK_CONFIG = {
    'Monaco Grand Prix': {'Type': 'Street', 'Downforce': 'High'},
    'Singapore Grand Prix': {'Type': 'Street', 'Downforce': 'High'},
    'Hungarian Grand Prix': {'Type': 'Circuit', 'Downforce': 'High'},
    'Dutch Grand Prix': {'Type': 'Circuit', 'Downforce': 'High'},
    'Spanish Grand Prix': {'Type': 'Circuit', 'Downforce': 'High'},
    'Mexico City Grand Prix': {'Type': 'Circuit', 'Downforce': 'High'},
    'Bahrain Grand Prix': {'Type': 'Circuit', 'Downforce': 'Medium'},
    'Chinese Grand Prix': {'Type': 'Circuit', 'Downforce': 'Medium'},
    'Japanese Grand Prix': {'Type': 'Circuit', 'Downforce': 'High'},
    'United States Grand Prix': {'Type': 'Circuit', 'Downforce': 'Medium'},
    'Miami Grand Prix': {'Type': 'Street', 'Downforce': 'Medium'},
    'Canadian Grand Prix': {'Type': 'Hybrid', 'Downforce': 'Medium'},
    'Australian Grand Prix': {'Type': 'Street', 'Downforce': 'Medium'},
    'Qatar Grand Prix': {'Type': 'Circuit', 'Downforce': 'High'},
    'Abu Dhabi Grand Prix': {'Type': 'Circuit', 'Downforce': 'Medium'},
    'Italian Grand Prix': {'Type': 'Power', 'Downforce': 'Low'},
    'Belgian Grand Prix': {'Type': 'Power', 'Downforce': 'Low'},
    'British Grand Prix': {'Type': 'Power', 'Downforce': 'Medium'},
    'Austrian Grand Prix': {'Type': 'Power', 'Downforce': 'Medium'},
    'Saudi Arabian Grand Prix': {'Type': 'Street', 'Downforce': 'Low'},
    'Azerbaijan Grand Prix': {'Type': 'Street', 'Downforce': 'Low'},
    'Las Vegas Grand Prix': {'Type': 'Street', 'Downforce': 'Low'},
    'Sao Paulo Grand Prix': {'Type': 'Circuit', 'Downforce': 'Medium'},
}

@st.cache_resource
def load_race_suite():
    try:
        import pickle
        with open('data/race_model.pkl', 'rb') as f:
            return pickle.load(f)
    except: return None

@st.cache_data
def get_driver_profiles():
    """
    Loads the latest known stats (Pit Speed, Volatility, etc.) for every driver
    from the master history file.
    """
    if os.path.exists('data/race_data_master.parquet'):
        df = pd.read_parquet('data/race_data_master.parquet')
        # Sort by date and take the last entry for each driver
        latest = df.sort_values(['Year', 'RoundNumber']).groupby('Driver').tail(1).set_index('Driver')
        return latest
    return pd.DataFrame()

data_store = load_data()
quali_artifacts = load_model()
race_suite = load_race_suite()
driver_profiles = get_driver_profiles()
df_history = data_store['history']

st.title("📊 F1 Intelligence Hub")

# --- TABS ---
tab1, tab2 = st.tabs(["🚀 Weekend Command Center", "🧪 Lab Simulator"])

# ==============================================================================
# TAB 1: COMMAND CENTER
# ==============================================================================
with tab1:
    st.header("🏁 Race Weekend Command Center")
    
    # Refresh
    if st.button("🔄 Refresh Live Data"):
        st.cache_data.clear()
        st.rerun()

    now = pd.Timestamp.now()
    year = now.year
    
    # --- 1. NEXT RACE CONTEXT ---
    schedule = get_schedule(year)
    if not schedule.empty and schedule['Session5Date'].dt.tz is not None:
        now = now.tz_localize(schedule['Session5Date'].dt.tz)
    
    future_races = schedule[schedule['Session5Date'] > now]
    
    if future_races.empty:
        st.success("Season Complete.")
        st.stop()
        
    next_race = future_races.iloc[0]
    st.subheader(f"📍 {next_race['EventName']}")
    
    # Status Check
    status_dict, _ = get_weekend_status(year, next_race['RoundNumber'])
    is_sprint = (next_race['EventFormat'] == 'sprint')
    
    # Display Status Row (Simplified)
    sessions = ["Qualifying", "Race"] if not is_sprint else ["Sprint Qualifying", "Sprint", "Qualifying", "Race"]
    cols = st.columns(len(sessions))
    for i, sess in enumerate(sessions):
        key = "Quali" if sess == "Qualifying" else sess
        if sess == "Sprint Qualifying": key = "Sprint Shootout"
        
        with cols[i]:
            if key in status_dict:
                s = status_dict[key]['state']
                if s == 'Complete': st.success(f"{sess}\n\n✅")
                else: st.info(f"{sess}\n\n📅")

    st.divider()

    # --- 2. INTELLIGENCE MODE SELECTION ---
    # Auto-detect if Quali is done
    quali_done = status_dict.get('Quali', {}).get('state') == 'Complete'
    
    if not quali_done:
        st.info("🔮 **Qualifying Predictor Active** (Waiting for Grid)")
        # ... (Keep your existing Quali Predictor code here if you want) ...
        st.caption("Use the 'Qualifying Predictor' logic from previous versions here.")
    
    else:
        st.subheader("🏎️ Race Strategy Oracle")
        
        # --- INPUTS ---
        col_rain, col_fetch = st.columns([1, 2])
        is_wet = col_rain.toggle("🌧️ Wet Race?", value=False)
        
        # 1. Initialize session state variables
        if 'live_grid' not in st.session_state:
            st.session_state.live_grid = pd.DataFrame()
        if 'grid_auto_loaded' not in st.session_state:
            st.session_state.grid_auto_loaded = False

        # 2. AUTO-FETCH: Triggers once if Quali is done but grid isn't loaded yet
        if quali_done and not st.session_state.grid_auto_loaded:
            with st.spinner("Auto-fetching Official Grid..."):
                try:
                    qs = fastf1.get_session(year, next_race['RoundNumber'], 'Q')
                    qs.load(telemetry=False, messages=False)
                    if 'Position' in qs.results.columns:
                        grid_df = qs.results[['Abbreviation', 'TeamName', 'Position']].copy()
                        grid_df.columns = ['Driver', 'Team', 'Grid']
                        grid_df['Grid'] = pd.to_numeric(grid_df['Grid'], errors='coerce').fillna(20)
                        
                        st.session_state.live_grid = grid_df
                        st.session_state.grid_auto_loaded = True
                        st.success("Official Grid Auto-Loaded! ✅")
                except Exception:
                    st.warning("Qualifying is complete, but FastF1 results aren't published yet.")

        # 3. MANUAL REFRESH: Useful for post-quali grid penalties
        if col_fetch.button("🔄 Refresh Official Grid"):
            with st.spinner("Fetching latest Official Grid..."):
                try:
                    qs = fastf1.get_session(year, next_race['RoundNumber'], 'Q')
                    qs.load(telemetry=False, messages=False)
                    if 'Position' in qs.results.columns:
                        grid_df = qs.results[['Abbreviation', 'TeamName', 'Position']].copy()
                        grid_df.columns = ['Driver', 'Team', 'Grid']
                        grid_df['Grid'] = pd.to_numeric(grid_df['Grid'], errors='coerce').fillna(20)
                        
                        st.session_state.live_grid = grid_df
                        st.session_state.grid_auto_loaded = True
                        st.success("Grid Refreshed! ✅")
                except: 
                    st.warning("Could not fetch grid.")
            
        # 4. FALLBACK: If API fails or data is empty
        if st.session_state.live_grid.empty:
            active_drivers = sorted(driver_profiles.index.tolist()) if not driver_profiles.empty else []
            grid_data = [{'Driver': d, 'Team': 'Unknown', 'Grid': i+1} for i, d in enumerate(active_drivers[:20])]
            st.session_state.live_grid = pd.DataFrame(grid_data)
            
        # 5. RENDER EDITOR
        edited_grid = st.data_editor(
            st.session_state.live_grid,
            column_config={
                "Grid": st.column_config.NumberColumn("Start Pos", min_value=1, max_value=20),
                "Driver": st.column_config.TextColumn("Driver", disabled=True)
            },
            hide_index=True,
            use_container_width=True
        )
        
        # --- PREDICTION ENGINE ---
        if st.button("🎲 Run AI Strategy Suite", type="primary"):
            if race_suite:
                results = []
                
                # Pre-calc Team Context
                team_best_map = edited_grid.groupby('Team')['Grid'].min().to_dict()
                team_avg_map = edited_grid.groupby('Team')['Grid'].mean().to_dict()
                
                # Encode Track Info
                t_name = next_race['EventName']
                t_cfg = TRACK_CONFIG.get(t_name, {'Type': 'Circuit', 'Downforce': 'Medium'})
                
                try: 
                    t_code = race_suite['le_track'].transform([t_name])[0]
                    dt_code = race_suite['le_type'].transform([t_cfg['Type']])[0]
                    df_code = race_suite['le_downforce'].transform([t_cfg['Downforce']])[0]
                except: 
                    t_code, dt_code, df_code = 0, 0, 0 # Fallbacks
                
                progress = st.progress(0)
                
                for idx, row in edited_grid.iterrows():
                    d = row['Driver']
                    t = row['Team']
                    g = row['Grid']
                    
                    # 1. RETRIEVE PROFILE (The "Deep Stats")
                    if d in driver_profiles.index:
                        prof = driver_profiles.loc[d]
                        # Features from history
                        season_avg = prof.get('Season_Avg_Grid', 10.0)
                        form_5     = prof.get('Form_Last5_Finish', 10.0)
                        q_vol      = prof.get('Quali_Volatility', 2.0)
                        pos_gain   = prof.get('Avg_Pos_Gained', 0.0)
                        pit_speed  = prof.get('Team_Pit_Speed', 30.0)
                        q_gap      = prof.get('Quali_Gap_Pct', 0.05) # Use their historical average as proxy
                    else:
                        # Rookie Defaults
                        season_avg, form_5, q_vol, pos_gain, pit_speed, q_gap = 10, 10, 2.0, 0, 30.0, 0.05
                    
                    # 2. CALCULATED CONTEXT
                    car_pot = g - team_best_map.get(t, g)
                    tm_delta = g - team_avg_map.get(t, g)

                    # 3. BUILD FEATURE VECTOR (Must match train_suite list exactly!)
                    # ['Grid_Pos', 'Car_Potential', 'Teammate_Delta_Grid', 'Quali_Gap_Pct', 
                    #  'Season_Avg_Grid', 'Form_Last5_Finish', 'Quali_Volatility', 
                    #  'Avg_Pos_Gained', 'Team_Pit_Speed', 'Track_Code', 'Downforce_Code', 'Is_Rain']
                    
                    feats = np.array([[
                        g, car_pot, tm_delta, q_gap, season_avg, form_5, q_vol, 
                        pos_gain, pit_speed, t_code, df_code, 1 if is_wet else 0
                    ]])
                    
                    # 4. EXECUTE MODELS
                    # Model A: Pace
                    pred_pos = race_suite['model_pace'].predict(feats)[0]
                    pred_pos = max(1, min(20, pred_pos))
                    
                    # Model B: DNF Risk (Probability)
                    dnf_prob = race_suite['model_dnf'].predict_proba(feats)[0][1] # Class 1 = DNF
                    
                    # Model C: Big Mover (Class)
                    is_mover = race_suite['model_move'].predict(feats)[0]
                    
                    # 5. GENERATE INTEL BADGES
                    badges = []
                    if dnf_prob > 0.35: badges.append("⚠️ DNF Risk")
                    elif dnf_prob > 0.20: badges.append("🔸 Volatile")
                    
                    if is_mover == 1: badges.append("🚀 Big Mover")
                    
                    # Recovery Logic (If predicted to gain > 4 spots but not flagged by mover model, flag it anyway)
                    if (g - pred_pos) >= 5: badges.append("📈 Recovery")
                    
                    results.append({
                        'Driver': d,
                        'Start': g,
                        'Finish': round(pred_pos, 1),
                        'Net': int(g - pred_pos),
                        'Strategy Intel': " ".join(badges) if badges else "—"
                    })
                    
                    progress.progress((idx+1)/len(edited_grid))
                    
                # DISPLAY
                res_df = pd.DataFrame(results).sort_values('Finish')
                
                st.success("Analysis Complete")
                
                st.dataframe(
                    res_df,
                    column_config={
                        "Net": st.column_config.NumberColumn("Gain/Loss", format="%+d", help="Predicted position change"),
                        "Finish": st.column_config.NumberColumn("Pred. Finish", format="P%.1f"),
                        "Strategy Intel": st.column_config.TextColumn("AI Scouting Report")
                    },
                    hide_index=True,
                    use_container_width=True
                )
                
            else:
                st.error("Race Models not found. Run train_race_model.py")
import streamlit as st
import pandas as pd
import numpy as np
import fastf1
from utils import setup_app, load_data, load_model, get_schedule, get_weekend_status
from datetime import datetime, timedelta

setup_app()
data_store = load_data()
artifacts = load_model()
df = data_store['history']
df_train = data_store.get('training', pd.DataFrame())

st.title("📊 F1 Intelligence Hub")

if df.empty:
    st.error("No Data found. Please run etl_process.py")
    st.stop()

# --- CONFIG: TRACK TYPES (Must match generate_features.py) ---
TRACK_CONFIG = {
    'Monaco Grand Prix': 'Street', 'Singapore Grand Prix': 'Street', 
    'Azerbaijan Grand Prix': 'Street', 'Saudi Arabian Grand Prix': 'Street',
    'Las Vegas Grand Prix': 'Street', 'Miami Grand Prix': 'Street',
    'Australian Grand Prix': 'Street', 'Canadian Grand Prix': 'Hybrid',
    'Italian Grand Prix': 'Power', 'Belgian Grand Prix': 'Power'
}

# Use Tabs
tab1, tab2 = st.tabs(["Next Race Predictor", "Scenario Predictor"])

# ==============================================================================
# TAB 1: RACE PREDICTOR
# ==============================================================================
with tab1:
    st.header("🏁 Race Weekend Command Center")
    
    # 1. Force Refresh
    if st.button("🔄 Refresh Live Data"):
        st.cache_data.clear()
        st.rerun()

    now = pd.Timestamp.now()
    year = now.year
    
    try:
        schedule = get_schedule(year)
        if not schedule.empty and schedule['Session5Date'].dt.tz is not None:
            now = now.tz_localize(schedule['Session5Date'].dt.tz)
            
        future_races = schedule[schedule['Session5Date'] > now]
        
        if not future_races.empty:
            next_race = future_races.iloc[0]
            st.subheader(f"📍 {next_race['EventName']} ({year})")
            
            # --- STATUS CHECKER ---
            status_dict, _ = get_weekend_status(year, next_race['RoundNumber'])
            sessions_ordered = ["FP1", "FP2", "FP3", "Quali", "Race"]
            next_session_name = "Unknown"
            
            cols = st.columns(5)
            for i, sess in enumerate(sessions_ordered):
                with cols[i]:
                    if sess in status_dict:
                        s = status_dict[sess]['state']
                        d = status_dict[sess]['date']
                        if s == 'Complete':
                            st.success(f"**{sess}**\n\n✅ Done")
                        elif s == 'N/A':
                            st.write(f"**{sess}**\n\n⚪ --")
                        else:
                            if next_session_name == "Unknown":
                                next_session_name = status_dict[sess]['full']
                            time_str = d.strftime('%a %H:%M')
                            if (d - now) < timedelta(days=1):
                                st.warning(f"**{sess}**\n\n⏳ {time_str}") 
                            else:
                                st.info(f"**{sess}**\n\n📅 {time_str}") 

            st.divider()
            
            # --- PREDICTION LOGIC ---
            quali_done = (status_dict.get('Quali', {}).get('state') == 'Complete')
            fp3_done = (status_dict.get('FP3', {}).get('state') == 'Complete')
            
            if quali_done:
                st.info("🔒 Qualifying is complete. Predictions are locked.")
                # (Fetch official results logic here if desired...)
            else:
                st.subheader("🔮 Predict Qualifying Grid")
                st.caption(f"Next Session: {next_session_name}")
                
                # 1. Get Live FP3 Data if available
                fp3_data = {}
                if fp3_done:
                    try:
                        with st.spinner("Fetching FP3 data..."):
                            fp3_sess = fastf1.get_session(year, next_race['RoundNumber'], 'FP3')
                            fp3_sess.load(telemetry=False, messages=False)
                            fp3_data = fp3_sess.results.set_index('Abbreviation')['Position'].to_dict()
                            st.success("✅ Using Real FP3 Speeds")
                    except:
                        st.warning("⚠️ Live FP3 failed. Using estimates.")

                # 2. Prepare Inputs
                if not df_train.empty:
                    # Filter for active drivers (last season in training data)
                    latest_year = df_train['Year'].max()
                    recent_races = df_train[df_train['Year'] == latest_year]
                    active_drivers = sorted(recent_races['Driver'].unique())
                    
                    driver_stats = []
                    
                    for d in active_drivers:
                        # Get Estimates
                        d_history = recent_races[recent_races['Driver'] == d]
                        team = d_history['TeamName'].iloc[-1] if not d_history.empty else "Unknown"
                        
                        # FP3 Pos (Live or Avg)
                        if d in fp3_data:
                            est_pos = fp3_data[d]
                        else:
                            est_pos = d_history['FP_Pos'].mean() if not d_history.empty else 10
                            
                        driver_stats.append({
                            'Driver': d, 
                            'Team': team, 
                            'Est_FP_Pos': int(est_pos)
                        })
                    
                    input_df = pd.DataFrame(driver_stats).sort_values('Est_FP_Pos')
                    
                    with st.expander("Adjust Driver Form (FP3 Estimates)"):
                        edited_df = st.data_editor(
                            input_df, 
                            column_config={"Est_FP_Pos": st.column_config.NumberColumn("FP3 Rank", min_value=1, max_value=20)},
                            disabled=["Driver", "Team"],
                            hide_index=True,
                            use_container_width=True
                        )
                    
                    if st.button("🚀 Run Prediction Model", type="primary"):
                        if artifacts:
                            results = []
                            # Pre-calculate Team Averages for Delta Calculation
                            temp_df = edited_df.copy()
                            # Proxy for Gap: Position * 0.1s (e.g. P1=0.1s, P2=0.2s)
                            temp_df['Proxy_Gap'] = temp_df['Est_FP_Pos'] * 0.1 
                            team_gap_avgs = temp_df.groupby('Team')['Proxy_Gap'].mean()
                            
                            # Prepare Encoders & Maps
                            driver_map = artifacts['driver_map']
                            team_map = artifacts['team_map']
                            le_track = artifacts['le_track']
                            le_type = artifacts['le_type']
                            
                            # Get Track Details
                            track_name = next_race['EventName']
                            track_type_str = TRACK_CONFIG.get(track_name, 'Circuit')
                            
                            # Encode Track
                            try:
                                track_code = le_track.transform([track_name])[0]
                            except: track_code = 0 # Default if new track
                                
                            try:
                                track_type_code = le_type.transform([track_type_str])[0]
                            except: track_type_code = 0

                            progress_bar = st.progress(0)
                            
                            for idx, row in temp_df.iterrows():
                                d_name = row['Driver']
                                team = row['Team']
                                fp_pos = row['Est_FP_Pos']
                                fp_gap = fp_pos * 0.1 # Proxy
                                
                                # 1. Calculate Teammate Delta
                                t_avg_gap = team_gap_avgs.get(team, 0.5)
                                delta_gap = fp_gap - t_avg_gap
                                
                                # 2. Calculate Form (Last 3 Races)
                                # Look up in df_train
                                d_past = df_train[df_train['Driver'] == d_name].sort_values(['Year', 'RoundNumber'])
                                if not d_past.empty:
                                    form_last3 = d_past['Quali_Pos'].tail(3).median()
                                else:
                                    form_last3 = 10 # Rookie default
                                    
                                # 3. Calculate Track History
                                d_track_hist = df_train[(df_train['Driver'] == d_name) & (df_train['EventName'] == track_name)]
                                if not d_track_hist.empty:
                                    track_avg = d_track_hist['Quali_Pos'].mean()
                                else:
                                    track_avg = d_past['Quali_Pos'].mean() if not d_past.empty else 10

                                # 4. Get Ratings
                                d_rating = driver_map.get(d_name, 10.0)
                                t_rating = team_map.get(team, 10.0)
                                
                                # 5. Build Feature Array (Order MUST match train_model.py)
                                # ['FP_Pos', 'FP_Gap', 'Teammate_Delta_Gap', 'Form_Last3', 'Driver_Track_Avg', 'Driver_Rating', 'Team_Rating', 'Track_Type_Code']
                                features = np.array([[
                                    fp_pos, fp_gap, delta_gap, form_last3, track_avg, d_rating, t_rating, track_type_code
                                ]])
                                
                                # 6. Predict
                                raw_pred = artifacts['model'].predict(features)[0]
                                pred_pos = max(1, min(20, raw_pred)) # Clip 1-20
                                
                                results.append({'Driver': d_name, 'Team': team, 'Predicted_Pos': pred_pos})
                                progress_bar.progress((idx+1)/len(temp_df))
                                
                            res_df = pd.DataFrame(results).sort_values('Predicted_Pos')
                            res_df['Grid Position'] = range(1, len(res_df)+1)
                            
                            st.success("Analysis Complete!")
                            st.dataframe(res_df[['Grid Position', 'Driver', 'Team', 'Predicted_Pos']], hide_index=True, use_container_width=True)
                        else:
                            st.error("Model artifacts missing. Run train_model.py")
        else:
            st.success("Season Complete!")
    except Exception as e:
        st.error(f"System Error: {e}")

# ==============================================================================
# TAB 2: SCENARIO PREDICTOR (Single Driver)
# ==============================================================================
with tab2:
    st.header("🧪 Scenario Simulator")
    st.caption("What if... simulation for specific conditions.")
    
    if not df_train.empty:
        c1, c2 = st.columns(2)
        with c1:
            drv = st.selectbox("Driver", sorted(df_train['Driver'].unique()))
            trk = st.selectbox("Track", sorted(df_train['EventName'].unique()))
        with c2:
            fp = st.slider("Practice Rank", 1, 20, 5)
            tm_fp = st.slider("Teammate Rank", 1, 20, 5)
            
        if st.button("Simulate"):
            if artifacts:
                # 1. Infer Context Data from Selection
                # We use historical averages for the 'hidden' features to make the tool easy to use
                d_past = df_train[df_train['Driver'] == drv]
                team = d_past['TeamName'].iloc[-1] if not d_past.empty else "Unknown"
                
                # Form
                form_val = d_past['Quali_Pos'].tail(3).median() if not d_past.empty else 10
                
                # Track History
                tr_hist = df_train[(df_train['Driver'] == drv) & (df_train['EventName'] == trk)]
                track_avg = tr_hist['Quali_Pos'].mean() if not tr_hist.empty else 10
                
                # Gaps (Proxy)
                fp_gap = fp * 0.1
                tm_gap = tm_fp * 0.1
                # Simplified Delta: Driver Gap - Avg Team Gap (approx (Driver+Teammate)/2)
                team_avg_gap = (fp_gap + tm_gap) / 2
                delta_gap = fp_gap - team_avg_gap
                
                # Encoders
                d_rating = artifacts['driver_map'].get(drv, 10.0)
                t_rating = artifacts['team_map'].get(team, 10.0)
                
                track_type = TRACK_CONFIG.get(trk, 'Circuit')
                try: tr_type_code = artifacts['le_type'].transform([track_type])[0]
                except: tr_type_code = 0
                
                # Predict
                feats = np.array([[
                    fp, fp_gap, delta_gap, form_val, track_avg, d_rating, t_rating, tr_type_code
                ]])
                
                pred = artifacts['model'].predict(feats)[0]
                final_pred = max(1, min(20, pred))
                
                st.metric("Predicted Grid Spot", f"P{int(final_pred)}")
                st.progress(max(0.0, min(1.0, (21-final_pred)/20)))
                
            else:
                st.error("Model missing")
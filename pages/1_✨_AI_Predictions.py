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

st.title("📊 F1 Intelligence Hub")

if df.empty:
    st.error("No Data.")
    st.stop()

# Use Tabs
tab1, tab2 = st.tabs(["Next Race Predictor", "Scenario Predictor"])

# ==============================================================================
# TAB 1: RACE PREDICTOR
# ==============================================================================
with tab1:
    st.title("🏁 Race Weekend Command Center")
    
    # 1. Force Refresh Button
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
            
            # Find the TRUE next session
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
                            # If we haven't found a next session yet, this is it
                            if next_session_name == "Unknown":
                                next_session_name = status_dict[sess]['full']
                                
                            time_str = d.strftime('%a %H:%M')
                            if (d - now) < timedelta(days=1):
                                st.warning(f"**{sess}**\n\n⏳ {time_str}") 
                            else:
                                st.info(f"**{sess}**\n\n📅 {time_str}") 

            st.divider()
            
            # --- PREDICTION INTERFACE ---
            quali_done = (status_dict.get('Quali', {}).get('state') == 'Complete')
            fp3_done = (status_dict.get('FP3', {}).get('state') == 'Complete')
            
            if quali_done:
                st.info("🔒 Qualifying is complete. Predictions are locked.")
                # ... (Your existing code for fetching Official Quali Results) ...
                try:
                    with st.spinner("Fetching official results..."):
                        qs = fastf1.get_session(year, next_race['RoundNumber'], 'Q')
                        qs.load(telemetry=False, messages=False)
                        
                        res_cols = ['Position', 'Driver', 'TeamName']
                        for q in ['Q1', 'Q2', 'Q3']:
                            if q in qs.results.columns:
                                res_cols.append(q)
                                
                        results = qs.results[res_cols].copy()
                        for q in ['Q1', 'Q2', 'Q3']:
                            if q in results.columns:
                                results[q] = results[q].astype(str).str.replace('NaT', '-')
                        st.dataframe(results, hide_index=True, use_container_width=True)
                except Exception as e:
                    st.warning(f"Official results pending: {e}")

            else:
                st.subheader("🔮 Predict Qualifying Grid")
                st.caption(f"Next Session: {next_session_name}")
                
                # --- LIVE FP3 DATA FETCH ---
                fp3_data = {}
                if fp3_done:
                    try:
                        with st.spinner("FP3 Complete! Fetching latest practice speeds..."):
                            fp3_sess = fastf1.get_session(year, next_race['RoundNumber'], 'FP3')
                            fp3_sess.load(telemetry=False, messages=False)
                            # Create a map: {'VER': 1, 'HAM': 5, ...}
                            fp3_data = fp3_sess.results.set_index('Abbreviation')['Position'].to_dict()
                            st.success("✅ Prediction inputs updated with REAL FP3 data.")
                    except:
                        st.warning("⚠️ Could not load live FP3 data. Using historical estimates.")

                # Build Input Data
                df_train = data_store['training']
                if not df_train.empty:
                    latest_year = df_train['Year'].max()
                    recent_races = df_train[df_train['Year'] == latest_year]
                    if recent_races.empty: recent_races = df_train[df_train['Year'] == latest_year-1]
                    
                    active_drivers = recent_races['Driver'].unique()
                    driver_stats = []
                    
                    for d in active_drivers:
                        d_data = recent_races[recent_races['Driver'] == d].tail(5)
                        team = d_data['TeamName'].iloc[-1] if not d_data.empty else "Unknown"
                        
                        # LOGIC: If we have live FP3 data, use it. Else, use historical mean.
                        if d in fp3_data:
                            estimated_pos = fp3_data[d]
                        else:
                            estimated_pos = d_data['FP_Pos'].mean() if not d_data.empty else 10
                            
                        driver_stats.append({
                            'Driver': d, 
                            'Team': team, 
                            'Est_FP_Pos': int(estimated_pos)
                        })
                    
                    input_df = pd.DataFrame(driver_stats).sort_values('Est_FP_Pos')
                    
                    with st.expander("Adjust Driver Form (FP3 Estimates)"):
                        edited_df = st.data_editor(
                            input_df, 
                            column_config={"Est_FP_Pos": st.column_config.NumberColumn("Est. FP3 Rank", min_value=1, max_value=20)},
                            disabled=["Driver", "Team"],
                            hide_index=True,
                            use_container_width=True
                        )
                    
                    if st.button("🚀 Run Prediction Model", type="primary"):
                        if artifacts:
                            results = []
                            temp_df = edited_df.copy()
                            team_avgs = temp_df.groupby('Team')['Est_FP_Pos'].mean()
                            
                            progress_bar = st.progress(0)
                            for idx, row in temp_df.iterrows():
                                d_name = row['Driver']
                                fp_rank = row['Est_FP_Pos']
                                team = row['Team']
                                t_avg = team_avgs.get(team, 10)
                                delta = fp_rank - t_avg
                                
                                le_d = artifacts['le_driver']
                                le_t = artifacts['le_track']
                                d_code = le_d.transform([d_name])[0] if d_name in le_d.classes_ else 0
                                t_code = le_t.transform([next_race['EventName']])[0] if next_race['EventName'] in le_t.classes_ else 0
                                
                                model_input = np.array([[fp_rank, t_avg, delta, d_code, t_code]])
                                raw_pred = artifacts['model'].predict(model_input)[0]
                                
                                # --- FIX: P0 CORRECTION ---
                                # Clip result to be at least 1
                                pred_pos = max(1, raw_pred)
                                
                                results.append({'Driver': d_name, 'Team': team, 'Predicted_Pos': pred_pos})
                                progress_bar.progress((idx+1)/len(temp_df))
                                
                            res_df = pd.DataFrame(results).sort_values('Predicted_Pos')
                            res_df['Grid Position'] = range(1, len(res_df)+1)
                            st.dataframe(res_df[['Grid Position', 'Driver', 'Team', 'Predicted_Pos']], hide_index=True, use_container_width=True)
                        else:
                            st.error("Model not loaded.")
        else:
            st.success("Season Complete!")
            
    except Exception as e:
        st.error(f"Error: {e}")

# ==============================================================================
# TAB 2: SCENARIO PREDICTOR
# ==============================================================================
with tab2:
    st.title("🧪 Scenario Simulator")
    
    df_train = data_store['training']
    if not df_train.empty:
        c1, c2 = st.columns(2)
        with c1:
            drv = st.selectbox("Driver", sorted(df_train['Driver'].unique()))
            trk = st.selectbox("Track", sorted(df_train['EventName'].unique()))
        with c2:
            fp = st.slider("FP3 Pos", 1, 20, 1)
            tm = st.slider("Teammate FP3", 1, 20, 5)
            
        if st.button("Simulate Result"):
            if artifacts:
                d_c = artifacts['le_driver'].transform([drv])[0] if drv in artifacts['le_driver'].classes_ else 0
                t_c = artifacts['le_track'].transform([trk])[0] if trk in artifacts['le_track'].classes_ else 0
                avg = (fp+tm)/2
                
                # Run Prediction
                raw_pred = artifacts['model'].predict(np.array([[fp, avg, fp-avg, d_c, t_c]]))[0]
                
                # --- FIX: P0 CORRECTION ---
                final_pred = max(1, raw_pred)
                
                st.metric("Predicted Grid Spot", f"P{int(final_pred)}")
                
                # Visual Bar
                st.progress(max(0.0, min(1.0, (21-final_pred)/20)))
            else:
                st.error("Model missing")
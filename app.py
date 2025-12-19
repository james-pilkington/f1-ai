import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle
import numpy as np
import fastf1
import fastf1.plotting
from datetime import datetime, timedelta # <--- FIXED: Added timedelta
import os # <--- FIXED: Added os

# 1. Page Configuration
st.set_page_config(
    page_title="F1 AI Strategist",
    page_icon="🏎️",
    layout="wide"
)

# --- FIXED: Enable FastF1 Cache (CRITICAL FOR PERFORMANCE) ---
if not os.path.exists('f1_cache'):
    os.makedirs('f1_cache')
fastf1.Cache.enable_cache('f1_cache') 
fastf1.plotting.setup_mpl(misc_mpl_mods=False)

# TEAM COLORS (2024/2025 Palette)
TEAM_COLORS = {
    'Mercedes': '#00D7B6',
    'Red Bull Racing': '#4781D7',
    'Red Bull': '#4781D7',
    'Ferrari': '#ED1131',
    'McLaren': '#F47600',
    'Alpine': '#00A1E8',
    'Renault': '#00A1E8',
    'Racing Bulls': '#6C98FF',
    'RB': '#6C98FF',
    'AlphaTauri': '#2B4562',
    'Toro Rosso': '#469BFF',
    'Aston Martin': '#229971',
    'Aston Martin Aramco': '#229971',
    'Racing Point': '#F596C8',
    'Williams': '#1868DB',
    'Kick Sauber': '#01C00E',
    'Sauber': '#01C00E',
    'Alfa Romeo': '#900000',
    'Haas F1 Team': '#9C9FA2',
    'Haas': '#9C9FA2',
}

# 2. Helper Functions
@st.cache_data(ttl=3600)
def get_schedule(year):
    """
    Fetches the schedule and forces date columns to be datetime objects.
    Fixes the '.dt accessor' error.
    """
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        
        # FIX: Force convert all date columns to datetime
        date_cols = ['EventDate', 'Session1Date', 'Session2Date', 'Session3Date', 'Session4Date', 'Session5Date']
        for col in date_cols:
            if col in schedule.columns:
                schedule[col] = pd.to_datetime(schedule[col], errors='coerce')
                
        return schedule
    except Exception as e:
        print(f"Error fetching schedule: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_next_race():
    now = pd.Timestamp.now()
    current_year = now.year
    
    # Check current year
    try:
        schedule = get_schedule(current_year)
        
        # Handle timezone if present
        if not schedule.empty and schedule['EventDate'].dt.tz is not None:
            now = now.tz_localize(schedule['EventDate'].dt.tz)
            
        future_races = schedule[schedule['EventDate'] > now]
        if not future_races.empty:
            return future_races.iloc[0], current_year
    except Exception as e:
        print(f"Error checking {current_year}: {e}")

    # Check next year
    next_year = current_year + 1
    try:
        schedule_next = get_schedule(next_year)
        if not schedule_next.empty:
            return schedule_next.iloc[0], next_year
    except Exception as e:
        print(f"Error checking {next_year}: {e}")
        
    return None, None

@st.cache_data(ttl=3600)
def get_weekend_status(year, round_num):
    status = {}
    try:
        schedule = get_schedule(year)
        event = schedule[schedule['RoundNumber'] == round_num].iloc[0]
        
        now = pd.Timestamp.now()
        if event['Session5Date'].tzinfo is not None:
            now = now.tz_localize(event['Session5Date'].tzinfo)

        sessions = [
            ('FP1', event['Session1Date'], 'Practice 1'),
            ('FP2', event['Session2Date'], 'Practice 2'),
            ('FP3', event['Session3Date'], 'Practice 3'),
            ('Quali', event['Session4Date'], 'Qualifying'),
            ('Race', event['Session5Date'], 'Race')
        ]
        
        for short_name, date_obj, full_name in sessions:
            if pd.isna(date_obj):
                status[short_name] = {'state': 'N/A', 'full_name': full_name}
                continue
                
            if now > (date_obj + timedelta(hours=2)):
                status[short_name] = {'state': 'Complete', 'full_name': full_name, 'date': date_obj}
            else:
                status[short_name] = {'state': 'TBD', 'full_name': full_name, 'date': date_obj}
                
        return status, event['EventName']
    except Exception as e:
        return {}, "Unknown"

@st.cache_data
def load_data():
    data = {}
    try:
        df = pd.read_parquet('data/processed_f1_data.parquet')
        for c in ['Position', 'GridPosition', 'Points', 'RoundNumber', 'Year']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        data['history'] = df
    except FileNotFoundError:
        data['history'] = pd.DataFrame()
        
    try:
        data['training'] = pd.read_parquet('data/quali_training_data.parquet')
    except FileNotFoundError:
        data['training'] = pd.DataFrame()
    
    try:
        data['maps'] = pd.read_parquet('data/track_maps.parquet')
    except FileNotFoundError:
        data['maps'] = pd.DataFrame()
        
    return data

@st.cache_resource
def load_model():
    try:
        with open('data/quali_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

# 3. Load Resources
data_store = load_data()
artifacts = load_model()

# 4. Sidebar & Navigation
st.sidebar.title("🏎️ F1 AI Strategist")
st.sidebar.markdown("---")

if artifacts:
    st.sidebar.success(f"🧠 Model Active\nAccuracy: ±{artifacts['mae']:.2f} spots")
else:
    st.sidebar.warning("⚠️ Model not found. Run train_model.py")

page = st.sidebar.radio("Navigate", ["Next Race Oracle", "Scenario Predictor", "Stats Dashboard"])

# ==============================================================================
# VIEW 1: NEXT RACE ORACLE
# ==============================================================================    
if page == "Next Race Oracle":
    st.title("🏁 Race Weekend Command Center")
    
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
            
            # --- 1. LIVE WEEKEND TRACKER ---
            status_dict, _ = get_weekend_status(year, next_race['RoundNumber'])
            
            cols = st.columns(5)
            sessions = ["FP1", "FP2", "FP3", "Quali", "Race"]
            
            for i, sess in enumerate(sessions):
                with cols[i]:
                    if sess in status_dict:
                        s = status_dict[sess]['state']
                        d = status_dict[sess]['date']
                        
                        if s == 'Complete':
                            st.success(f"**{sess}**\n\n✅ Complete")
                        elif s == 'N/A':
                            st.write(f"**{sess}**\n\n⚪ --") 
                        else:
                            time_str = d.strftime('%a %H:%M')
                            if (d - now) < timedelta(days=1):
                                st.warning(f"**{sess}**\n\n⏳ {time_str}") 
                            else:
                                st.info(f"**{sess}**\n\n📅 {time_str}") 
            
            st.divider()
            
            # --- 2. PREDICTION INTERFACE ---
            quali_done = (status_dict.get('Quali', {}).get('state') == 'Complete')
            
            if quali_done:
                st.info("🔒 Qualifying is complete. Predictions are locked. Official results loaded below.")
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
                    st.warning(f"Official results pending (API processing): {e}")
            else:
                st.subheader("🔮 Predict Qualifying Grid")
                st.caption(f"Next Session: {next_race['EventName']} Qualifying")
                
                df_train = data_store['training']
                if not df_train.empty:
                    latest_year = df_train['Year'].max()
                    recent_races = df_train[df_train['Year'] == latest_year]
                    if recent_races.empty: recent_races = df_train[df_train['Year'] == latest_year-1]
                    
                    active_drivers = recent_races['Driver'].unique()
                    driver_stats = []
                    for d in active_drivers:
                        d_data = recent_races[recent_races['Driver'] == d].tail(5)
                        avg_fp = d_data['FP_Pos'].mean() if not d_data.empty else 10
                        team = d_data['TeamName'].iloc[-1] if not d_data.empty else "Unknown"
                        driver_stats.append({'Driver': d, 'Team': team, 'Est_FP_Pos': int(avg_fp)})
                    
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
                                pred_pos = artifacts['model'].predict(model_input)[0]
                                results.append({'Driver': d_name, 'Team': team, 'Predicted_Pos': pred_pos})
                                progress_bar.progress((idx+1)/len(temp_df))
                                
                            res_df = pd.DataFrame(results).sort_values('Predicted_Pos')
                            res_df['Grid Position'] = range(1, len(res_df)+1)
                            st.success("Prediction Complete!")
                            st.dataframe(res_df[['Grid Position', 'Driver', 'Team', 'Predicted_Pos']], hide_index=True, use_container_width=True)
                        else:
                            st.error("Model not loaded.")

            # 3. Race Prediction
            st.divider()
            st.subheader("🏎️ Predict Race Result")
            race_btn = st.button("Predict Race Strategy", disabled=not quali_done)
            if not quali_done:
                st.caption("🚫 Available after Qualifying is complete.")
            elif race_btn:
                st.info("🚧 Race Strategy Module is under construction (Coming v1.1)")
        else:
            st.success("Season Complete! No upcoming races detected.")
            
    except Exception as e:
        st.error(f"Error loading schedule: {e}")

# ==============================================================================
# VIEW 2: SCENARIO PREDICTOR
# ==============================================================================
elif page == "Scenario Predictor":
    st.title("🧪 Single Driver Scenario")
    st.markdown("Simulate how a specific practice result affects qualifying.")
    
    df_train = data_store['training']
    if df_train.empty:
        st.stop()
        
    col1, col2 = st.columns(2)
    with col1:
        drivers = sorted(df_train['Driver'].unique())
        s_driver = st.selectbox("Driver", drivers, index=drivers.index('VER') if 'VER' in drivers else 0)
        tracks = sorted(df_train['EventName'].unique())
        s_track = st.selectbox("Track", tracks)
        
    with col2:
        fp_pos = st.slider("FP3 Position", 1, 20, 1)
        tm_pos = st.slider("Teammate FP3 Position", 1, 20, 5)
        
    if st.button("Predict"):
        if artifacts:
            team_avg = (fp_pos + tm_pos) / 2
            delta = fp_pos - team_avg
            
            le_d = artifacts['le_driver']
            le_t = artifacts['le_track']
            
            d_code = le_d.transform([s_driver])[0] if s_driver in le_d.classes_ else 0
            t_code = le_t.transform([s_track])[0] if s_track in le_t.classes_ else 0
            
            pred = artifacts['model'].predict(np.array([[fp_pos, team_avg, delta, d_code, t_code]]))[0]
            
            st.balloons()
            st.metric("Predicted Grid Spot", f"P{int(pred)}")
            st.progress(max(0.0, min(1.0, (21-pred)/20)))
        else:
            st.error("Model missing")

# ==============================================================================
# VIEW 3: STATS DASHBOARD
# ==============================================================================
elif page == "Stats Dashboard":
    st.title("📊 F1 Intelligence Hub")
    
    df = data_store['history']
    if df.empty:
        st.error("No historic data found. Run etl_process.py")
        st.stop()

    stats_mode = st.radio("Select View:", ["By Season", "By Race", "By Track", "By Driver"], horizontal=True)
    st.divider()

    # --- A. BY SEASON (Fixed Legend: Shows Driver Name with Team Color) ---
    if stats_mode == "By Season":
        years = sorted(df['Year'].unique(), reverse=True)
        selected_year = st.selectbox("Select Season", years)
        season_df = df[df['Year'] == selected_year].copy()
        
        total_races = season_df['RoundNumber'].nunique()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Races Completed", total_races)
        
        # Calculate Remaining if current season
        current_year = datetime.now().year
        if selected_year == current_year:
            schedule = get_schedule(selected_year)
            if not schedule.empty:
                scheduled_races = schedule[schedule['EventFormat'] != 'testing'].shape[0]
                races_left = scheduled_races - total_races
                col2.metric("Races Remaining", races_left)
                col3.metric("Max Points Available", races_left * 26)
                col4.metric("Progress", f"{int((total_races/scheduled_races)*100)}%")

        st.subheader("🏆 Championship Battle")
        season_df = season_df.sort_values('RoundNumber')
        points_matrix = season_df.pivot_table(index='RoundNumber', columns='Driver', values='Points', aggfunc='sum').fillna(0)
        cumsum_df = points_matrix.cumsum()
        chart_data = cumsum_df.reset_index()
        long_df = chart_data.melt(id_vars='RoundNumber', var_name='Driver', value_name='Total Points')
        
        # Filter top 10
        top_drivers = cumsum_df.iloc[-1].sort_values(ascending=False).head(10).index
        
        # Get Team Names
        driver_teams = season_df[['Driver', 'TeamName']].drop_duplicates(subset=['Driver'], keep='last')
        chart_data = pd.merge(long_df, driver_teams, on='Driver', how='left')
        
        # Calculate Rank for Line Dash (Solid for #1 driver, Dash for #2)
        chart_data['Rank'] = chart_data.groupby(['TeamName', 'Driver'])['Total Points'].transform('max')
        team_ranks = chart_data[['TeamName', 'Driver', 'Rank']].drop_duplicates().sort_values(['TeamName', 'Rank'], ascending=[True, False])
        team_ranks['Style'] = team_ranks.groupby('TeamName').cumcount()
        team_ranks['LineDash'] = team_ranks['Style'].map({0: 'solid', 1: 'dash', 2: 'dot', 3: 'dashdot'})
        
        final_chart = pd.merge(chart_data[chart_data['Driver'].isin(top_drivers)], team_ranks[['Driver', 'LineDash']], on='Driver')
        
        # --- KEY FIX: Map Driver Names to Team Colors ---
        driver_color_map = {}
        for _, row in final_chart[['Driver', 'TeamName']].drop_duplicates().iterrows():
            # Look up team color, default to grey if missing
            driver_color_map[row['Driver']] = TEAM_COLORS.get(row['TeamName'], '#808080')

        fig = px.line(
            final_chart, 
            x='RoundNumber', 
            y='Total Points', 
            color='Driver',          # <--- Change: Color by Driver Name (fixes Legend)
            line_dash='LineDash', 
            title=f"{selected_year} Driver Championship Evolution", 
            color_discrete_map=driver_color_map, # <--- Change: Apply manual color map
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Season Statistics")
        stats = season_df.groupby('Driver').agg(
            Total_Points=('Points', 'sum'),
            Wins=('Position', lambda x: (x==1).sum()),
            Podiums=('Position', lambda x: (x<=3).sum()),
            Poles=('GridPosition', lambda x: (x==1).sum()),
            DNFs=('Status', lambda x: x.astype(str).str.contains('Retired|Collision|Accident|Engine').sum())
        ).sort_values('Total_Points', ascending=False)
        st.dataframe(stats, use_container_width=True)

    # --- B. BY RACE ---
    elif stats_mode == "By Race":
        years = sorted(df['Year'].unique(), reverse=True)
        col_y, col_r = st.columns(2)
        with col_y:
            selected_year = st.selectbox("Season", years)
        races = df[df['Year'] == selected_year]['EventName'].unique()
        with col_r:
            selected_race = st.selectbox("Race", races)
        
        race_data = df[(df['Year'] == selected_year) & (df['EventName'] == selected_race)].sort_values('Position')
        
        st.subheader(f"🏁 Race Results: {selected_race}")
        st.dataframe(race_data[['Position', 'Driver', 'TeamName', 'Points', 'GridPosition']], hide_index=True, use_container_width=True)
        
        st.divider()
        st.subheader("⏱️ Qualifying Pace Analysis")
        
        try:
            with st.spinner(f"Analyzing {selected_year} {selected_race} Qualifying..."):
                rnd_num = race_data.iloc[0]['RoundNumber']
                qs = fastf1.get_session(selected_year, rnd_num, 'Q')
                qs.load(telemetry=False, messages=False)
                
                q_res = qs.results.copy()
                q_res['Time'] = q_res['Q3'].combine_first(q_res['Q2']).combine_first(q_res['Q1'])
                q_res = q_res.dropna(subset=['Time']) 
                
                if not q_res.empty:
                    pole_time = q_res['Time'].min()
                    q_res['Gap'] = (q_res['Time'] - pole_time).dt.total_seconds()
                    q_res = q_res.sort_values('Gap')
                    
                    plot_data = q_res.head(15).copy()
                    plot_data['Gap_Text'] = plot_data['Gap'].apply(lambda x: f"+{x:.3f}" if x > 0 else "POLE")
                    
                    fig_gap = px.bar(
                        plot_data, 
                        y='Abbreviation', 
                        x='Gap', 
                        text='Gap_Text', 
                        color='TeamName', 
                        orientation='h', 
                        color_discrete_map=TEAM_COLORS,
                        title=f"Gap to Pole ({pole_time.total_seconds() // 60:.0f}:{pole_time.total_seconds() % 60:06.3f})",
                        height=500
                    )
                    
                    fig_gap.update_traces(textposition='outside', cliponaxis=False)
                    fig_gap.update_layout(
                        plot_bgcolor='#15151e', 
                        paper_bgcolor='#15151e',
                        font=dict(color='white'), 
                        yaxis=dict(autorange="reversed", title=""), 
                        xaxis=dict(title="Gap (Seconds)", showgrid=False),
                        showlegend=False,
                        margin=dict(r=100) 
                    )
                    st.plotly_chart(fig_gap, use_container_width=True)
                else:
                    st.warning("No valid qualifying times found.")
        except Exception as e:
            st.info(f"Qualifying data unavailable: {e}")

    # --- C. BY TRACK ---
    elif stats_mode == "By Track":
        tracks = sorted(df['EventName'].unique())
        selected_track = st.selectbox("Select Track", tracks)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"Circuit Map: {selected_track}")
            map_df = data_store.get('maps', pd.DataFrame())
            
            if not map_df.empty and selected_track in map_df['EventName'].values:
                track_row = map_df[map_df['EventName'] == selected_track].iloc[0]
                x_line = track_row['X_Data']
                y_line = track_row['Y_Data']
                
                fig_map = go.Figure()
                fig_map.add_trace(go.Scatter(
                    x=x_line, y=y_line, 
                    mode='lines', 
                    line=dict(color='#ff1801', width=4), 
                    hoverinfo='skip'
                ))
                
                fig_map.update_layout(
                    plot_bgcolor='#15151e', 
                    paper_bgcolor='#15151e',
                    xaxis=dict(visible=False, showgrid=False),
                    yaxis=dict(visible=False, showgrid=False, scaleanchor="x", scaleratio=1),
                    margin=dict(l=20, r=20, t=20, b=20), 
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig_map, use_container_width=True)
            else:
                st.warning("⚠️ Map data not available.")

        with col2:
            st.subheader("Track Stats")
            track_df = df[df['EventName'] == selected_track]
            wins = track_df[track_df['Position'] == 1]['Driver'].value_counts().reset_index()
            wins.columns = ['Driver', 'Wins']
            st.write("**👑 Most Wins**")
            st.dataframe(wins.head(5), hide_index=True, use_container_width=True)
            
            st.write("**📅 Recent Winners**")
            recent = track_df[track_df['Position'] == 1][['Year', 'Driver']].sort_values('Year', ascending=False)
            st.dataframe(recent.head(5), hide_index=True, use_container_width=True)
            
        winners = track_df[track_df['Position'] == 1]
        fig = px.histogram(winners, x='GridPosition', nbins=20, title="Winning Grid Positions")
        fig.update_layout(bargap=0.2)
        st.plotly_chart(fig, use_container_width=True)

    # --- D. BY DRIVER ---
    elif stats_mode == "By Driver":
        drivers = sorted(df['Driver'].unique())
        s_driver = st.selectbox("Select Driver", drivers, index=drivers.index('HAM') if 'HAM' in drivers else 0)
        
        driver_df = df[df['Driver'] == s_driver].copy()
        
        st.subheader(f"Career Overview: {s_driver}")
        
        total_races = driver_df['RoundNumber'].count()
        total_points = driver_df['Points'].sum()
        total_wins = len(driver_df[driver_df['Position'] == 1])
        total_podiums = len(driver_df[driver_df['Position'] <= 3])
        avg_finish = driver_df['Position'].mean()
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Races", total_races)
        col2.metric("Points", f"{int(total_points)}")
        col3.metric("Wins", total_wins)
        col4.metric("Podiums", total_podiums)
        col5.metric("Avg Finish", f"P{avg_finish:.1f}")
        
        st.subheader("⚔️ Head-to-Head: Driver vs. Teammate")
        
        comparison_data = []
        seasons_active = sorted(driver_df['Year'].unique())
        
        for year in seasons_active:
            d_season = driver_df[driver_df['Year'] == year]
            d_points = d_season['Points'].sum()
            team_name = d_season['TeamName'].iloc[0]
            
            team_season = df[(df['Year'] == year) & (df['TeamName'] == team_name)]
            teammate_season = team_season[team_season['Driver'] != s_driver]
            
            tm_points = 0
            if not teammate_season.empty:
                tm_points = teammate_season['Points'].sum()
                if len(teammate_season['Driver'].unique()) > 1:
                    tm_points = tm_points / len(teammate_season['Driver'].unique())
                
            comparison_data.append({'Year': year, 'Driver': d_points, 'Teammate': tm_points, 'Team': team_name})
            
        comp_df = pd.DataFrame(comparison_data)
        comp_long = comp_df.melt(id_vars=['Year', 'Team'], value_vars=['Driver', 'Teammate'], var_name='Metric', value_name='Points')
        
        fig = px.bar(comp_long, x='Year', y='Points', color='Metric', barmode='group',
                     title="Season Performance vs. Teammate",
                     color_discrete_map={'Driver': '#1f77b4', 'Teammate': '#ff7f0e'})
        st.plotly_chart(fig, use_container_width=True)
        
        if total_wins > 0:
            st.subheader(f"🏆 Trophy Cabinet ({total_wins} Wins)")
            wins = driver_df[driver_df['Position'] == 1][['Year', 'EventName', 'TeamName', 'GridPosition']].sort_values('Year', ascending=False)
            st.dataframe(wins, hide_index=True, use_container_width=True)
        
        st.subheader("Consistency Analysis")
        fig_hist = px.histogram(driver_df, x='Position', nbins=20, title="Finishing Position Distribution", color_discrete_sequence=['#2ca02c'])
        fig_hist.update_layout(bargap=0.1)
        st.plotly_chart(fig_hist, use_container_width=True)
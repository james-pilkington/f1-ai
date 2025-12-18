import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import numpy as np
import fastf1
from datetime import datetime

# 1. Page Configuration
st.set_page_config(
    page_title="F1 AI Strategist",
    page_icon="🏎️",
    layout="wide"
)

# 2. Helper Functions
@st.cache_data(ttl=3600)
def get_next_race():
    """
    Automatically detects the next upcoming race.
    Handles the rollover from current season to next season.
    """
    now = datetime.now()
    current_year = now.year
    
    # Check remaining races in current year
    try:
        schedule = fastf1.get_event_schedule(current_year, include_testing=False)
        future_races = schedule[schedule['EventDate'] > now]
        if not future_races.empty:
            return future_races.iloc[0], current_year
    except Exception as e:
        print(f"Error checking {current_year}: {e}")

    # If no races left, check next year
    next_year = current_year + 1
    try:
        schedule_next = fastf1.get_event_schedule(next_year, include_testing=False)
        if not schedule_next.empty:
            return schedule_next.iloc[0], next_year
    except Exception as e:
        print(f"Error checking {next_year}: {e}")
        
    return None, None

@st.cache_data(ttl=3600)
def get_schedule(year):
    """Fetches the full schedule for a given year to calculate remaining races."""
    try:
        return fastf1.get_event_schedule(year, include_testing=False)
    except:
        return pd.DataFrame()

@st.cache_data
def load_data():
    data = {}
    try:
        # Historic Results (Main dataset for stats)
        df = pd.read_parquet('data/processed_f1_data.parquet')
        
        # CLEANING: Ensure Position and Points are numeric for stats
        df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
        df['GridPosition'] = pd.to_numeric(df['GridPosition'], errors='coerce')
        df['Points'] = pd.to_numeric(df['Points'], errors='coerce')
        
        data['history'] = df
    except FileNotFoundError:
        data['history'] = pd.DataFrame()
        
    try:
        data['training'] = pd.read_parquet('data/quali_training_data.parquet')
    except FileNotFoundError:
        data['training'] = pd.DataFrame()
        
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
    st.title("🏁 Next Race Oracle")
    st.markdown("Predict the starting grid for the upcoming race based on **Recent Form**.")
    
    df_train = data_store['training']
    if df_train.empty:
        st.error("Training data missing. Please run generate_features.py")
        st.stop()

    # --- A. Detect Track ---
    next_race, season = get_next_race()
    
    if next_race is not None:
        race_name = next_race['EventName']
        race_date = next_race['EventDate'].strftime('%d %b %Y')
        st.info(f"📍 Detected Next Race: **{race_name}** ({season}) on **{race_date}**")
        target_track = st.selectbox("Confirm Track", [race_name])
    else:
        st.warning("⚠️ Could not detect upcoming race (Season might be over). Select manually:")
        tracks = sorted(df_train['EventName'].unique())
        target_track = st.selectbox("Select Track", tracks)

    st.divider()

    # --- B. Estimate Recent Form ---
    st.subheader("Step 1: Driver Form Estimation")
    st.caption("The AI estimates 'Practice Pace' using each driver's average FP3 rank from the most recent season.")

    latest_year = df_train['Year'].max()
    recent_races = df_train[df_train['Year'] == latest_year]
    
    if recent_races.empty:
        recent_races = df_train[df_train['Year'] == latest_year - 1]

    active_drivers = recent_races['Driver'].unique()
    driver_stats = []
    
    for d in active_drivers:
        d_data = recent_races[recent_races['Driver'] == d].tail(5)
        avg_fp = d_data['FP_Pos'].mean() if not d_data.empty else 10
        team = d_data['TeamName'].iloc[-1] if not d_data.empty else "Unknown"
        driver_stats.append({'Driver': d, 'Team': team, 'Est_FP_Pos': int(avg_fp)})
    
    input_df = pd.DataFrame(driver_stats).sort_values('Est_FP_Pos')
    edited_df = st.data_editor(
        input_df, 
        column_config={
            "Est_FP_Pos": st.column_config.NumberColumn("Est. FP3 Rank", min_value=1, max_value=20)
        },
        disabled=["Driver", "Team"],
        hide_index=True,
        use_container_width=True 
    )
    
    # --- C. Run Prediction ---
    if st.button("🔮 Generate Grid Prediction", type="primary"):
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
            t_code = le_t.transform([target_track])[0] if target_track in le_t.classes_ else 0
            
            model_input = np.array([[fp_rank, t_avg, delta, d_code, t_code]])
            pred_pos = artifacts['model'].predict(model_input)[0]
            
            results.append({'Driver': d_name, 'Team': team, 'Predicted_Pos': pred_pos})
            progress_bar.progress((idx + 1) / len(temp_df))
            
        res_df = pd.DataFrame(results).sort_values('Predicted_Pos')
        res_df['Grid Position'] = range(1, len(res_df) + 1)
        
        st.success("Prediction Complete!")
        st.subheader(f"Predicted Starting Grid: {target_track}")
        
        st.dataframe(
            res_df[['Grid Position', 'Driver', 'Team', 'Predicted_Pos']],
            column_config={
                "Predicted_Pos": st.column_config.ProgressColumn(
                    "Model Confidence Score",
                    format="%.2f",
                    min_value=1,
                    max_value=20,
                ),
            },
            hide_index=True,
            use_container_width=True
        )

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

# ==============================================================================
# VIEW 3: STATS DASHBOARD
# ==============================================================================
elif page == "Stats Dashboard":
    st.title("📊 F1 Intelligence Hub")
    
    df = data_store['history']
    if df.empty:
        st.error("No historic data found. Run etl_process.py")
        st.stop()

    # Navigation for Dashboard
    stats_mode = st.radio("Select View:", ["By Season", "By Race", "By Track", "By Driver"], horizontal=True)
    st.divider()

    # --------------------------------------------------------------------------
    # A. BY SEASON
    # --------------------------------------------------------------------------
    if stats_mode == "By Season":
        years = sorted(df['Year'].unique(), reverse=True)
        selected_year = st.selectbox("Select Season", years)
        season_df = df[df['Year'] == selected_year].copy()
        
        total_races = season_df['RoundNumber'].nunique()
        current_year = datetime.now().year
        is_current_season = (selected_year == current_year)
        
        if is_current_season:
            schedule = get_schedule(selected_year)
            if not schedule.empty:
                scheduled_races = schedule[schedule['EventFormat'] != 'testing'].shape[0]
                races_left = scheduled_races - total_races
                points_available = races_left * 26
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Races Completed", total_races)
                col2.metric("Races Remaining", races_left)
                col3.metric("Max Points Available", points_available)
                col4.metric("Season Progress", f"{int((total_races/scheduled_races)*100)}%")
        else:
            st.metric("Total Races", total_races)

        st.subheader("🏆 Championship Battle")
        season_df = season_df.sort_values('RoundNumber')
        points_matrix = season_df.pivot_table(index='RoundNumber', columns='Driver', values='Points', aggfunc='sum').fillna(0)
        cumsum_df = points_matrix.cumsum()
        chart_data = cumsum_df.reset_index()
        long_df = chart_data.melt(id_vars='RoundNumber', var_name='Driver', value_name='Total Points')
        
        # Filter top 10
        top_drivers = cumsum_df.iloc[-1].sort_values(ascending=False).head(10).index
        final_chart_data = long_df[long_df['Driver'].isin(top_drivers)]
        
        fig = px.line(final_chart_data, x='RoundNumber', y='Total Points', color='Driver', 
                      title=f"{selected_year} Driver Championship Evolution", markers=True)
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

    # --------------------------------------------------------------------------
    # B. BY RACE
    # --------------------------------------------------------------------------
    elif stats_mode == "By Race":
        years = sorted(df['Year'].unique(), reverse=True)
        col_y, col_r = st.columns(2)
        with col_y:
            selected_year = st.selectbox("Season", years)
        df_year = df[df['Year'] == selected_year]
        races = df_year['EventName'].unique()
        with col_r:
            selected_race = st.selectbox("Race", races)
        
        race_data = df_year[df_year['EventName'] == selected_race].copy()
        race_data = race_data.sort_values(by='Position', ascending=True)
        race_data = race_data.dropna(subset=['Position'])
        
        st.subheader(f"{selected_year} {selected_race} Results")
        col1, col2, col3 = st.columns(3)
        drivers = race_data['Driver'].values
        teams = race_data['TeamName'].values
        
        if len(race_data) >= 1: col1.metric("🥇 Winner", drivers[0], teams[0])
        if len(race_data) >= 2: col2.metric("🥈 Second", drivers[1], teams[1])
        if len(race_data) >= 3: col3.metric("🥉 Third", drivers[2], teams[2])
            
        display_cols = ['Position', 'Driver', 'TeamName', 'GridPosition', 'Points', 'Status']
        st.dataframe(race_data[display_cols], hide_index=True, use_container_width=True)
        
        race_data['Positions_Gained'] = race_data['GridPosition'] - race_data['Position']
        fig_gain = px.bar(race_data, x='Driver', y='Positions_Gained', 
                          color='Positions_Gained', title="Positions Gained/Lost",
                          color_continuous_scale=px.colors.diverging.RdBu)
        st.plotly_chart(fig_gain, use_container_width=True)

    # --------------------------------------------------------------------------
    # C. BY TRACK
    # --------------------------------------------------------------------------
    elif stats_mode == "By Track":
        tracks = sorted(df['EventName'].unique())
        selected_track = st.selectbox("Select Track", tracks)
        track_df = df[df['EventName'] == selected_track].copy()
        
        st.subheader(f"History: {selected_track}")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### 👑 Track Kings (Most Wins)")
            wins = track_df[track_df['Position'] == 1]['Driver'].value_counts().reset_index()
            wins.columns = ['Driver', 'Wins']
            st.dataframe(wins.head(5), hide_index=True, use_container_width=True)
            
            st.markdown("#### ⚡ Qualifying Masters")
            poles = track_df[track_df['GridPosition'] == 1]['Driver'].value_counts().reset_index()
            poles.columns = ['Driver', 'Poles']
            st.dataframe(poles.head(5), hide_index=True, use_container_width=True)

        with col2:
            st.markdown("#### 📅 Recent Winners")
            recent_winners = track_df[track_df['Position'] == 1][['Year', 'Driver', 'TeamName']].sort_values('Year', ascending=False)
            st.dataframe(recent_winners, hide_index=True, use_container_width=True)
            
        winners = track_df[track_df['Position'] == 1]
        fig = px.histogram(winners, x='GridPosition', nbins=20, 
                           title=f"Where do winners start at {selected_track}?",
                           labels={'GridPosition': 'Starting Grid Spot'})
        fig.update_layout(bargap=0.2)
        st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------------------------
    # D. BY DRIVER (Added Back!)
    # --------------------------------------------------------------------------
    elif stats_mode == "By Driver":
        drivers = sorted(df['Driver'].unique())
        selected_driver = st.selectbox("Select Driver", drivers, index=drivers.index('HAM') if 'HAM' in drivers else 0)
        
        driver_df = df[df['Driver'] == selected_driver].copy()
        
        st.subheader(f"Career Overview: {selected_driver}")
        
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
            teammate_season = team_season[team_season['Driver'] != selected_driver]
            
            if not teammate_season.empty:
                total_team_points = team_season['Points'].sum()
                tm_points = total_team_points - d_points
            else:
                tm_points = 0
                
            comparison_data.append({
                'Year': year,
                'Driver Points': d_points,
                'Teammate Points': tm_points,
                'Team': team_name
            })
            
        comp_df = pd.DataFrame(comparison_data)
        comp_long = comp_df.melt(id_vars=['Year', 'Team'], value_vars=['Driver Points', 'Teammate Points'], 
                                var_name='Metric', value_name='Points')
        
        fig = px.bar(comp_long, x='Year', y='Points', color='Metric', barmode='group',
                     title="Season Performance vs. Teammate(s)",
                     hover_data=['Team'],
                     color_discrete_map={'Driver Points': '#1f77b4', 'Teammate Points': '#ff7f0e'})
        st.plotly_chart(fig, use_container_width=True)
        
        if total_wins > 0:
            st.subheader(f"🏆 Trophy Cabinet ({total_wins} Wins)")
            wins = driver_df[driver_df['Position'] == 1][['Year', 'EventName', 'TeamName', 'GridPosition']].sort_values('Year', ascending=False)
            st.dataframe(wins, hide_index=True, use_container_width=True)
        
        st.subheader("Consistency Analysis")
        fig_hist = px.histogram(driver_df, x='Position', nbins=20, 
                                title="Finishing Position Distribution",
                                labels={'Position': 'Race Result'},
                                color_discrete_sequence=['#2ca02c'])
        fig_hist.update_layout(bargap=0.1)
        st.plotly_chart(fig_hist, use_container_width=True)
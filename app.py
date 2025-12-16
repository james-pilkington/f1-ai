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

@st.cache_data
def load_data():
    data = {}
    try:
        data['history'] = pd.read_parquet('data/processed_f1_data.parquet')
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
# VIEW 1: NEXT RACE ORACLE (The Main Feature)
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
        
        # Use detected track
        target_track = st.selectbox("Confirm Track", [race_name])
    else:
        st.warning("⚠️ Could not detect upcoming race (Season might be over). Select manually:")
        tracks = sorted(df_train['EventName'].unique())
        target_track = st.selectbox("Select Track", tracks)

    st.divider()

    # --- B. Estimate Recent Form ---
    st.subheader("Step 1: Driver Form Estimation")
    st.caption("The AI estimates 'Practice Pace' using each driver's average FP3 rank from the most recent season.")

    # Get the latest available year in our data (likely 2025)
    latest_year = df_train['Year'].max()
    recent_races = df_train[df_train['Year'] == latest_year]
    
    # If 2025 is empty (e.g. start of season), fall back to previous year
    if recent_races.empty:
        recent_races = df_train[df_train['Year'] == latest_year - 1]

    # Calculate Form
    active_drivers = recent_races['Driver'].unique()
    driver_stats = []
    
    for d in active_drivers:
        d_data = recent_races[recent_races['Driver'] == d].tail(5) # Last 5 races
        avg_fp = d_data['FP_Pos'].mean() if not d_data.empty else 10
        team = d_data['TeamName'].iloc[-1] if not d_data.empty else "Unknown"
        driver_stats.append({'Driver': d, 'Team': team, 'Est_FP_Pos': int(avg_fp)})
    
    # Editable Table
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
        
        # Calculate Team Averages dynamically from user input
        team_avgs = temp_df.groupby('Team')['Est_FP_Pos'].mean()
        
        progress_bar = st.progress(0)
        
        for idx, row in temp_df.iterrows():
            d_name = row['Driver']
            fp_rank = row['Est_FP_Pos']
            team = row['Team']
            
            # Prepare Features
            t_avg = team_avgs.get(team, 10)
            delta = fp_rank - t_avg
            
            le_d = artifacts['le_driver']
            le_t = artifacts['le_track']
            
            # Handle unknown drivers/tracks (e.g. Rookies or New Tracks)
            d_code = le_d.transform([d_name])[0] if d_name in le_d.classes_ else 0
            t_code = le_t.transform([target_track])[0] if target_track in le_t.classes_ else 0
            
            # Predict
            model_input = np.array([[fp_rank, t_avg, delta, d_code, t_code]])
            pred_pos = artifacts['model'].predict(model_input)[0]
            
            results.append({
                'Driver': d_name,
                'Team': team,
                'Predicted_Pos': pred_pos
            })
            progress_bar.progress((idx + 1) / len(temp_df))
            
        # Display Results
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
    st.title("📊 Historical Analysis")
    df = data_store['history']
    
    if df.empty:
        st.error("No historic data found.")
    else:
        years = sorted(df['Year'].unique(), reverse=True)
        selected_year = st.sidebar.selectbox("Season", years)
        
        df_year = df[df['Year'] == selected_year]
        races = df_year['EventName'].unique()
        selected_race = st.sidebar.selectbox("Race", races)
        
        race_data = df_year[df_year['EventName'] == selected_race]
        final_results = race_data[['Driver', 'TeamName', 'Position', 'GridPosition']].drop_duplicates().sort_values('Position')
        
        st.subheader(f"{selected_year} {selected_race}")
        if len(final_results) >= 3:
            col1, col2, col3 = st.columns(3)
            col1.metric("🥇 Winner", final_results.iloc[0]['Driver'])
            col2.metric("🥈 P2", final_results.iloc[1]['Driver'])
            col3.metric("🥉 P3", final_results.iloc[2]['Driver'])
            
        st.dataframe(final_results, hide_index=True, use_container_width=True)
        
        fig = px.scatter(final_results, x='GridPosition', y='Position', color='TeamName', text='Driver', title="Grid vs. Finish Position")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)
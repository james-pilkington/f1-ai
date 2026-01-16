import streamlit as st
import pandas as pd  # <--- FIXED: Added missing import
import plotly.express as px
import plotly.graph_objects as go
import fastf1
from utils import setup_app, load_data, get_schedule, TEAM_COLORS
from datetime import datetime

# Initialize
setup_app()
data_store = load_data()
df = data_store['history']

st.title("📊 Season Analytics")

if df.empty:
    st.error("No Data found. Please run the ETL process.")
    st.stop()

# --- FIXED INDENTATION STARTING HERE ---
years = sorted(df['Year'].unique(), reverse=True)
selected_year = st.selectbox("Select Season", years)
season_df = df[df['Year'] == selected_year].copy()

# 1. Season Overview Metrics
total_races = season_df['RoundNumber'].nunique()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Races Completed", total_races)

# Calculate Remaining if current season
current_year = datetime.now().year
if selected_year == current_year:
    schedule = get_schedule(selected_year)
    if not schedule.empty:
        # Count non-testing events
        scheduled_races = schedule[schedule['EventFormat'] != 'testing'].shape[0]
        races_left = scheduled_races - total_races
        
        col2.metric("Races Remaining", races_left)
        # 26 pts = 25 (Win) + 1 (Fastest Lap) estimate
        col3.metric("Max Points Available", races_left * 26)
        
        if scheduled_races > 0:
            col4.metric("Progress", f"{int((total_races/scheduled_races)*100)}%")

# 2. Championship Battle Chart
st.subheader("🏆 Championship Battle")
season_df = season_df.sort_values('RoundNumber')

# Create Pivot Table (Drivers vs Rounds)
points_matrix = season_df.pivot_table(index='RoundNumber', columns='Driver', values='Points', aggfunc='sum').fillna(0)
cumsum_df = points_matrix.cumsum()

# Melt back to long format for Plotly
chart_data = cumsum_df.reset_index()
long_df = chart_data.melt(id_vars='RoundNumber', var_name='Driver', value_name='Total Points')

# Filter Top 10 Drivers only (to avoid clutter)
top_drivers = cumsum_df.iloc[-1].sort_values(ascending=False).head(10).index

# Merge Team Name back for coloring
driver_teams = season_df[['Driver', 'TeamName']].drop_duplicates(subset=['Driver'], keep='last')
chart_data = pd.merge(long_df, driver_teams, on='Driver', how='left')

# Calculate Rank for Line Styles (Solid vs Dash)
# Rank 1 in team = Solid line, Rank 2 = Dash
chart_data['Rank'] = chart_data.groupby(['TeamName', 'Driver'])['Total Points'].transform('max')
team_ranks = chart_data[['TeamName', 'Driver', 'Rank']].drop_duplicates().sort_values(['TeamName', 'Rank'], ascending=[True, False])
team_ranks['Style'] = team_ranks.groupby('TeamName').cumcount()
team_ranks['LineDash'] = team_ranks['Style'].map({0: 'solid', 1: 'dash', 2: 'dot', 3: 'dashdot'})

# Merge styles back
final_chart = pd.merge(chart_data[chart_data['Driver'].isin(top_drivers)], team_ranks[['Driver', 'LineDash']], on='Driver')

# Map Colors manually to ensure Legend shows Driver Name but uses Team Color
driver_color_map = {}
for _, row in final_chart[['Driver', 'TeamName']].drop_duplicates().iterrows():
    driver_color_map[row['Driver']] = TEAM_COLORS.get(row['TeamName'], '#808080')

fig = px.line(
    final_chart, 
    x='RoundNumber', 
    y='Total Points', 
    color='Driver',          # Key: Color by Driver to get names in legend
    line_dash='LineDash',    # Key: Differentiate teammates by style
    title=f"{selected_year} Driver Championship Evolution", 
    color_discrete_map=driver_color_map, 
    markers=True
)
st.plotly_chart(fig, use_container_width=True)

# 3. Stats Table
st.subheader("Season Statistics")
stats = season_df.groupby('Driver').agg(
    Total_Points=('Points', 'sum'),
    Wins=('Position', lambda x: (x==1).sum()),
    Podiums=('Position', lambda x: (x<=3).sum()),
    Poles=('GridPosition', lambda x: (x==1).sum()),
    DNFs=('Status', lambda x: x.astype(str).str.contains('Retired|Collision|Accident|Engine').sum())
).sort_values('Total_Points', ascending=False)

st.dataframe(stats, use_container_width=True)
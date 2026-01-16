import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils import setup_app, load_data, TEAM_COLORS

# --- INITIALIZATION ---
setup_app()
data_store = load_data()
df = data_store['history']

st.title("🛠️ Constructor Grid")

if df.empty:
    st.error("No data found.")
    st.stop()

# State Management
if 'selected_team' not in st.session_state:
    st.session_state.selected_team = None

# --- CONSTANTS: LOGO MAPPING ---
# Using official F1 website logo naming convention where possible
# Fallback to a generic F1 car silhouette
LOGO_BASE = "https://media.formula1.com/content/dam/fom-website/teams/"
LOGOS = {
    "Red Bull Racing": f"{LOGO_BASE}2024/red-bull-racing-logo.png.transform/2col/image.png",
    "Ferrari": f"{LOGO_BASE}2024/ferrari-logo.png.transform/2col/image.png",
    "Mercedes": f"{LOGO_BASE}2024/mercedes-logo.png.transform/2col/image.png",
    "McLaren": f"{LOGO_BASE}2024/mclaren-logo.png.transform/2col/image.png",
    "Aston Martin": f"{LOGO_BASE}2024/aston-martin-logo.png.transform/2col/image.png",
    "Alpine": f"{LOGO_BASE}2024/alpine-logo.png.transform/2col/image.png",
    "Williams": f"{LOGO_BASE}2024/williams-logo.png.transform/2col/image.png",
    "Racing Bulls": f"{LOGO_BASE}2024/rb-logo.png.transform/2col/image.png",
    "RB": f"{LOGO_BASE}2024/rb-logo.png.transform/2col/image.png",
    "Kick Sauber": f"{LOGO_BASE}2024/kick-sauber-logo.png.transform/2col/image.png",
    "Haas F1 Team": f"{LOGO_BASE}2024/haas-f1-team-logo.png.transform/2col/image.png",
    "DEFAULT": "https://media.formula1.com/content/dam/fom-website/teams/2024/red-bull-racing-logo.png.transform/2col/image.png" # Safe fallback
}

# ==============================================================================
# VIEW 1: THE GRID (MASTER VIEW)
# ==============================================================================
if st.session_state.selected_team is None:
    # --- CONTROLS ---
    years = sorted(df['Year'].unique(), reverse=True)
    selected_year = st.selectbox("Select Season", years)

    # Filter Data
    season_df = df[df['Year'] == selected_year]

    # Aggregate Team Stats (Summing both drivers)
    team_stats = season_df.groupby('TeamName').agg(
        Points=('Points', 'sum'),
        Wins=('Position', lambda x: (x==1).sum()),
        Podiums=('Position', lambda x: (x<=3).sum()),
        Races=('RoundNumber', 'nunique'),
        Best_Finish=('Position', 'min')
    ).reset_index().sort_values('Points', ascending=False)

    # --- GRID LAYOUT ---
    cols_per_row = 3  # Wider cards for teams look better
    rows = [st.columns(cols_per_row) for _ in range(len(team_stats) // cols_per_row + 1)]

    for i, (index, row) in enumerate(team_stats.iterrows()):
        team_name = row['TeamName']
        team_color = TEAM_COLORS.get(team_name, "#333333")
        # Try exact match, then short match, then default
        #logo_url = LOGOS.get(team_name, LOGOS["DEFAULT"])
        # New:
        from utils import get_team_logo_url # Import at top
        # ... inside the loop ...
        logo_url = get_team_logo_url(team_name, selected_year)
        
        row_idx = i // cols_per_row
        col_idx = i % cols_per_row
        
        with rows[row_idx][col_idx]:
            with st.container(border=True):
                # Branding Strip
                st.markdown(f"""<div style="background-color: {team_color}; height: 5px; width: 100%; border-radius: 5px;"></div>""", unsafe_allow_html=True)
                
                # Header
                c1, c2 = st.columns([1, 3])
                with c1: st.write("🏁") # Placeholder icon if logo fails, or put st.image here if you have good URLs
                with c2: st.subheader(team_name)
                
                # Drivers in this team
                drivers = season_df[season_df['TeamName'] == team_name]['Driver'].unique()
                st.caption(f"Drivers: {', '.join(drivers)}")
                
                st.markdown("---")
                m1, m2, m3 = st.columns(3)
                m1.metric("Pts", int(row['Points']))
                m2.metric("Wins", row['Wins'])
                m3.metric("Pod", row['Podiums'])
                
                if st.button("View Analysis", key=f"btn_{team_name}", use_container_width=True):
                    st.session_state.selected_team = team_name
                    st.rerun()

# ==============================================================================
# VIEW 2: TEAM DETAIL (DRILL DOWN)
# ==============================================================================
else:
    # --- HEADER ---
    col_back, col_title = st.columns([1, 5])
    with col_back:
        if st.button("← Back"):
            st.session_state.selected_team = None
            st.rerun()
            
    s_team = st.session_state.selected_team
    team_color = TEAM_COLORS.get(s_team, "#333333")
    
    st.header(f"{s_team} Performance")
    st.markdown(f"""<div style="background-color: {team_color}; height: 5px; width: 100%; margin-bottom: 20px;"></div>""", unsafe_allow_html=True)

    # Get Team Data
    # Note: We need the YEAR from the previous view, but since we are in a new view, 
    # we usually grab the 'Latest' or ask the user again. 
    # Better UX: Store 'selected_year' in session state too.
    # For now, we'll infer the year from the most recent data for this team or default to 2025.
    
    # Let's add a year selector inside the detail view for flexibility
    years = sorted(df[df['TeamName'] == s_team]['Year'].unique(), reverse=True)
    d_year = st.selectbox("Season", years)
    
    team_df = df[(df['TeamName'] == s_team) & (df['Year'] == d_year)].copy()
    
    # Top Stats
    total_pts = team_df['Points'].sum()
    wins = len(team_df[team_df['Position'] == 1])
    # Calculate Championship Position (Rank among teams)
    all_teams = df[df['Year'] == d_year].groupby('TeamName')['Points'].sum().sort_values(ascending=False).reset_index()
    try:
        rank = all_teams[all_teams['TeamName'] == s_team].index[0] + 1
    except: rank = "-"
        
    c1, c2, c3 = st.columns(3)
    c1.metric("Championship Rank", f"P{rank}")
    c2.metric("Total Points", int(total_pts))
    c3.metric("Race Wins", wins)
    
    st.divider()

    # --- CHART 1: DRIVER CONTRIBUTION (Stacked Bar) ---
    st.subheader("👥 Driver Contribution")
    
    driver_contrib = team_df.groupby('Driver')['Points'].sum().reset_index()
    driver_contrib['Percentage'] = (driver_contrib['Points'] / total_pts * 100).round(1)
    
    col_chart, col_table = st.columns([2, 1])
    
    with col_chart:
        fig_contrib = px.bar(
            driver_contrib, 
            y='Driver', 
            x='Points', 
            text='Points', 
            color='Driver',
            orientation='h',
            title="Points per Driver",
            # We map generic colors to ensure contrast
            color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c'] 
        )
        fig_contrib.update_layout(showlegend=False)
        st.plotly_chart(fig_contrib, use_container_width=True)
        
    with col_table:
        st.write("**Impact Breakdown**")
        st.dataframe(
            driver_contrib[['Driver', 'Points', 'Percentage']].sort_values('Points', ascending=False),
            hide_index=True
        )

    # --- CHART 2: SEASON TRAJECTORY ---
    st.subheader("📈 Season Trajectory")
    
    # Calculate Cumulative Points by Round
    team_df = team_df.sort_values('RoundNumber')
    
    # We need to sum points per round (both cars)
    round_points = team_df.groupby('RoundNumber')['Points'].sum().cumsum().reset_index()
    
    # Compare with the Rival (Next best team)
    # Find who finished just above or below them in championship
    try:
        rival_idx = rank - 2 if rank > 1 else 1 # If P1, compare to P2. If P5, compare to P4.
        rival_name = all_teams.iloc[rival_idx]['TeamName']
        
        rival_df = df[(df['TeamName'] == rival_name) & (df['Year'] == d_year)]
        rival_points = rival_df.groupby('RoundNumber')['Points'].sum().cumsum().reset_index()
        
        # Merge for plotting
        merged = pd.merge(round_points, rival_points, on='RoundNumber', suffixes=(f'_{s_team}', f'_{rival_name}'))
        
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=merged['RoundNumber'], y=merged[f'Points_{s_team}'],
            name=s_team,
            line=dict(color=team_color, width=4)
        ))
        fig_line.add_trace(go.Scatter(
            x=merged['RoundNumber'], y=merged[f'Points_{rival_name}'],
            name=rival_name,
            line=dict(color=TEAM_COLORS.get(rival_name, 'grey'), width=2, dash='dot')
        ))
        fig_line.update_layout(title=f"Battle vs. {rival_name}", xaxis_title="Round", yaxis_title="Total Points")
        st.plotly_chart(fig_line, use_container_width=True)
        
    except:
        # Fallback simple line if calculation fails
        st.line_chart(round_points.set_index('RoundNumber'))

    # --- CHART 3: FINISHING POSITIONS (Heatmap style) ---
    st.subheader("🏁 Race Results Heatmap")
    
    # Pivot: Driver vs Round -> Position
    # This creates a nice grid showing P1, P2, DNF for each driver each race
    heatmap_data = team_df.pivot_table(
        index='Driver', 
        columns='RoundNumber', 
        values='Position',
        aggfunc='min' # In case of weird duplicates
    )
    
    fig_heat = px.imshow(
        heatmap_data,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r", # Red = Bad (High number), Blue = Good (Low number)
        title="Finishing Positions (Darker Blue = Better)"
    )
    fig_heat.update_xaxes(dtick=1)
    st.plotly_chart(fig_heat, use_container_width=True)

    # --- CHART 3: HISTORICAL LINEAGE (Name Changes) ---
    st.subheader("📜 Team History & Lineage")
    
    from utils import TEAM_LINEAGE # Import locally or at top
    
    # 1. Identify Lineage
    # Find which lineage list this team belongs to
    lineage_names = [s_team] # Default to just current name
    for current_name, past_names in TEAM_LINEAGE.items():
        if s_team in past_names or s_team == current_name:
            lineage_names = past_names
            break
            
    st.caption(f"Tracking history for: {', '.join(lineage_names)}")
    
    # 2. Calculate Historic Performance
    # We need to scan ALL years to find where these specific team names finished
    history_data = []
    all_years = sorted(df['Year'].unique())
    
    for y in all_years:
        # Get total points for ALL teams that year to determine rank
        y_df = df[df['Year'] == y]
        if y_df.empty: continue
            
        yearly_standings = y_df.groupby('TeamName')['Points'].sum().sort_values(ascending=False).reset_index()
        yearly_standings['Rank'] = yearly_standings.index + 1
        
        # Check if any of our lineage names raced this year
        # We look for a match in the yearly_standings
        match = yearly_standings[yearly_standings['TeamName'].isin(lineage_names)]
        
        if not match.empty:
            # Found the team!
            row = match.iloc[0]
            history_data.append({
                'Year': y,
                'Team Name': row['TeamName'], # Capture the name used THAT year
                'Rank': row['Rank'],
                'Points': row['Points']
            })
            
    hist_df = pd.DataFrame(history_data)
    
    if not hist_df.empty:
        fig_hist = px.line(
            hist_df, 
            x='Year', 
            y='Rank', 
            text='Team Name', # Show the name change on the chart points
            markers=True,
            title=f"Constructors' Championship History ({hist_df['Year'].min()} - {hist_df['Year'].max()})"
        )
        
        # Style: Invert Y axis (P1 at top), Label points with Team Name
        fig_hist.update_traces(textposition="top center")
        fig_hist.update_layout(
            yaxis=dict(autorange="reversed", title="Championship Finish", dtick=1),
            xaxis=dict(dtick=1) # Show every year
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("No historical data found for this lineage.")
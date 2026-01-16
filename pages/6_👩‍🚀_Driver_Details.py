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

st.title("🏎️ Driver Grid")

if df.empty:
    st.error("No data found. Run etl_process.py")
    st.stop()

# Initialize Session State for navigation
if 'selected_driver' not in st.session_state:
    st.session_state.selected_driver = None

# --- CONSTANTS ---
HEADSHOTS = {
    "VER": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/M/MAXVER01_Max_Verstappen/maxver01.png.transform/2col/image.png",
    "PER": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/S/SERPER01_Sergio_Perez/serper01.png.transform/2col/image.png",
    "HAM": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/L/LEWHAM01_Lewis_Hamilton/lewham01.png.transform/2col/image.png",
    "RUS": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/G/GEORUS01_George_Russell/georus01.png.transform/2col/image.png",
    "LEC": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/C/CHALEC01_Charles_Leclerc/chalec01.png.transform/2col/image.png",
    "SAI": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/C/CARSAI01_Carlos_Sainz/carsai01.png.transform/2col/image.png",
    "NOR": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/L/LANNOR01_Lando_Norris/lannor01.png.transform/2col/image.png",
    "PIA": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/O/OSCPIA01_Oscar_Piastri/oscpia01.png.transform/2col/image.png",
    "ALO": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/F/FERALO01_Fernando_Alonso/feralo01.png.transform/2col/image.png",
    "STR": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/L/LANSTR01_Lance_Stroll/lanstr01.png.transform/2col/image.png",
    "GAS": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/P/PIEGAS01_Pierre_Gasly/piegas01.png.transform/2col/image.png",
    "OCO": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/E/ESTOCO01_Esteban_Ocon/estoco01.png.transform/2col/image.png",
    "ALB": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/A/ALEALB01_Alexander_Albon/alealb01.png.transform/2col/image.png",
    "SAR": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/L/LOGSAR01_Logan_Sargeant/logsar01.png.transform/2col/image.png",
    "TSU": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/Y/YUKTSU01_Yuki_Tsunoda/yuktsu01.png.transform/2col/image.png",
    "RIC": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/D/DANRIC01_Daniel_Ricciardo/danric01.png.transform/2col/image.png",
    "BOT": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/V/VALBOT01_Valtteri_Bottas/valbot01.png.transform/2col/image.png",
    "ZHO": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/G/GUAZHO01_Guanyu_Zhou/guazho01.png.transform/2col/image.png",
    "HUL": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/N/NICHUL01_Nico_Hulkenberg/nichul01.png.transform/2col/image.png",
    "MAG": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/K/KEVMAG01_Kevin_Magnussen/kevmag01.png.transform/2col/image.png",
    "DEFAULT": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/driver-fallback-image.png"
}

# ==============================================================================
# VIEW 1: THE GRID (MASTER VIEW)
# ==============================================================================
if st.session_state.selected_driver is None:
    # --- CONTROLS ---
    years = sorted(df['Year'].unique(), reverse=True)
    selected_year = st.selectbox("Select Season", years)

    # Filter Data
    season_df = df[df['Year'] == selected_year]

    # Aggregate Stats
    driver_stats = season_df.groupby('Driver').agg(
        Points=('Points', 'sum'),
        Wins=('Position', lambda x: (x==1).sum()),
        Podiums=('Position', lambda x: (x<=3).sum()),
        Team=('TeamName', 'last'),
        Best_Finish=('Position', 'min')
    ).reset_index().sort_values('Points', ascending=False)

    # --- GRID LAYOUT ---
    cols_per_row = 4
    rows = [st.columns(cols_per_row) for _ in range(len(driver_stats) // cols_per_row + 1)]

    for i, (index, row) in enumerate(driver_stats.iterrows()):
        driver_code = row['Driver']
        team_name = row['Team']
        team_color = TEAM_COLORS.get(team_name, "#333333")
        img_url = HEADSHOTS.get(driver_code, HEADSHOTS["DEFAULT"])
        
        row_idx = i // cols_per_row
        col_idx = i % cols_per_row
        
        with rows[row_idx][col_idx]:
            with st.container(border=True):
                # Color Strip
                st.markdown(f"""<div style="background-color: {team_color}; height: 5px; width: 100%; border-radius: 5px;"></div>""", unsafe_allow_html=True)
                
                # Image & Header
                c1, c2 = st.columns([1, 2])
                with c1: st.image(img_url, use_container_width=True)
                with c2: 
                    st.subheader(driver_code)
                    st.caption(team_name)
                
                # Stats
                st.markdown("---")
                m1, m2, m3 = st.columns(3)
                m1.metric("Pts", int(row['Points']))
                m2.metric("Win", row['Wins'])
                m3.metric("Pod", row['Podiums'])
                
                # NAVIGATION BUTTON (The "Click" Action)
                if st.button("View Profile", key=f"btn_{driver_code}", use_container_width=True):
                    st.session_state.selected_driver = driver_code
                    st.rerun()

# ==============================================================================
# VIEW 2: DRIVER DETAIL (DRILL DOWN)
# ==============================================================================
else:
    # --- HEADER & NAVIGATION ---
    col_back, col_title = st.columns([1, 5])
    with col_back:
        if st.button("← Back to Grid"):
            st.session_state.selected_driver = None
            st.rerun()
    
    s_driver = st.session_state.selected_driver
    
    # Get Driver Info
    driver_df = df[df['Driver'] == s_driver].copy()
    
    # Header Section with Image
    col_img, col_stats = st.columns([1, 3])
    with col_img:
        img_url = HEADSHOTS.get(s_driver, HEADSHOTS["DEFAULT"])
        st.image(img_url, width=200)
    
    with col_stats:
        st.header(f"{s_driver} Career Overview")
        
        # Career Totals
        total_races = driver_df['RoundNumber'].count()
        total_points = driver_df['Points'].sum()
        total_wins = len(driver_df[driver_df['Position'] == 1])
        total_podiums = len(driver_df[driver_df['Position'] <= 3])
        avg_finish = driver_df['Position'].mean()
        
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Races", total_races)
        m2.metric("Points", int(total_points))
        m3.metric("Wins", total_wins)
        m4.metric("Podiums", total_podiums)
        m5.metric("Avg Finish", f"P{avg_finish:.1f}")

    st.divider()

# --- 1. HEAD TO HEAD (Points vs Teammate + Championship Position) ---
    st.subheader("⚔️ Head-to-Head: Driver vs. Teammate")
    
    comparison_data = []
    seasons_active = sorted(driver_df['Year'].unique())
    
    # Calculate stats for each season
    for year in seasons_active:
        # 1. Driver Points
        d_season = driver_df[driver_df['Year'] == year]
        d_points = d_season['Points'].sum()
        team_name = d_season['TeamName'].iloc[0] if not d_season.empty else "Unknown"
        
        # 2. Teammate Points
        team_season = df[(df['Year'] == year) & (df['TeamName'] == team_name)]
        teammate_season = team_season[team_season['Driver'] != s_driver]
        
        tm_points = 0
        if not teammate_season.empty:
            tm_points = teammate_season['Points'].sum()
            # Normalize for multi-driver seats (e.g. Red Bull 2019 Gasly/Albon)
            # We assume 1 teammate car. If 2 drivers shared it, we sum their points.
            # If 3 drivers (rare), we just sum. This is the "Other Car" total.
            
        # 3. Calculate Championship Position (The Line Chart)
        # Get total points for EVERY driver that year
        full_season_standings = df[df['Year'] == year].groupby('Driver')['Points'].sum().sort_values(ascending=False).reset_index()
        # Find where our driver ranked (Index + 1)
        try:
            rank = full_season_standings[full_season_standings['Driver'] == s_driver].index[0] + 1
        except:
            rank = None

        comparison_data.append({
            'Year': year, 
            'Driver Points': d_points, 
            'Teammate Points': tm_points, 
            'Team': team_name,
            'Season Rank': rank
        })
            
    comp_df = pd.DataFrame(comparison_data)
    
    if not comp_df.empty:
        # Create Combo Chart
        fig = go.Figure()

        # Bar: Driver Points
        fig.add_trace(go.Bar(
            x=comp_df['Year'], 
            y=comp_df['Driver Points'],
            name=s_driver,
            marker_color='#1f77b4' # Blue
        ))

        # Bar: Teammate Points
        fig.add_trace(go.Bar(
            x=comp_df['Year'], 
            y=comp_df['Teammate Points'],
            name="Teammate(s)",
            marker_color='#ff7f0e' # Orange
        ))

        # Line: Championship Position (Secondary Axis)
        fig.add_trace(go.Scatter(
            x=comp_df['Year'],
            y=comp_df['Season Rank'],
            name="Season Finish (P)",
            mode='lines+markers',
            line=dict(color='white', width=3, dash='dot'),
            marker=dict(size=8, color='white'),
            yaxis='y2' # Link to secondary axis
        ))

        # Layout: Dual Axis Logic
        fig.update_layout(
            title=f"Season Performance: Points vs. Teammate & Final Standing",
            xaxis=dict(title="Season", type='category'), # Treat years as categories so they don't get decimals (2021.5)
            yaxis=dict(title="Points Scored", side='left', showgrid=False),
            yaxis2=dict(
                title="Championship Finish (Pos)", 
                side='right', 
                overlaying='y', # Layer on top of primary y
                autorange="reversed", # P1 at top, P20 at bottom
                showgrid=False,
                tickmode='linear', # Show integers
                dtick=1
            ),
            barmode='group',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

    # --- 2. TROPHY CABINET ---
    col_trophy, col_consist = st.columns([1, 1])
    
    with col_trophy:
        if total_wins > 0:
            st.subheader(f"🏆 Trophy Cabinet ({total_wins} Wins)")
            wins = driver_df[driver_df['Position'] == 1][['Year', 'EventName', 'TeamName', 'GridPosition']].sort_values('Year', ascending=False)
            st.dataframe(wins, hide_index=True, use_container_width=True)
        else:
            st.subheader("🏆 Trophy Cabinet")
            st.info("No wins yet.")

    # --- 3. CONSISTENCY ---
    with col_consist:
        st.subheader("Consistency Analysis")
        fig_hist = px.histogram(driver_df, x='Position', nbins=20, 
                                title="Finishing Position Distribution",
                                labels={'Position': 'Race Result'},
                                color_discrete_sequence=['#2ca02c'])
        fig_hist.update_layout(bargap=0.1)
        st.plotly_chart(fig_hist, use_container_width=True)
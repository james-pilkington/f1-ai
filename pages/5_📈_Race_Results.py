

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

st.title("📊 Race Results")

# --- B. BY RACE ---
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

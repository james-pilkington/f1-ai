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

st.title("🛠️ Track Analysis")

# --- C. BY TRACK ---
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
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import fastf1
from utils import setup_app, load_data, get_schedule, TEAM_COLORS
from datetime import datetime

setup_app()
data_store = load_data()
df = data_store['history']

st.title("📊 F1 Intelligence Hub")

if df.empty:
    st.error("No Data.")
    st.stop()

# Use Tabs instead of Radio for a cleaner look
tab1, tab2, tab3, tab4 = st.tabs(["By Season", "By Race", "By Track", "By Driver"])

with tab1:
    # ... Paste your "By Season" logic here ...
    # Ensure you use TEAM_COLORS imported from utils
    pass

with tab2:
    # ... Paste your "By Race" logic here ...
    pass

with tab3:
    # ... Paste your "By Track" logic here ...
    pass

with tab4:
    # ... Paste your "By Driver" logic here ...
    pass
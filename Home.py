import streamlit as st
import pandas as pd
import numpy as np
import fastf1
from datetime import datetime, timedelta
# Import from your new utils file
from utils import setup_app, load_data, load_model, get_next_race, get_weekend_status

# Run setup
setup_app()

# Load Data
data_store = load_data()
artifacts = load_model()

# Sidebar
st.sidebar.title("🏎️ F1 AI Strategist")
st.sidebar.info("Navigation: Use the sidebar menu to switch pages.")
if artifacts:
    st.sidebar.success(f"🧠 Model Active\nMAE: ±{artifacts['mae']:.2f}")

# --- MAIN ORACLE LOGIC ---
st.title("🏁 Race Weekend Command Center")
next_race, season = get_next_race()

if next_race is not None:
    st.subheader(f"📍 {next_race['EventName']} ({season})")
    
    status, _ = get_weekend_status(season, next_race['RoundNumber'])
    cols = st.columns(5)
    for i, (k, v) in enumerate(status.items()):
        state = v['state']
        if state == 'Complete':
            cols[i].success(f"**{k}**\n\n✅ Done")
        elif state == 'N/A':
            cols[i].write(f"**{k}**\n\n⚪ --")
        else:
            d_str = v['date'].strftime('%d %b %H:%M')
            cols[i].warning(f"**{k}**\n\n⏳ {d_str}")
            
    st.divider()
    
    # Prediction Interface (Paste the rest of your Oracle logic here...)
    # [Use the exact code from your previous app.py for the prediction section]
    # Just remember to use `data_store` and `artifacts` which we loaded above.

else:
    st.success("Season Complete!")
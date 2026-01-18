import streamlit as st
import pandas as pd
from utils import setup_app, load_model, get_next_race, get_weekend_status

# --- APP SETUP ---
setup_app()
st.set_page_config(page_title="F1 AI Strategist", page_icon="🏎️", layout="wide")

# --- SIDEBAR: MODEL STATUS ---
# Note: In "Automatic" mode, this sidebar content only shows when you are on the Home page.
# (To show it everywhere, you'd add this snippet to every page).
artifacts = load_model()

st.sidebar.title("🏎️ F1 AI Strategist")
st.sidebar.info("Select a tool from the menu above to begin.")

if artifacts:
    st.sidebar.divider()
    st.sidebar.success("🧠 AI Model Online")
    
    # Retrieve Segmented Metrics
    mae_global = artifacts.get('mae_global', artifacts.get('mae', 0))
    mae_top10 = artifacts.get('mae_top10', 0)
    mae_clean = artifacts.get('mae_clean', 0)
    
    st.sidebar.markdown("### Model Accuracy")
    c1, c2 = st.sidebar.columns(2)
    c1.metric("Global", f"±{mae_global:.1f}")
    c2.metric("Top 10", f"±{mae_top10:.1f}")
    st.sidebar.caption(f"Clean MAE: ±{mae_clean:.1f}")

# --- MAIN PAGE CONTENT ---
st.title("🏁 F1 Strategy Command Center")

# Quick "Next Race" Summary for the Landing Page
next_race, season = get_next_race()

if next_race is not None:
    st.subheader(f"Next Event: {next_race['EventName']} ({season})")
    
    # Simple Countdown / Status
    status, _ = get_weekend_status(season, next_race['RoundNumber'])
    
    # Create a nice visual row of session statuses
    cols = st.columns(5)
    sessions = ["FP1", "FP2", "FP3", "Quali", "Race"]
    
    for i, sess in enumerate(sessions):
        with cols[i]:
            if sess in status:
                state = status[sess]['state']
                date_str = status[sess]['date'].strftime('%d %b %H:%M')
                
                if state == 'Complete':
                    st.success(f"**{sess}**\n\n✅")
                elif state == 'N/A':
                    st.write(f"**{sess}**\n\n--")
                else:
                    st.info(f"**{sess}**\n\n📅 {date_str}")

    st.divider()
    st.markdown("""
    ### 🚀 Available Tools
    
    * **Race Oracle:** Predict Qualifying grids using live Practice data.
    * **Scenario Simulator:** "What if" analysis for specific driver conditions.
    * **Stats Dashboard:** Deep dive into historical season data.
    * **Driver & Team Grids:** Career analysis and head-to-head comparisons.
    """)
    
    if st.button("Go to Race Oracle →"):
        st.switch_page("pages/1_ai_predictions.py")

else:
    st.success("Season Complete! Use the Historical Data tools to analyze the past season.")
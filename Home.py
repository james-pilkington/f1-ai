import streamlit as st
import pandas as pd
import os
from utils import setup_app, load_model, get_next_race, get_weekend_status

# --- APP SETUP ---
setup_app()
st.set_page_config(page_title="F1 AI Strategist", page_icon="🏎️", layout="wide")

# --- LOAD MODELS ---
# 1. Quali Model (Standard Loader)
quali_artifacts = load_model()

# 2. Race Model (Custom Loader)
@st.cache_resource
def load_race_model():
    try:
        import pickle
        if os.path.exists('data/race_model.pkl'):
            with open('data/race_model.pkl', 'rb') as f:
                return pickle.load(f)
    except: return None
    return None

race_artifacts = load_race_model()

# --- SIDEBAR: INTELLIGENCE REPORT ---
st.sidebar.title("🏎️ F1 AI Strategist")
st.sidebar.info("Select a tool from the menu above to begin.")

st.sidebar.divider()

# A. QUALIFYING STATS
if quali_artifacts:
    st.sidebar.subheader("🔮 Qualifying AI")
    
    q_global = quali_artifacts.get('mae_global', quali_artifacts.get('mae', 0))
    q_top10 = quali_artifacts.get('mae_top10', 0)
    q_clean = quali_artifacts.get('mae_clean', 0)
    
    c1, c2 = st.sidebar.columns(2)
    c1.metric("Global", f"±{q_global:.1f}")
    c2.metric("Top 10", f"±{q_top10:.1f}")
    
    st.sidebar.caption(f"**Clean MAE: ±{q_clean:.1f}**")
    st.sidebar.caption("*(Excludes anomalies > 8 spots)*")
else:
    st.sidebar.warning("Quali Model Offline")

st.sidebar.divider()

# B. RACE STATS
if race_artifacts:
    st.sidebar.subheader("🏁 Race AI")
    
    r_global = race_artifacts.get('mae_global', 0)
    # Check if split stats exist (requires updated train_race_model.py)
    r_front = race_artifacts.get('mae_front', 0) 
    r_back = race_artifacts.get('mae_back', 0)
    r_clean = race_artifacts.get('mae_clean', 0)
    
    # Detailed Split
    if r_front > 0:
        st.sidebar.write("**Grid Split Accuracy**")
        c5, c6 = st.sidebar.columns(2)
        c5.metric("Front Grid", f"±{r_front:.1f}", help="Drivers starting P1-P10")
        c6.metric("Back Grid", f"±{r_back:.1f}", help="Drivers starting P11+")
    
    st.sidebar.caption(f"*(Clean excludes DNFs)*")
else:
    st.sidebar.warning("Race Model Offline")

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
    
    if st.button("Go to Race Oracle →", type="primary"):
        st.switch_page("pages/1_ai_predictions.py")

else:
    st.success("Season Complete! Use the Historical Data tools to analyze the past season.")
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
def _load_race_model_cached(_mtime):
    # _mtime busts the cache whenever the pkl on disk changes (e.g. a new
    # deploy) - cache_resource otherwise keys only on the function itself
    # and would keep serving a stale (or previously-failed) result forever.
    try:
        import pickle
        if os.path.exists('data/race_model.pkl'):
            with open('data/race_model.pkl', 'rb') as f:
                return pickle.load(f)
    except: return None
    return None

def load_race_model():
    try:
        mtime = os.path.getmtime('data/race_model.pkl')
    except OSError:
        mtime = None
    return _load_race_model_cached(mtime)

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

                if state == 'Complete':
                    st.success(f"**{sess}**\n\n✅")
                elif state == 'N/A':
                    st.write(f"**{sess}**\n\n--")
                else:
                    date_str = status[sess]['date'].strftime('%d %b %H:%M')
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
        st.switch_page("pages/1_✨_AI_Predictions.py")

    st.divider()
    with st.expander("🧠 How the AI models work — features & methodology"):
        st.markdown("""
Both models are retrained from scratch each time new race data comes in — nothing here is hand-tuned per race, it's whatever the training script fits from the numbers. Current accuracy is in the sidebar; this is *how* it gets there.
""")

        col_q, col_r = st.columns(2)

        with col_q:
            st.markdown("""
#### 🔮 Qualifying AI
Predicts final grid position from Friday/Saturday practice pace.

**Algorithm:** Gradient Boosting Regressor — tuned with a randomized search over 20 combinations of tree count, depth, learning rate and subsampling, picked by 3-fold cross-validation on mean absolute error.

**Features:**
- `FP_Pos` / `FP_Gap` — final practice rank and time gap to the fastest car
- `Teammate_Delta_Gap` — pace vs. teammate in the identical car, to separate driver skill from car performance
- `Form_Last3` — **median** (not mean, so one crash doesn't skew it) qualifying position over the last 3 races
- `Driver_Track_Avg` — that driver's own history at this specific circuit
- `Driver_Rating` / `Team_Rating` — career-long average qualifying position, used as a baseline
- `Track_Type_Code` — street circuit vs. permanent circuit

**Target:** actual qualifying position, clipped to a legal 1–20 range.
""")

        with col_r:
            st.markdown("""
#### 🏁 Race Strategy Oracle
Three separate models sharing one feature set, each answering a different question.

**Models:**
- **Pace** — HistGradientBoostingRegressor → predicted finishing position
- **DNF Risk** — RandomForestClassifier → probability of retirement
- **Big Mover** — HistGradientBoostingClassifier → will this driver gain 3+ places?

**Shared features:**
- `Grid_Pos`, `Car_Potential` (grid gap to the team's best-placed car that weekend), `Teammate_Delta_Grid`
- `Quali_Gap_Pct` — % off pole, more informative than raw position when the field is bunched up or spread out
- `Season_Avg_Grid`, `Form_Last5_Finish`, `Quali_Volatility` — rolling season form and consistency
- `Avg_Pos_Gained` — rolling 10-race average of grid→finish gain ("racecraft")
- `Team_Pit_Speed` — team's season-to-date average pit stop duration
- `Track_Code`, `Downforce_Code`, `Is_Rain` — circuit identity and conditions
""")

        st.markdown("""
**On avoiding lookahead bias:** every rolling/season-average feature above (`Form_Last3`, `Season_Avg_Grid`, `Quali_Volatility`, `Avg_Pos_Gained`, `Team_Pit_Speed`, `Driver_Track_Avg`) is computed with a one-race lag — a driver's "form" going into Round 5 only ever sees Rounds 1–4. None of these features can see the result they're trying to predict.

Training data currently spans every qualifying and race session from the 2023 season through the most recently completed round.
""")

else:
    st.success("Season Complete! Use the Historical Data tools to analyze the past season.")
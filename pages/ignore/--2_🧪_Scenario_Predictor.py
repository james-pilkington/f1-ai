import streamlit as st
import numpy as np
from utils import setup_app, load_data, load_model

setup_app()
data_store = load_data()
artifacts = load_model()

st.title("🧪 Scenario Simulator")

df_train = data_store['training']
if not df_train.empty:
    c1, c2 = st.columns(2)
    with c1:
        drv = st.selectbox("Driver", sorted(df_train['Driver'].unique()))
        trk = st.selectbox("Track", sorted(df_train['EventName'].unique()))
    with c2:
        fp = st.slider("FP3 Pos", 1, 20, 1)
        tm = st.slider("Teammate FP3", 1, 20, 5)
        
    if st.button("Simulate"):
        if artifacts:
            # Encoding logic from utils/artifacts
            d_c = artifacts['le_driver'].transform([drv])[0] if drv in artifacts['le_driver'].classes_ else 0
            t_c = artifacts['le_track'].transform([trk])[0] if trk in artifacts['le_track'].classes_ else 0
            avg = (fp+tm)/2
            pred = artifacts['model'].predict(np.array([[fp, avg, fp-avg, d_c, t_c]]))[0]
            st.metric("Predicted Grid", f"P{int(pred)}")
        else:
            st.error("Model missing")
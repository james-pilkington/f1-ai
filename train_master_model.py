import pandas as pd
import numpy as np
import pickle
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, accuracy_score, precision_score, classification_report, log_loss

# --- CONFIG ---
DATA_PATH = 'data/race_data_master.parquet'
MODEL_PATH = 'data/race_model.pkl'

def train_suite():
    print("🚀 INITIALIZING F1 STRATEGY AI TRAINING SUITE...")
    start_time = time.time()
    
    # 1. LOAD DATA
    try:
        df = pd.read_parquet(DATA_PATH)
        print(f"   📂 Loaded {len(df)} rows from Feature Store.")
    except Exception as e:
        print(f"   ❌ Error loading data: {e}")
        return

    # --- PREPROCESSING & FEATURE ENGINEERING ---
    # We create a "Clean" dataset for the Regression model (No DNFs)
    # And a "Full" dataset for the DNF model
    
    # 1. Encode Categorical Features
    le_track = LabelEncoder()
    df['Track_Code'] = le_track.fit_transform(df['EventName'])
    
    le_type = LabelEncoder()
    df['Track_Type_Code'] = le_type.fit_transform(df['Track_Type'])
    
    # Downforce/Overtake might be strings, encode them
    le_downforce = LabelEncoder()
    df['Downforce_Code'] = le_downforce.fit_transform(df['Track_Downforce'].astype(str))
    
    # 2. Target Encoding for Drivers/Teams (Using Clean Data to avoid DNF bias)
    df_clean = df[df['Finish_Pos'] < 20].copy()
    
    driver_pace_map = df_clean.groupby('Driver')['Finish_Pos'].mean().to_dict()
    team_pace_map = df_clean.groupby('TeamName')['Finish_Pos'].mean().to_dict()
    
    # Apply Mappings
    df['Driver_Rating'] = df['Driver'].map(driver_pace_map).fillna(15)
    df['Team_Rating'] = df['TeamName'].map(team_pace_map).fillna(15)
    
    # 3. Define The Feature Set (The "Mega" List)
    features = [
        'Grid_Pos',
        'Car_Potential',        # Grid - Team Best
        'Teammate_Delta_Grid',
        'Quali_Gap_Pct',        # Normalized Speed
        'Season_Avg_Grid',
        'Form_Last5_Finish',
        'Quali_Volatility',     # Consistency
        'Avg_Pos_Gained',       # Racecraft
        'Team_Pit_Speed',       # Pit Crew
        'Track_Code',
        'Downforce_Code',
        'Is_Rain'
    ]
    
    # Check for missing columns (in case older data generation)
    missing_cols = [c for c in features if c not in df.columns]
    if missing_cols:
        print(f"   ⚠️ Warning: Missing features {missing_cols}. Filling with 0.")
        for c in missing_cols: df[c] = 0

    # Fill NaNs in features
    df[features] = df[features].fillna(0)

    print("-" * 60)
    
    # ==============================================================================
    # MODEL A: THE STRATEGIST (Finish Position Regressor)
    # ==============================================================================
    print("🤖 TRAINING MODEL A: RACE PACE REGRESSOR...")
    
    # Filter: Only finishers
    df_reg = df[df['Finish_Pos'] < 20].copy()
    X_reg = df_reg[features]
    y_reg = df_reg['Finish_Pos']
    
    X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    
    model_a = HistGradientBoostingRegressor(
        learning_rate=0.04, max_iter=500, max_depth=5, 
        l2_regularization=0.1, random_state=42
    )
    model_a.fit(X_train_A, y_train_A)
    
    # Evaluation A
    preds_a = model_a.predict(X_test_A)
    preds_a = np.clip(preds_a, 1, 20)
    mae_a = mean_absolute_error(y_test_A, preds_a)
    
    # Segmented MAE
    test_df_A = X_test_A.copy()
    test_df_A['Actual'] = y_test_A
    test_df_A['Pred'] = preds_a
    test_df_A['Error'] = np.abs(test_df_A['Actual'] - test_df_A['Pred'])
    
    mae_front = test_df_A[test_df_A['Grid_Pos'] <= 10]['Error'].mean()
    mae_back = test_df_A[test_df_A['Grid_Pos'] > 10]['Error'].mean()
    
    print(f"   ✅ Model A Accuracy:")
    print(f"      🌍 Global Clean MAE: ±{mae_a:.2f}")
    print(f"      🏎️ Front Grid MAE:  ±{mae_front:.2f}")
    print(f"      🎢 Back Grid MAE:   ±{mae_back:.2f}")

    # ==============================================================================
    # MODEL B: THE RISK MANAGER (DNF Probability)
    # ==============================================================================
    print("\n🤖 TRAINING MODEL B: DNF PREDICTOR...")
    
    # Target: 1 if Finish_Pos == 20, else 0
    df['Target_DNF'] = (df['Finish_Pos'] >= 20).astype(int)
    
    X_cls = df[features]
    y_cls = df['Target_DNF']
    
    X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls)
    
    # Using Random Forest for Probability calibration
    model_b = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight='balanced', random_state=42)
    model_b.fit(X_train_B, y_train_B)
    
    # Evaluation B
    preds_b_proba = model_b.predict_proba(X_test_B)[:, 1] # Probability of DNF
    loss_b = log_loss(y_test_B, preds_b_proba)
    
    print(f"   ✅ Model B Stats:")
    print(f"      📉 Log Loss: {loss_b:.3f} (Lower is better)")
    print(f"      ⚠️ Average DNF Probability predicted: {preds_b_proba.mean():.1%}")

    # ==============================================================================
    # MODEL C: THE GAMBLER (Big Mover Classifier)
    # ==============================================================================
    print("\n🤖 TRAINING MODEL C: 'BIG MOVER' CLASSIFIER...")
    
    # Target: 1 if Gained > 2 spots, 0 otherwise
    # We use 'Positions_Gained' from the master file or calc it
    if 'Positions_Gained' not in df.columns:
        df['Positions_Gained'] = df['Grid_Pos'] - df['Finish_Pos']
        
    df['Target_BigMove'] = (df['Positions_Gained'] >= 3).astype(int)
    
    # Filter to Clean races only (Don't reward DNFs of others as "Skill")
    df_move = df[df['Finish_Pos'] < 20].copy()
    X_move = df_move[features]
    y_move = df_move['Target_BigMove']
    
    X_train_C, X_test_C, y_train_C, y_test_C = train_test_split(X_move, y_move, test_size=0.2, random_state=42, stratify=y_move)
    
    model_c = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_depth=4, random_state=42)
    model_c.fit(X_train_C, y_train_C)
    
    preds_c = model_c.predict(X_test_C)
    prec_c = precision_score(y_test_C, preds_c, zero_division=0)
    acc_c = accuracy_score(y_test_C, preds_c)
    
    print(f"   ✅ Model C Stats:")
    print(f"      🎯 Accuracy:  {acc_c:.1%}")
    print(f"      ⚡ Precision: {prec_c:.1%} (When it says 'Big Move', is it right?)")

    # ==============================================================================
    # SAVING THE SUITE
    # ==============================================================================
    print("-" * 60)
    
    artifacts = {
        'model_pace': model_a,      # The Main Prediction
        'model_dnf': model_b,       # The Risk Warning
        'model_move': model_c,      # The "Hot Tip"
        'driver_map': driver_pace_map,
        'team_map': team_pace_map,
        'le_track': le_track,
        'le_type': le_type,
        'le_downforce': le_downforce,
        'features': features,
        'mae_global': mae_a,
        'mae_front': mae_front,
        'mae_back': mae_back
    }
    
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(artifacts, f)
        
    print(f"💾 ALL MODELS SAVED TO {MODEL_PATH}")
    print(f"⏱️ Total Training Time: {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    train_suite()
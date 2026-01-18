import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

def train_race_model():
    print("🚀 Starting RACE Model Training (V3 - Expert)...")
    
    try:
        df = pd.read_parquet('data/race_training_data.parquet')
        print(f"   Loaded {len(df)} rows.")
    except FileNotFoundError:
        print("❌ Data not found. Run generate_race_features.py first.")
        return

    # --- 1. FILTER DNFs (The "Pure Pace" Logic) ---
    # We remove any row where Finish_Pos is 20 (our DNF marker).
    # This prevents the model from learning that "Max Verstappen sometimes finishes P20 for no reason."
    # We strictly want to know: "If the car runs, how fast is it?"
    n_before = len(df)
    df_clean = df[df['Finish_Pos'] < 20].copy()
    print(f"   🧹 Filtered out {n_before - len(df_clean)} DNF rows for training.")
    
    # --- 2. ENCODING ---
    # Create Ratings based on this CLEAN data (Pure Pace Ratings)
    driver_map = df_clean.groupby('Driver')['Finish_Pos'].mean().to_dict()
    team_map = df_clean.groupby('TeamName')['Finish_Pos'].mean().to_dict()
    
    # Create Chaos Map for the UI (Using the FULL dataset including crashes)
    # We want to know the % chance of DNF for the UI warning
    chaos_map = df.groupby('EventName')['Chaos_Factor'].mean().to_dict()
    
    df_clean['Driver_Rating'] = df_clean['Driver'].map(driver_map)
    df_clean['Team_Rating'] = df_clean['TeamName'].map(team_map)
    
    le_track = LabelEncoder()
    df_clean['Track_Code'] = le_track.fit_transform(df_clean['EventName'])
    
    le_type = LabelEncoder()
    df_clean['Track_Type_Code'] = le_type.fit_transform(df_clean['Track_Type'])

    # --- 3. ADVANCED FEATURE ENGINEERING ---
    
    # A. Recovery Potential (Out of Position)
    # Logic: If I start P15, but my Team Rating is P4, my delta is +11.
    # High Positive Number = High Incentive/Ability to Overtake.
    df_clean['Recovery_Potential'] = df_clean['Grid_Pos'] - df_clean['Team_Rating']
    
    # B. Grid Lock (Impossibility Factor)
    # Logic: Grid Position weighted by Difficulty.
    # P15 at Monaco (Diff 0.9) -> 15 * 0.9 = 13.5 (Very High Lock)
    # P15 at Spa (Diff 0.4)    -> 15 * 0.4 = 6.0 (Low Lock)
    df_clean['Grid_Lock'] = df_clean['Grid_Pos'] * df_clean['Overtaking_Difficulty']

    features = [
        'Grid_Pos',             
        'Recovery_Potential',   # NEW: The "Anger/Motivation" metric
        'Grid_Lock',            # NEW: The "Monaco" metric
        'Teammate_Delta_Grid',  
        'Form_Last3_Race',      
        'Driver_Track_Avg_Race',
        'Overtaking_Difficulty',
        'Chaos_Factor',         
        'Is_Rain',              
        'Driver_Rating',        
        'Team_Rating',          
        'Track_Type_Code'
    ]
    target = 'Finish_Pos'

    # Filter Missing
    df_clean = df_clean.dropna(subset=features + [target])
    
    X = df_clean[features]
    y = df_clean[target]

    # --- 4. TRAINING ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"   Tuning Model on {len(X_train)} pure-pace examples...")
    
    param_grid = {
        'n_estimators': [200, 400, 600],
        'learning_rate': [0.01, 0.05],
        'max_depth': [3, 4, 5],
        'subsample': [0.8, 0.9],
        'min_samples_leaf': [2, 4]
    }
    
    gb = GradientBoostingRegressor(random_state=42)
    search = RandomizedSearchCV(gb, param_grid, n_iter=15, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    # --- 5. EVALUATION ---
    preds = best_model.predict(X_test)
    preds = np.clip(preds, 1, 20)
    
    # Analysis
    results_df = pd.DataFrame({'Actual': y_test, 'Predicted': preds})
    results_df['Error'] = np.abs(results_df['Actual'] - results_df['Predicted'])
    
    mae = results_df['Error'].mean()
    
    # Split: Front Runners vs Midfield
    mask_front = X_test['Grid_Pos'] <= 10
    mae_front = results_df[mask_front]['Error'].mean()
    mae_back = results_df[~mask_front]['Error'].mean()

    print("-" * 40)
    print(f"✅ PURE PACE MODEL RESULTS:")
    print(f"   🌍 Clean MAE:        ±{mae:.2f} (Speed Prediction Only)")
    print(f"   --------------------------------")
    print(f"   🏎️ Front Grid MAE:   ±{mae_front:.2f}")
    print(f"   🎢 Back Grid MAE:    ±{mae_back:.2f}")
    print("-" * 40)
    
    # Importance
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("\n   🧠 Key Predictors:")
    for i in range(5):
        print(f"   {i+1}. {features[indices[i]]} ({importances[indices[i]]:.1%})")

    # --- 6. SAVE ---
    artifacts = {
        'model': best_model,
        'driver_map': driver_map,
        'team_map': team_map,
        'chaos_map': chaos_map, # Save this for the UI Warning!
        'le_track': le_track,
        'le_type': le_type,
        'features': features,
        'mae_clean': mae
    }
    
    with open('data/race_model.pkl', 'wb') as f:
        pickle.dump(artifacts, f)
    print("💾 Saved race_model.pkl")

if __name__ == "__main__":
    train_race_model()
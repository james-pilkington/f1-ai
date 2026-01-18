import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

def train_quali_model():
    print("🚀 Starting Model Training (V3 - Advanced)...")
    
    # --- 1. Load Data ---
    try:
        df = pd.read_parquet('data/quali_training_data.parquet')
        print(f"   Loaded {len(df)} rows.")
    except FileNotFoundError:
        print("❌ Data not found. Run generate_features.py first.")
        return

    if len(df) == 0:
        print("❌ Error: Training data is empty.")
        return

    # --- 2. INTELLIGENT ENCODING (Target Encoding) ---
    # We map drivers/teams to their average historical performance.
    # This gives the model a mathematical baseline (e.g. Max=1.5, Logan=18.0)
    
    # Calculate Averages (using the whole dataset for the baseline map)
    driver_map = df.groupby('Driver')['Quali_Pos'].mean().to_dict()
    team_map = df.groupby('TeamName')['Quali_Pos'].mean().to_dict()
    
    # Map them to create new numeric features
    df['Driver_Rating'] = df['Driver'].map(driver_map)
    df['Team_Rating'] = df['TeamName'].map(team_map)
    
    # Encode Categories (Track Type needs a standard LabelEncoder)
    le_track = LabelEncoder()
    df['Track_Code'] = le_track.fit_transform(df['EventName'])
    
    le_type = LabelEncoder()
    df['Track_Type_Code'] = le_type.fit_transform(df['Track_Type'])

    # --- 3. Feature Selection ---
    # These must match exactly what you generated in generate_features.py
    features = [
        'FP_Pos',              # Session Performance (Raw Rank)
        'FP_Gap',              # Session Performance (Time Gap)
        'Teammate_Delta_Gap',  # Driver Skill vs Car
        'Form_Last3',          # Recent Momentum (Median)
        'Driver_Track_Avg',    # Course History (Horse for Course)
        'Driver_Rating',       # Long-term Driver Baseline
        'Team_Rating',         # Long-term Team Baseline
        'Track_Type_Code'      # Street vs Circuit
    ]
    target = 'Quali_Pos'

    # Double check features exist
    missing_cols = [f for f in features if f not in df.columns]
    if missing_cols:
        print(f"❌ Missing columns in parquet: {missing_cols}")
        print("   Run generate_features.py with force_rebuild=True")
        return

    X = df[features]
    y = df[target]

    # --- 4. Split Data ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 5. HYPERPARAMETER TUNING ---
    # We let the computer try different "Brain Structures" to find the smartest one.
    print(f"   Tuning hyperparameters on {len(X_train)} rows...")
    
    param_grid = {
        'n_estimators': [100, 200, 300, 400],    # How many trees?
        'learning_rate': [0.01, 0.05, 0.1],      # How fast to learn?
        'max_depth': [3, 4, 5],                  # How complex?
        'subsample': [0.8, 0.9, 1.0],            # Use 80-100% of data (prevents overfitting)
        'min_samples_leaf': [1, 2, 4]            # Minimum size of leaves
    }
    
    gb = GradientBoostingRegressor(random_state=42)
    
    search = RandomizedSearchCV(
        gb, 
        param_grid, 
        n_iter=20, # Try 20 random combinations
        cv=3,      # 3-fold cross-validation
        scoring='neg_mean_absolute_error',
        n_jobs=-1, # Use all CPU cores
        random_state=42
    )
    
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    print(f"   ✅ Best Params: {search.best_params_}")

    # --- 6. ADVANCED EVALUATION ---
    preds = best_model.predict(X_test)
    preds = np.clip(preds, 1, 20) # Enforce physics (cannot finish P0 or P25)
    
    # Create Analysis DataFrame
    results_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': preds,
        'Error': np.abs(y_test - preds)
    })
    
    # A. Global Stats
    mae_global = results_df['Error'].mean()
    r2 = r2_score(y_test, preds)

    # B. Split Stats (Top 10 vs Wild West)
    mask_top10 = results_df['Actual'] <= 10
    mae_top10 = results_df[mask_top10]['Error'].mean()
    mae_bottom10 = results_df[~mask_top10]['Error'].mean()
    
    # C. Clean Stats (Excluding Crashes/Anomalies > 8 spots off)
    crash_threshold = 8
    mask_clean = results_df['Error'] <= crash_threshold
    mae_clean = results_df[mask_clean]['Error'].mean()
    n_crashes = len(results_df) - len(results_df[mask_clean])

    print("-" * 40)
    print(f"✅ DETAILED RESULTS:")
    print(f"   🌍 Global MAE:       ±{mae_global:.2f}")
    print(f"   --------------------------------")
    print(f"   🏎️ Top 10 MAE:       ±{mae_top10:.2f}  (Front of Grid)")
    print(f"   🎢 Mid/Back MAE:     ±{mae_bottom10:.2f}  (The Wild West)")
    print(f"   --------------------------------")
    print(f"   🧹 Clean MAE:        ±{mae_clean:.2f}  (Excluding {n_crashes} likely crashes)")
    print(f"   📊 R² Score:         {r2:.2f}")
    print("-" * 40)

    # --- 7. FEATURE IMPORTANCE ---
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("   🧠 What mattered most?")
    for i in range(min(5, len(features))):
        print(f"   {i+1}. {features[indices[i]]} ({importances[indices[i]]:.1%})")

    # --- 8. SAVE ARTIFACTS ---
    artifacts = {
        'model': best_model,
        'driver_map': driver_map,   # Save the maps!
        'team_map': team_map,
        'le_track': le_track,
        'le_type': le_type,
        'features': features,
        'mae_global': mae_global,
        'mae_top10': mae_top10,
        'mae_clean': mae_clean
    }
    
    with open('data/quali_model.pkl', 'wb') as f:
        pickle.dump(artifacts, f)
    print("\n💾 Saved model & analytics to data/quali_model.pkl")

if __name__ == "__main__":
    train_quali_model()
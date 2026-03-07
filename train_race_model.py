import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

def train_race_model():
    print("🚀 Starting RACE Model Training (V11 - Voting Ensemble)...")
    
    try:
        df = pd.read_parquet('data/race_training_data.parquet')
        print(f"   Loaded {len(df)} rows.")
    except: return

    # Filter DNFs
    df_clean = df[df['Finish_Pos'] < 20].copy()
    
    # Target: Delta
    df_clean['Target_Delta'] = df_clean['Grid_Pos'] - df_clean['Finish_Pos']
    
    # ENCODING
    # Calculate Team Strength (Average Finish)
    team_strength = df_clean.groupby('TeamName')['Finish_Pos'].mean().to_dict()
    
    driver_map = df_clean.groupby('Driver')['Target_Delta'].mean().to_dict()
    team_map = df_clean.groupby('TeamName')['Target_Delta'].mean().to_dict()
    chaos_map = df.groupby('EventName')['Chaos_Factor'].mean().to_dict()
    
    df_clean['Driver_Rating'] = df_clean['Driver'].map(driver_map)
    df_clean['Team_Rating'] = df_clean['TeamName'].map(team_map)
    
    le_track = LabelEncoder()
    df_clean['Track_Code'] = le_track.fit_transform(df_clean['EventName'])
    le_type = LabelEncoder()
    df_clean['Track_Type_Code'] = le_type.fit_transform(df_clean['Track_Type'])

    # --- NEW FEATURE: TIER MISMATCH ---
    # 1. Define "Top Team" (Avg Finish < 8)
    # 2. Define "Back Grid" (Start > 12)
    # Mismatch = Top Team starting in Back Grid
    
    def get_tier_score(row):
        strength = team_strength.get(row['TeamName'], 10)
        grid = row['Grid_Pos']
        
        # Rocket Ship: Strong Team (Avg < 6) starting Low (> 10)
        if strength < 6 and grid > 10:
            return 1 # High gain expected
            
        # Falling Stone: Weak Team (Avg > 12) starting High (< 8)
        if strength > 12 and grid < 8:
            return -1 # High drop expected
            
        return 0 # Neutral
    
    df_clean['Tier_Mismatch'] = df_clean.apply(get_tier_score, axis=1)

    # Standard Features
    df_clean['Car_Potential'] = df_clean['Grid_Pos'] - df_clean['Team_Best_Grid']
    df_clean['Season_Delta'] = df_clean['Grid_Pos'] - df_clean['Season_Avg_Grid']
    df_clean['Overtake_Opportunity'] = df_clean['Car_Potential'] * (1 - df_clean['Overtaking_Difficulty'])

    features = [
        'Grid_Pos',
        'Tier_Mismatch',        # <-- NEW: The "Rocket Ship" flag
        'Car_Potential',        
        'Season_Delta',         
        'Overtake_Opportunity', 
        'Teammate_Delta_Grid',
        'Team_Best_Grid',
        'Driver_Rating',
        'Team_Rating',
        'Chaos_Factor',
        'Is_Rain',
        'Track_Type_Code'
    ]
    target = 'Target_Delta'

    df_clean = df_clean.dropna(subset=features + [target])
    X = df_clean[features]
    y = df_clean[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"   Training Ensemble on {len(X_train)} rows...")
    
    # --- MODEL 1: The Pattern Matcher (Gradient Boosting) ---
    model_gb = HistGradientBoostingRegressor(
        learning_rate=0.04, max_iter=600, max_depth=5, l2_regularization=0.1, random_state=42
    )
    model_gb.fit(X_train, y_train)
    
    # --- MODEL 2: The Rule Maker (Random Forest) ---
    # Random Forest is excellent at capturing "If X then Y" distinct rules (like our Tier Mismatch)
    model_rf = RandomForestRegressor(
        n_estimators=300, max_depth=8, min_samples_leaf=4, random_state=42
    )
    model_rf.fit(X_train, y_train)
    
    # --- VOTING ---
    # We trust the Gradient Booster slightly more for nuance
    pred_gb = model_gb.predict(X_test)
    pred_rf = model_rf.predict(X_test)
    
    pred_delta = (0.6 * pred_gb) + (0.4 * pred_rf)
    
    pred_finish = X_test['Grid_Pos'] - pred_delta
    pred_finish = np.clip(pred_finish, 1, 20)
    
    actual_finish = X_test['Grid_Pos'] - y_test
    
    results = pd.DataFrame({'Actual': actual_finish, 'Predicted': pred_finish, 'Grid': X_test['Grid_Pos']})
    results['Error'] = np.abs(results['Actual'] - results['Predicted'])
    
    mae_global = results['Error'].mean()
    mae_front = results[results['Grid'] <= 10]['Error'].mean()
    mae_back = results[results['Grid'] > 10]['Error'].mean()
    
    print("-" * 40)
    print(f"✅ VOTING ENSEMBLE RESULTS:")
    print(f"   🌍 Clean MAE:        ±{mae_global:.2f}")
    print(f"   🏎️ Front Grid MAE:   ±{mae_front:.2f}")
    print(f"   🎢 Back Grid MAE:    ±{mae_back:.2f}")
    print("-" * 40)

    # Wrapper for UI
    class VotingModel:
        def __init__(self, gb, rf):
            self.gb = gb
            self.rf = rf
            
        def predict(self, X):
            p1 = self.gb.predict(X)
            p2 = self.rf.predict(X)
            return (0.6 * p1) + (0.4 * p2)

    final_model = VotingModel(model_gb, model_rf)

    artifacts = {
        'model': final_model,
        'driver_map': driver_map, 'team_map': team_map, 'chaos_map': chaos_map,
        'team_strength': team_strength, # Need this for the UI calculation!
        'le_track': le_track, 'le_type': le_type,
        'features': features,
        'mae_global': mae_global, 'mae_clean': mae_global,
        'mae_front': mae_front, 'mae_back': mae_back,
        'is_delta_model': True
    }
    
    with open('data/race_model.pkl', 'wb') as f:
        pickle.dump(artifacts, f)
    print("💾 Saved race_model.pkl")

if __name__ == "__main__":
    train_race_model()
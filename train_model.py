import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

def train_quali_model():
    print("Loading feature data...")
    try:
        df = pd.read_parquet('data/quali_training_data.parquet')
    except FileNotFoundError:
        print("Error: data/quali_training_data.parquet not found. Run generate_features.py first.")
        return

    # --- 1. Prepare Encoders ---
    # We need to turn names into numbers
    le_driver = LabelEncoder()
    df['Driver_Code'] = le_driver.fit_transform(df['Driver'])
    
    le_team = LabelEncoder()
    df['Team_Code'] = le_team.fit_transform(df['TeamName'])
    
    le_track = LabelEncoder()
    df['Track_Code'] = le_track.fit_transform(df['EventName'])

    # --- 2. Select Features ---
    # The MVP 'Signals'
    features = [
        'FP_Pos',              # Raw speed in practice
        'Team_Avg_FP_Pos',     # Car performance proxy
        'Teammate_Delta_Pos',  # Driver skill relative to car
        'Driver_Code',         # Driver skill baseline
        'Track_Code'           # Track specificity
    ]
    target = 'Quali_Pos'

    X = df[features]
    y = df[target]

    # --- 3. Split & Train ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Gradient Boosting model (better for ranking)...")
    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    # --- 4. Evaluate ---
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print(f"✅ Model Results:")
    print(f"   MAE: {mae:.2f} (On average, we are off by {mae:.2f} grid spots)")
    print(f"   R2 Score: {r2:.2f}")

    # --- 5. Save ---
    artifacts = {
        'model': model,
        'le_driver': le_driver,
        'le_team': le_team,
        'le_track': le_track,
        'features': features,
        'mae': mae
    }
    
    with open('data/quali_model.pkl', 'wb') as f:
        pickle.dump(artifacts, f)
    print("Saved model to data/quali_model.pkl")

if __name__ == "__main__":
    train_quali_model()
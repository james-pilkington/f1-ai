import fastf1
import pandas as pd
import os
import numpy as np
import time

# Create cache
if not os.path.exists('f1_cache'): os.makedirs('f1_cache')
fastf1.Cache.enable_cache('f1_cache')

DATA_FILE = 'data/race_training_data.parquet'

# --- CONFIG: TRACK TYPES ---
TRACK_CONFIG = {
    'Monaco Grand Prix': 'Street', 'Singapore Grand Prix': 'Street', 
    'Azerbaijan Grand Prix': 'Street', 'Saudi Arabian Grand Prix': 'Street',
    'Las Vegas Grand Prix': 'Street', 'Miami Grand Prix': 'Street',
    'Australian Grand Prix': 'Street', 'Canadian Grand Prix': 'Hybrid',
    'Italian Grand Prix': 'Power', 'Belgian Grand Prix': 'Power'
}

def calculate_track_stats(df):
    """
    Calculates the 'Personality' of each track based on historical data.
    """
    # 1. CLEANUP: Drop old stats columns if they exist (prevents _x, _y errors)
    cols_to_drop = ['Overtaking_Difficulty', 'Chaos_Factor']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    # 2. Overtaking Factor: Correlation between Grid and Finish Position
    # We need enough data to calculate correlation. If only 1 race, default to 0.7
    try:
        track_stats = df.groupby('EventName')[['Grid_Pos', 'Finish_Pos']].corr().iloc[0::2, -1].reset_index()
        track_stats = track_stats.rename(columns={'Finish_Pos': 'Overtaking_Difficulty'})
        track_stats = track_stats.drop(columns=['level_1']) 
    except:
        # Fallback if data is too scarce
        track_stats = pd.DataFrame(columns=['EventName', 'Overtaking_Difficulty'])

    # 3. Chaos Factor: Percentage of DNFs
    dnf_stats = df.groupby('EventName')['Finish_Pos'].apply(lambda x: (x == 20).sum() / len(x)).reset_index()
    dnf_stats = dnf_stats.rename(columns={'Finish_Pos': 'Chaos_Factor'})
    
    # Merge stats back into main df
    df = pd.merge(df, track_stats, on='EventName', how='left')
    df = pd.merge(df, dnf_stats, on='EventName', how='left')
    
    # Fill defaults for tracks with single races or errors
    df['Overtaking_Difficulty'] = df['Overtaking_Difficulty'].fillna(0.7)
    df['Chaos_Factor'] = df['Chaos_Factor'].fillna(0.1)
    
    return df

def calculate_rolling_features(df):
    """
    Calculates recent form going into the race.
    """
    # 1. CLEANUP: Drop old rolling columns
    cols_to_drop = ['Form_Last3_Race', 'Driver_Track_Avg_Race']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    df = df.sort_values(['Year', 'RoundNumber'])
    
    # 2. Driver Race Form (Median finish of last 3 races)
    df['Form_Last3_Race'] = df.groupby('Driver')['Finish_Pos'] \
                         .transform(lambda x: x.shift(1).rolling(3, min_periods=1).median())
    df['Form_Last3_Race'] = df['Form_Last3_Race'].fillna(10)

    # 3. Track History (Avg finish at this specific track)
    df['Driver_Track_Avg_Race'] = df.groupby(['Driver', 'EventName'])['Finish_Pos'] \
                               .transform(lambda x: x.shift(1).expanding().mean())
    
    # Fill NaN with current season avg
    season_avg = df.groupby(['Year', 'Driver'])['Finish_Pos'].transform('mean')
    df['Driver_Track_Avg_Race'] = df['Driver_Track_Avg_Race'].fillna(season_avg).fillna(10)
    
    return df

def get_data(years=[2023, 2024, 2025], force_rebuild=False):
    existing_df = pd.DataFrame()
    existing_rounds = set()
    
    # If file exists and not forcing rebuild, load it
    if os.path.exists(DATA_FILE) and not force_rebuild:
        try:
            existing_df = pd.read_parquet(DATA_FILE)
            if not existing_df.empty:
                existing_rounds = set(zip(existing_df['Year'], existing_df['RoundNumber']))
                print(f"📂 Resuming Race Data... Loaded {len(existing_df)} rows.")
        except: 
            print("⚠️ File error. Starting fresh.")
    
    for year in years:
        try:
            time.sleep(0.5)
            schedule = fastf1.get_event_schedule(year, include_testing=False)
            completed = schedule[schedule['EventDate'] < pd.Timestamp.now()]
        except: continue

        print(f"📅 Processing Race Data {year}...")

        for i, row in completed.iterrows():
            race_name = row['EventName']
            round_num = row['RoundNumber']
            
            if (year, round_num) in existing_rounds:
                continue
                
            print(f"   📍 R{round_num}: {race_name}")

            try:
                time.sleep(1.0) # Respect rate limits
                
                # --- LOAD RACE SESSION ---
                # We need weather=True for the rain feature
                race = fastf1.get_session(year, round_num, 'R')
                race.load(telemetry=False, messages=False, weather=True)
                
                if not hasattr(race, 'results') or race.results.empty: continue

                # --- EXTRACT RESULTS ---
                r_df = race.results[['DriverNumber', 'Abbreviation', 'TeamName', 'GridPosition', 'Position', 'Status']].copy()
                r_df = r_df.rename(columns={
                    'Abbreviation': 'Driver', 
                    'GridPosition': 'Grid_Pos',
                    'Position': 'Finish_Pos'
                })
                r_df['DriverNumber'] = r_df['DriverNumber'].astype(str).str.strip()

                # --- HANDLING DNFs ---
                # If Position is NaN, check Status.
                # For training, we penalize DNFs by setting them to 20 (Back of grid equivalent)
                # or we could set them to 'Grid_Pos + 5'. Let's stick to 20 for simplicity.
                r_df['Finish_Pos'] = r_df['Finish_Pos'].fillna(20)
                
                # --- FEATURE: WEATHER ---
                is_rain = False
                if not race.weather_data.empty and 'Rainfall' in race.weather_data.columns:
                    # If rainfall detected at any point
                    is_rain = race.weather_data['Rainfall'].any()
                r_df['Is_Rain'] = 1 if is_rain else 0

                # --- FEATURE: TEAMMATE COMPARISON ---
                team_grid_avg = r_df.groupby('TeamName')['Grid_Pos'].mean().reset_index()
                team_grid_avg = team_grid_avg.rename(columns={'Grid_Pos': 'Team_Avg_Grid'})
                
                r_df = pd.merge(r_df, team_grid_avg, on='TeamName', how='left')
                r_df['Teammate_Delta_Grid'] = r_df['Grid_Pos'] - r_df['Team_Avg_Grid']
                
                # --- METADATA ---
                r_df['RoundNumber'] = round_num
                r_df['Year'] = year
                r_df['EventName'] = race_name
                r_df['Track_Type'] = r_df['EventName'].map(TRACK_CONFIG).fillna('Circuit')

                # --- SAVE INCREMENTALLY ---
                if existing_df.empty:
                    existing_df = r_df
                else:
                    existing_df = pd.concat([existing_df, r_df], ignore_index=True)
                
                # --- RE-CALCULATE GLOBAL STATS ---
                # This was the step causing the crash. It is now fixed with drop() logic.
                existing_df = calculate_track_stats(existing_df)
                existing_df = calculate_rolling_features(existing_df)
                
                # Save
                existing_df.to_parquet(DATA_FILE, index=False)
                existing_rounds.add((year, round_num))
                print(f"      ✅ Saved. (Total: {len(existing_df)} rows)")

            except Exception as e:
                print(f"      ⚠️ Error in loop: {e}")
                time.sleep(2)
                continue

    print("🏁 Race Data Generation Complete.")

if __name__ == "__main__":
    # force_rebuild=True to delete the broken file and start over
    get_data(force_rebuild=True)
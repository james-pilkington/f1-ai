import fastf1
import pandas as pd
import os
import numpy as np
import time

# Create cache
if not os.path.exists('f1_cache'): os.makedirs('f1_cache')
fastf1.Cache.enable_cache('f1_cache')

DATA_FILE = 'data/race_training_data.parquet'

TRACK_CONFIG = {
    'Monaco Grand Prix': 'Street', 'Singapore Grand Prix': 'Street', 
    'Azerbaijan Grand Prix': 'Street', 'Saudi Arabian Grand Prix': 'Street',
    'Las Vegas Grand Prix': 'Street', 'Miami Grand Prix': 'Street',
    'Australian Grand Prix': 'Street', 'Canadian Grand Prix': 'Hybrid',
    'Italian Grand Prix': 'Power', 'Belgian Grand Prix': 'Power'
}

def calculate_advanced_stats(df):
    # Cleanup old columns
    cols = ['Overtaking_Difficulty', 'Chaos_Factor', 'Season_Avg_Grid']
    df = df.drop(columns=[c for c in cols if c in df.columns], errors='ignore')

    # 1. Track Stats
    try:
        # Correlation: High = Hard to Pass (Monaco), Low = Easy (Spa)
        track_stats = df.groupby('EventName')[['Grid_Pos', 'Finish_Pos']].corr().iloc[0::2, -1].reset_index()
        track_stats = track_stats.rename(columns={'Finish_Pos': 'Overtaking_Difficulty'})
        track_stats = track_stats.drop(columns=['level_1']) 
    except:
        track_stats = pd.DataFrame(columns=['EventName', 'Overtaking_Difficulty'])

    dnf_stats = df.groupby('EventName')['Finish_Pos'].apply(lambda x: (x == 20).sum() / len(x)).reset_index()
    dnf_stats = dnf_stats.rename(columns={'Finish_Pos': 'Chaos_Factor'})
    
    # 2. Season Average Baseline
    season_avg = df.groupby(['Year', 'Driver'])['Grid_Pos'].transform('mean')
    df['Season_Avg_Grid'] = season_avg
    
    # Merge
    df = pd.merge(df, track_stats, on='EventName', how='left')
    df = pd.merge(df, dnf_stats, on='EventName', how='left')
    
    df['Overtaking_Difficulty'] = df['Overtaking_Difficulty'].fillna(0.7)
    df['Chaos_Factor'] = df['Chaos_Factor'].fillna(0.1)
    
    return df

def get_data(years=[2023, 2024, 2025], force_rebuild=False):
    existing_df = pd.DataFrame()
    existing_rounds = set()
    
    if os.path.exists(DATA_FILE) and not force_rebuild:
        try:
            existing_df = pd.read_parquet(DATA_FILE)
            if not existing_df.empty:
                existing_rounds = set(zip(existing_df['Year'], existing_df['RoundNumber']))
                print(f"📂 Resuming... Loaded {len(existing_df)} rows.")
        except: pass
    
    for year in years:
        try:
            time.sleep(0.5)
            schedule = fastf1.get_event_schedule(year, include_testing=False)
            completed = schedule[schedule['EventDate'] < pd.Timestamp.now()]
        except: continue

        print(f"📅 Processing {year}...")

        for i, row in completed.iterrows():
            race_name = row['EventName']
            round_num = row['RoundNumber']
            
            if (year, round_num) in existing_rounds:
                continue
                
            print(f"   📍 R{round_num}: {race_name}")

            try:
                time.sleep(1)
                race = fastf1.get_session(year, round_num, 'R')
                race.load(telemetry=False, messages=False, weather=True)
                
                if race.results.empty: continue

                r_df = race.results[['DriverNumber', 'Abbreviation', 'TeamName', 'GridPosition', 'Position']].copy()
                r_df = r_df.rename(columns={'Abbreviation': 'Driver', 'GridPosition': 'Grid_Pos', 'Position': 'Finish_Pos'})
                r_df['DriverNumber'] = r_df['DriverNumber'].astype(str).str.strip()
                r_df['Finish_Pos'] = r_df['Finish_Pos'].fillna(20)

                # --- 1. THE "CAR POTENTIAL" LOGIC ---
                # Find the Best Grid Position for each Team this weekend
                team_best_grid = r_df.groupby('TeamName')['Grid_Pos'].min().reset_index()
                team_best_grid = team_best_grid.rename(columns={'Grid_Pos': 'Team_Best_Grid'})
                
                # Merge back
                r_df = pd.merge(r_df, team_best_grid, on='TeamName', how='left')
                
                # --- 2. METADATA ---
                is_rain = False
                if not race.weather_data.empty and 'Rainfall' in race.weather_data.columns:
                    is_rain = race.weather_data['Rainfall'].any()
                r_df['Is_Rain'] = 1 if is_rain else 0

                team_grid_avg = r_df.groupby('TeamName')['Grid_Pos'].mean().reset_index().rename(columns={'Grid_Pos': 'Team_Avg_Grid'})
                r_df = pd.merge(r_df, team_grid_avg, on='TeamName', how='left')
                r_df['Teammate_Delta_Grid'] = r_df['Grid_Pos'] - r_df['Team_Avg_Grid']
                
                r_df['RoundNumber'] = round_num
                r_df['Year'] = year
                r_df['EventName'] = race_name
                r_df['Track_Type'] = r_df['EventName'].map(TRACK_CONFIG).fillna('Circuit')

                if existing_df.empty: existing_df = r_df
                else: existing_df = pd.concat([existing_df, r_df], ignore_index=True)
                
                existing_df = calculate_advanced_stats(existing_df)
                existing_df.to_parquet(DATA_FILE, index=False)
                existing_rounds.add((year, round_num))
                print(f"      ✅ Saved R{round_num}. (Rows: {len(existing_df)})")

            except Exception as e:
                print(f"      ⚠️ Error: {e}")
                time.sleep(2)
                continue

    print("🏁 Update Complete.")

if __name__ == "__main__":
    get_data(force_rebuild=True)
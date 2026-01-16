import fastf1
import pandas as pd
import os
import time
from datetime import datetime

# 1. Setup Cache
if not os.path.exists('f1_cache'):
    os.makedirs('f1_cache')

fastf1.Cache.enable_cache('f1_cache') 

def clean_race_data(df):
    """Forces columns to correct types."""
    numeric_cols = ['Position', 'GridPosition', 'Points', 'RoundNumber', 'Year']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    string_cols = ['Driver', 'TeamName', 'Status', 'PosText', 'EventName', 'DriverNumber', 'SessionType']
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
            
    return df

def fetch_session_data(year, round_num, session_type, retries=3):
    """
    Fetches data for a specific session type ('R' or 'S').
    """
    attempt = 0
    while attempt < retries:
        try:
            session = fastf1.get_session(year, round_num, session_type)
            # Load light data only
            session.load(telemetry=False, weather=False, messages=False)
            return session
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg:
                print(f"    ⚠️ Rate Limit! Cooling down (60s)... Attempt {attempt+1}")
                time.sleep(60) 
                attempt += 1
            elif "Session not found" in error_msg or "index" in error_msg:
                # Normal for non-sprint weekends
                return None
            else:
                return None
    return None

def load_season_data(year, rounds_to_skip=None):
    """
    Fetches data for a season, optionally skipping specific rounds.
    """
    if rounds_to_skip is None:
        rounds_to_skip = []
        
    print(f"Fetching data for Season {year}...")
    try:
        schedule = fastf1.get_event_schedule(year)
    except Exception as e:
        print(f"  ❌ Error fetching schedule: {e}")
        return pd.DataFrame()
        
    # Filter completed races
    completed_races = schedule[schedule['EventDate'] < datetime.now()]
    completed_races = completed_races[completed_races['EventFormat'] != 'testing']
    
    season_results = []
    
    for i, row in completed_races.iterrows():
        round_num = row['RoundNumber']
        
        # --- OPTIMIZATION: SKIP EXISTING ROUNDS ---
        if round_num in rounds_to_skip:
            continue
            
        race_name = row['EventName']
        event_format = row['EventFormat']
        
        time.sleep(1.0) # Politeness delay
        print(f"  - Processing Round {round_num}: {race_name}")
        
        # Fetch Race and Sprint
        sessions_to_fetch = ['R']
        if 'sprint' in str(event_format).lower():
            sessions_to_fetch.append('S')
            
        for s_type in sessions_to_fetch:
            session = fetch_session_data(year, round_num, s_type)
            
            if session and hasattr(session, 'results'):
                try:
                    results = session.results
                    keep_cols = [
                        'Abbreviation', 'DriverNumber', 'TeamName', 
                        'Position', 'GridPosition', 'Points', 
                        'Status', 'ClassifiedPosition'
                    ]
                    available_cols = results.columns.intersection(keep_cols)
                    df_results = results[available_cols].copy()
                    
                    df_results['RoundNumber'] = round_num
                    df_results['EventName'] = race_name
                    df_results['Year'] = year
                    df_results['Date'] = row['EventDate']
                    df_results['SessionType'] = 'Race' if s_type == 'R' else 'Sprint'
                    
                    df_results = df_results.rename(columns={'Abbreviation': 'Driver', 'ClassifiedPosition': 'PosText'})
                    df_results = clean_race_data(df_results)
                    season_results.append(df_results)
                    
                except Exception as e:
                    print(f"    ! Error extracting {s_type}: {e}")
                    continue

    if not season_results:
        return pd.DataFrame()
        
    return pd.concat(season_results, ignore_index=True)

def main():
    current_year = datetime.now().year
    years_to_check = list(range(2005, current_year + 2))
    all_data_path = 'data/processed_f1_data.parquet'
    
    existing_df = pd.DataFrame()
    rounds_to_skip = []
    
    if os.path.exists(all_data_path):
        try:
            print("📂 Loading existing data...")
            existing_df = pd.read_parquet(all_data_path)
            existing_df = clean_race_data(existing_df)
            
            # --- SMART INCREMENTAL LOGIC ---
            # 1. Identify what we have for the current year
            current_data = existing_df[existing_df['Year'] == current_year]
            
            if not current_data.empty:
                existing_rounds = sorted(current_data['RoundNumber'].unique())
                # We SKIP everything except the LAST round we have.
                # Why? Because the last round might be "partial" (e.g. Sprint done, Race not).
                # Re-downloading the last round ensures we get the full weekend results.
                if len(existing_rounds) > 1:
                    rounds_to_skip = existing_rounds[:-1] # Skip all except the last one
                    
                # Remove the data we are about to re-download (the non-skipped rounds)
                # This prevents duplicates.
                existing_df = existing_df[
                    ~((existing_df['Year'] == current_year) & 
                      (~existing_df['RoundNumber'].isin(rounds_to_skip)))
                ]
                print(f"🔄 incremental update: Skipping Rounds {rounds_to_skip} (Already complete)")
            
            # Determine missing years
            downloaded_years = existing_df['Year'].unique()
            years_to_fetch = [y for y in years_to_check if y not in downloaded_years]
            # Always check current year
            if current_year not in years_to_fetch:
                years_to_fetch.append(current_year)
            years_to_fetch = sorted(list(set(years_to_fetch)))
            
        except Exception as e:
            print(f"⚠️ Error reading existing data: {e}. Starting fresh.")
            existing_df = pd.DataFrame()
            years_to_fetch = years_to_check
    else:
        years_to_fetch = years_to_check
    
    # Run Download
    new_data = []
    for year in years_to_fetch:
        # Pass skip list ONLY if it's the current year
        skip = rounds_to_skip if year == current_year else []
        
        df_year = load_season_data(year, rounds_to_skip=skip)
        
        if not df_year.empty:
            new_data.append(df_year)
            
            # Safe Save Loop
            save_df = pd.concat([existing_df] + new_data, ignore_index=True)
            if not os.path.exists('data'): os.makedirs('data')
            save_df.to_parquet(all_data_path, index=False)
            print(f"  💾 Saved progress through {year}")

    print("✅ Update Complete!")

if __name__ == "__main__":
    main()
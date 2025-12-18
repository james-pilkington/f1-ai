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
    """Forces columns to correct types to prevent Parquet crashes."""
    numeric_cols = ['Position', 'GridPosition', 'Points', 'RoundNumber', 'Year']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    string_cols = ['Driver', 'TeamName', 'Status', 'PosText', 'EventName', 'DriverNumber']
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
            
    return df

def fetch_race_data(year, round_num, race_name, retries=3):
    attempt = 0
    while attempt < retries:
        try:
            session = fastf1.get_session(year, round_num, 'R')
            session.load(telemetry=False, weather=False, messages=False)
            return session
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg:
                print(f"    ⚠️ Rate Limit Hit! Cooling down for 60 seconds... (Attempt {attempt+1}/{retries})")
                time.sleep(60) 
                attempt += 1
            else:
                return None
    return None

def load_season_data(year):
    print(f"Fetching data for Season {year}...")
    try:
        schedule = fastf1.get_event_schedule(year)
    except Exception as e:
        print(f"  ❌ Error fetching schedule for {year}: {e}")
        return pd.DataFrame()
        
    completed_races = schedule[schedule['EventDate'] < datetime.now()]
    completed_races = completed_races[completed_races['EventFormat'] != 'testing']
    
    season_results = []
    
    for i, row in completed_races.iterrows():
        round_num = row['RoundNumber']
        race_name = row['EventName']
        
        # Politeness Sleep
        time.sleep(1.5)
        
        print(f"  - Processing Round {round_num}: {race_name}")
        
        session = fetch_race_data(year, round_num, race_name)
        
        if session and hasattr(session, 'results'):
            try:
                results = session.results
                results['RoundNumber'] = round_num
                results['EventName'] = race_name
                results['Year'] = year
                results['Date'] = row['EventDate']
                
                keep_cols = [
                    'Year', 'RoundNumber', 'EventName', 'Date', 
                    'Abbreviation', 'DriverNumber', 'TeamName', 
                    'Position', 'GridPosition', 'Points', 
                    'Status', 'ClassifiedPosition'
                ]
                
                available_cols = results.columns.intersection(keep_cols)
                df_results = results[available_cols].copy()
                df_results = df_results.rename(columns={'Abbreviation': 'Driver', 'ClassifiedPosition': 'PosText'})
                df_results = clean_race_data(df_results)
                
                season_results.append(df_results)
                
            except Exception as e:
                print(f"    ! Error extracting results: {e}")
                continue
        else:
            print(f"    ! Skipped {race_name} (No data available)")

    if not season_results:
        return pd.DataFrame()
        
    return pd.concat(season_results, ignore_index=True)

def main():
    # Production Range: Check everything from 2005 onwards
    years_to_check = list(range(2005, datetime.now().year + 2))
    
    all_data_path = 'data/processed_f1_data.parquet'
    
    # 1. Load Existing Data
    if os.path.exists(all_data_path):
        try:
            existing_df = pd.read_parquet(all_data_path)
            existing_df = clean_race_data(existing_df)
            
            # Smart Check: Which years do we completely have?
            # Note: This simple check assumes if we have the year, we have *all* races.
            # For a weekly update, this is fine (we usually just add the current year).
            downloaded_years = existing_df['Year'].unique()
            print(f"✅ Found existing data for: {sorted(downloaded_years)}")
            
            # Filter: Only fetch years we are missing
            years_to_fetch = [y for y in years_to_check if y not in downloaded_years]
            
            # ALWAYS check the current year, just in case a new race happened since last run
            current_year = datetime.now().year
            if current_year not in years_to_fetch:
                years_to_fetch.append(current_year)
                # Remove current year from existing so we don't duplicate it (we will overwrite it with fresh data)
                existing_df = existing_df[existing_df['Year'] != current_year]
                
            years_to_fetch = sorted(list(set(years_to_fetch))) # De-duplicate
            
        except:
            existing_df = pd.DataFrame()
            years_to_fetch = years_to_check
    else:
        existing_df = pd.DataFrame()
        years_to_fetch = years_to_check
    
    print(f"🚀 Updating years: {years_to_fetch}")
    
    new_data = []
    
    for year in years_to_fetch:
        df_year = load_season_data(year)
        if not df_year.empty:
            new_data.append(df_year)
            
            # Save Progress
            if not existing_df.empty:
                combined_df = pd.concat([existing_df] + new_data, ignore_index=True)
            else:
                combined_df = pd.concat(new_data, ignore_index=True)
            
            if not os.path.exists('data'):
                os.makedirs('data')
                
            combined_df.to_parquet(all_data_path, index=False)
            print(f"  💾 Saved progress through {year}")

    print("✅ Data Update Complete!")

if __name__ == "__main__":
    main()
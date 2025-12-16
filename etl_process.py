import fastf1
import pandas as pd
import os
from datetime import datetime

# 1. Setup Cache (So we don't re-download data every time we run locally)
# FastF1 will create a folder called 'f1_cache' in your project root
if not os.path.exists('f1_cache'):
    os.makedirs('f1_cache')

fastf1.Cache.enable_cache('f1_cache') 

def load_season_data(year):
    """
    Fetches race results and lap data for a specific year.
    """
    print(f"Fetching data for Season {year}...")
    
    # Get the schedule for the year
    schedule = fastf1.get_event_schedule(year)
    
    # Filter: Only keep races that have already happened
    # We check if the 'EventDate' is in the past
    completed_races = schedule[schedule['EventDate'] < datetime.now()]
    
    # Exclude 'Pre-Season Testing' if it appears
    completed_races = completed_races[completed_races['EventFormat'] != 'testing']
    
    all_results = []
    
    for i, row in completed_races.iterrows():
        round_num = row['RoundNumber']
        race_name = row['EventName']
        print(f"  - Processing Round {round_num}: {race_name}")
        
        try:
            # Load the session
            session = fastf1.get_session(year, round_num, 'R') # 'R' = Race
            session.load(telemetry=False, weather=False, messages=False) # Load light data
            
            # 1. Extract Race Results
            results = session.results
            results['RoundNumber'] = round_num
            results['EventName'] = race_name
            results['Year'] = year
            results['Date'] = row['EventDate']
            
            # Clean up columns (we don't need everything)
            keep_cols = [
                'Year', 'RoundNumber', 'EventName', 'Date', 
                'Abbreviation', 'DriverNumber', 'TeamName', 
                'ClassifiedPosition', 'GridPosition', 'Points', 
                'Status', 'Q1', 'Q2', 'Q3' 
            ]
            
            # FastF1 results can be messy, ensure we have the cols we need
            # We select strictly what we need to keep file size down
            df_results = results.loc[:, results.columns.intersection(keep_cols)].copy()
            
            # Rename for clarity
            df_results = df_results.rename(columns={'Abbreviation': 'Driver', 'ClassifiedPosition': 'Position'})
            
            all_results.append(df_results)
            
        except Exception as e:
            print(f"    ! Error loading {race_name}: {e}")
            continue

    if not all_results:
        return pd.DataFrame()
        
    return pd.concat(all_results, ignore_index=True)

def main():
    # Define which seasons to fetch
    # Since you want history + current, let's grab 2024 and 2025
    years_to_fetch = [2024, 2025]
    
    full_dataset = []
    
    for year in years_to_fetch:
        df_year = load_season_data(year)
        full_dataset.append(df_year)
    
    # Combine all years
    final_df = pd.concat(full_dataset, ignore_index=True)
    
    # Ensure data folder exists
    if not os.path.exists('data'):
        os.makedirs('data')
        
    # Save to Parquet (The "Golden File")
    output_path = 'data/processed_f1_data.parquet'
    final_df.to_parquet(output_path, index=False)
    
    print(f"✅ Success! Data saved to {output_path}")
    print(f"   Total rows: {len(final_df)}")
    print(f"   Columns: {list(final_df.columns)}")

if __name__ == "__main__":
    main()
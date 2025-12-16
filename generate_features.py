import fastf1
import pandas as pd
import os
import numpy as np

# Create cache folder
if not os.path.exists('f1_cache'):
    os.makedirs('f1_cache')

# Enable cache
fastf1.Cache.enable_cache('f1_cache')

def calculate_fp_rankings(session):
    """
    Fallback method: If 'Position' is missing in results, 
    calculate it manually from the fastest lap times.
    """
    try:
        # Get all laps, excluding in/out laps
        laps = session.laps.pick_quicklaps()
        if laps.empty:
            return pd.DataFrame()
            
        # Find fastest lap per driver
        best_times = laps.groupby('DriverNumber')['LapTime'].min().sort_values()
        
        # Create a dataframe with the rank
        rankings = pd.DataFrame({
            'DriverNumber': best_times.index,
            'FP_Pos': range(1, len(best_times) + 1),
            'FP_Time': best_times.values
        }).reset_index(drop=True)
        
        return rankings
    except Exception as e:
        return pd.DataFrame()

def get_data(years=[2023, 2024,2025]):
    all_data = []
    
    for year in years:
        try:
            print(f"📅 Fetching schedule for {year}...")
            schedule = fastf1.get_event_schedule(year, include_testing=False)
            
            # Filter for completed races
            completed = schedule[schedule['EventDate'] < pd.Timestamp.now()]
        except Exception as e:
            print(f"❌ Error getting schedule for {year}: {e}")
            continue

        print(f"   Found {len(completed)} completed races in {year}.")

        for i, row in completed.iterrows():
            race_name = row['EventName']
            round_num = row['RoundNumber']
            # FIX: Use the date to get the year, or just use the loop variable
            event_year = row['EventDate'].year
            
            print(f"   📍 Processing Round {round_num}: {race_name}")

            try:
                # --- 1. LOAD QUALI (Target) ---
                qs = fastf1.get_session(event_year, round_num, 'Q')
                qs.load(telemetry=False, messages=False, weather=False)
                
                if not hasattr(qs, 'results') or qs.results.empty:
                    print("      ⚠️ No Quali results found.")
                    continue
                
                # Extract Results
                q_df = qs.results.loc[:, ['DriverNumber', 'Abbreviation', 'TeamName', 'Position']].copy()
                q_df = q_df.rename(columns={'Position': 'Quali_Pos', 'Abbreviation': 'Driver'})
                q_df['DriverNumber'] = q_df['DriverNumber'].astype(str)
                
                # --- 2. LOAD PRACTICE (Signal) ---
                # Try FP3, fallback to FP1
                try:
                    fp = fastf1.get_session(event_year, round_num, 'FP3')
                    fp.load(telemetry=False, messages=False, weather=False)
                except:
                    fp = fastf1.get_session(event_year, round_num, 'FP1')
                    fp.load(telemetry=False, messages=False, weather=False)

                # --- 3. EXTRACT FP POSITIONS ---
                fp_df = pd.DataFrame()
                
                # Strategy A: Use official Position
                if hasattr(fp, 'results') and not fp.results.empty and 'Position' in fp.results.columns:
                    if not fp.results['Position'].isna().all():
                        fp_df = fp.results.loc[:, ['DriverNumber', 'Position']].copy()
                        fp_df = fp_df.rename(columns={'Position': 'FP_Pos'})
                
                # Strategy B: Calculate from Laps if A fails
                if fp_df.empty or fp_df['FP_Pos'].isna().all():
                    fp_df = calculate_fp_rankings(fp)
                
                if fp_df.empty:
                    print(f"      ⚠️ No Practice data available.")
                    continue
                    
                fp_df['DriverNumber'] = fp_df['DriverNumber'].astype(str)

                # --- 4. MERGE ---
                merged = pd.merge(q_df, fp_df, on='DriverNumber', how='inner')
                
                # Add Context
                merged['RoundNumber'] = round_num
                merged['Year'] = event_year
                merged['EventName'] = race_name
                
                # Feature: Team Pace
                merged['FP_Pos'] = pd.to_numeric(merged['FP_Pos'], errors='coerce')
                team_stats = merged.groupby('TeamName')['FP_Pos'].mean().reset_index()
                team_stats = team_stats.rename(columns={'FP_Pos': 'Team_Avg_FP_Pos'})
                
                merged = pd.merge(merged, team_stats, on='TeamName', how='left')
                merged['Teammate_Delta_Pos'] = merged['FP_Pos'] - merged['Team_Avg_FP_Pos']
                
                all_data.append(merged)
                
            except Exception as e:
                print(f"      ❌ Skipped due to error: {e}")
                continue

    if not all_data:
        return pd.DataFrame()
        
    final_df = pd.concat(all_data, ignore_index=True)
    final_df = final_df.dropna(subset=['Quali_Pos', 'FP_Pos'])
    
    return final_df

if __name__ == "__main__":
    print("🚀 Starting feature generation...")
    df = get_data(years=[2023, 2024,2025])
    
    if df.empty:
        print("\n❌ DATASET IS EMPTY.")
    else:
        output_file = 'data/quali_training_data.parquet'
        df.to_parquet(output_file, index=False)
        print(f"\n✅ SUCCESS! Generated {len(df)} rows.")
        print(f"   Saved to: {output_file}")
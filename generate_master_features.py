import fastf1
import pandas as pd
import os
import numpy as np
import time

# Create cache
if not os.path.exists('f1_cache'): os.makedirs('f1_cache')
fastf1.Cache.enable_cache('f1_cache')

DATA_FILE = 'data/race_data_master.parquet'

# --- 1. CONFIG: THE WORLD TOUR (24 Tracks) ---
# We define physical characteristics for every possible track.
# This helps the model generalize: "If good at High Downforce (Monaco), likely good at Hungary."

TRACK_CONFIG = {
    # High Downforce / Street / Technical
    'Monaco Grand Prix': {'Type': 'Street', 'Downforce': 'High', 'Overtaking': 'Very_Hard'},
    'Singapore Grand Prix': {'Type': 'Street', 'Downforce': 'High', 'Overtaking': 'Hard'},
    'Hungarian Grand Prix': {'Type': 'Circuit', 'Downforce': 'High', 'Overtaking': 'Hard'},
    'Dutch Grand Prix': {'Type': 'Circuit', 'Downforce': 'High', 'Overtaking': 'Hard'},
    'Spanish Grand Prix': {'Type': 'Circuit', 'Downforce': 'High', 'Overtaking': 'Medium'},
    'Mexico City Grand Prix': {'Type': 'Circuit', 'Downforce': 'High', 'Overtaking': 'Medium'},

    # Balanced / Hybrid
    'Bahrain Grand Prix': {'Type': 'Circuit', 'Downforce': 'Medium', 'Overtaking': 'Easy'},
    'Chinese Grand Prix': {'Type': 'Circuit', 'Downforce': 'Medium', 'Overtaking': 'Medium'},
    'Japanese Grand Prix': {'Type': 'Circuit', 'Downforce': 'High', 'Overtaking': 'Hard'}, # Suzuka is technically High DF but fast
    'United States Grand Prix': {'Type': 'Circuit', 'Downforce': 'Medium', 'Overtaking': 'Medium'},
    'Miami Grand Prix': {'Type': 'Street', 'Downforce': 'Medium', 'Overtaking': 'Medium'},
    'Canadian Grand Prix': {'Type': 'Hybrid', 'Downforce': 'Medium', 'Overtaking': 'Medium'},
    'Australian Grand Prix': {'Type': 'Street', 'Downforce': 'Medium', 'Overtaking': 'Medium'},
    'Qatar Grand Prix': {'Type': 'Circuit', 'Downforce': 'High', 'Overtaking': 'Medium'},
    'Abu Dhabi Grand Prix': {'Type': 'Circuit', 'Downforce': 'Medium', 'Overtaking': 'Hard'},

    # Power / Low Downforce
    'Italian Grand Prix': {'Type': 'Power', 'Downforce': 'Low', 'Overtaking': 'Easy'},
    'Belgian Grand Prix': {'Type': 'Power', 'Downforce': 'Low', 'Overtaking': 'Easy'},
    'British Grand Prix': {'Type': 'Power', 'Downforce': 'Medium', 'Overtaking': 'Easy'}, # High speed corners
    'Austrian Grand Prix': {'Type': 'Power', 'Downforce': 'Medium', 'Overtaking': 'Easy'},
    'Saudi Arabian Grand Prix': {'Type': 'Street', 'Downforce': 'Low', 'Overtaking': 'Medium'},
    'Azerbaijan Grand Prix': {'Type': 'Street', 'Downforce': 'Low', 'Overtaking': 'Easy'},
    'Las Vegas Grand Prix': {'Type': 'Street', 'Downforce': 'Low', 'Overtaking': 'Easy'},
    'Sao Paulo Grand Prix': {'Type': 'Circuit', 'Downforce': 'Medium', 'Overtaking': 'Easy'},
}

def get_track_features(event_name):
    """Parses the config into flat columns"""
    cfg = TRACK_CONFIG.get(event_name, {'Type': 'Circuit', 'Downforce': 'Medium', 'Overtaking': 'Medium'})
    return cfg['Type'], cfg['Downforce'], cfg['Overtaking']

# --- 2. GLOBAL CALCULATIONS ---
def calculate_metrics(df):
    print("   ⚙️ Calculating Advanced Metrics...")
    
    # Sort for rolling calcs
    df = df.sort_values(['Year', 'RoundNumber'])
    
    # A. QUALI CONSISTENCY (Std Dev of Grid Pos)
    # How erratic is this driver?
    df['Quali_Volatility'] = df.groupby('Driver')['Grid_Pos'].transform(lambda x: x.shift(1).rolling(5, min_periods=2).std())
    
    # B. TEAMMATE DELTA TREND (Are they getting closer or further away?)
    # We calculate the slope of the last 3 races delta. 
    # Positive = Falling back from teammate. Negative = Catching up.
    # (Simplified: just the rolling mean for now)
    df['Teammate_Trend_3R'] = df.groupby('Driver')['Teammate_Delta_Grid'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    
    # C. FORM GUIDES
    df['Form_Last5_Grid'] = df.groupby('Driver')['Grid_Pos'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    df['Form_Last5_Finish'] = df.groupby('Driver')['Finish_Pos'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    
    # D. DRIVER SKILL: POSITIONS GAINED (Normalized)
    # Grid - Finish. We take the median gain over the last year.
    # High number = "Race Day Specialist" (Alonso/Hamilton)
    df['Avg_Pos_Gained'] = df.groupby('Driver')['Positions_Gained'].transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
    
    # E. TRACK SUITABILITY (Driver History at this specific event)
    df['Driver_Track_History'] = df.groupby(['Driver', 'EventName'])['Finish_Pos'].transform(lambda x: x.shift(1).expanding().mean())
    
    # F. PIT CREW PERFORMANCE (Season Avg)
    df['Team_Pit_Speed'] = df.groupby(['Year', 'TeamName'])['Avg_Pit_Time'].transform(lambda x: x.expanding().mean())

    # Fill NaNs
    df = df.fillna(0)
    return df

def get_data(years=[2023, 2024, 2025], force_rebuild=False):
    master_df = pd.DataFrame()
    existing_rounds = set()
    
    if os.path.exists(DATA_FILE) and not force_rebuild:
        try:
            master_df = pd.read_parquet(DATA_FILE)
            existing_rounds = set(zip(master_df['Year'], master_df['RoundNumber']))
            print(f"📂 Resuming... Loaded {len(master_df)} rows.")
        except: pass
    
    for year in years:
        try:
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
                # ==========================================================
                # PHASE 1: QUALIFYING (The Speed Truth)
                # ==========================================================
                time.sleep(1)
                quali = fastf1.get_session(year, round_num, 'Q')
                quali.load(telemetry=False, messages=False)
                
                # Calculate "Gap to Pole %" (Normalized Speed)
                # This is better than position because it accounts for close fields vs dominance
                q_res = quali.results
                if not q_res.empty:
                    pole_time = q_res['Q3'].min()
                    # If Q3 is empty (wet?), try Q1
                    if pd.isna(pole_time): pole_time = q_res['Q1'].min()
                    
                    # Safe division function
                    def calc_gap(row):
                        best = row['Q3'] if not pd.isna(row['Q3']) else (row['Q2'] if not pd.isna(row['Q2']) else row['Q1'])
                        if pd.isna(best) or pd.isna(pole_time): return 0.0
                        return (best - pole_time).total_seconds() / pole_time.total_seconds()

                    q_res['Quali_Gap_Pct'] = q_res.apply(calc_gap, axis=1)
                    q_data = q_res[['Abbreviation', 'Quali_Gap_Pct']].rename(columns={'Abbreviation': 'Driver'})
                else:
                    q_data = pd.DataFrame(columns=['Driver', 'Quali_Gap_Pct'])

                # ==========================================================
                # PHASE 2: THE RACE (The Result)
                # ==========================================================
                time.sleep(1)
                race = fastf1.get_session(year, round_num, 'R')
                race.load(telemetry=False, messages=True, weather=True) # Messages needed for SC?
                
                if race.results.empty: continue
                
                r_df = race.results[['DriverNumber', 'Abbreviation', 'TeamName', 'GridPosition', 'Position', 'Status']].copy()
                r_df = r_df.rename(columns={'Abbreviation': 'Driver', 'GridPosition': 'Grid_Pos', 'Position': 'Finish_Pos'})
                
                # Cleanup
                r_df['DriverNumber'] = r_df['DriverNumber'].astype(str).str.strip()
                r_df['Finish_Pos'] = r_df['Finish_Pos'].fillna(20)
                r_df['Positions_Gained'] = r_df['Grid_Pos'] - r_df['Finish_Pos']

                # --- 2A. PIT STOP PERFORMANCE ---
                # We calculate the average pit stop duration for this race per driver
                # FastF1 v3.1+ has a 'pit_stops' property
                try:
                    pits = race.pit_stops
                    if not pits.empty:
                        # Duration is commonly usually 'Duration' or calculated from 'PitOutTime' - 'PitInTime'
                        # 'Duration' in FastF1 is the stopped time. 'PitOut' - 'PitIn' is total lane time.
                        # We want stopped time (Crew performance).
                        pits = pits.rename(columns={'DriverNumber': 'DriverNumber_Pit'}) # Avoid merge conflict
                        
                        # Merge Driver Code to Pit Data to get Team
                        # Actually simpler: Group by DriverNumber in pits, calculate mean duration
                        avg_pit = pits.groupby('DriverNumber')['Duration'].mean().dt.total_seconds().reset_index()
                        avg_pit.columns = ['DriverNumber', 'Avg_Pit_Time']
                        
                        r_df = pd.merge(r_df, avg_pit, on='DriverNumber', how='left')
                        r_df['Avg_Pit_Time'] = r_df['Avg_Pit_Time'].fillna(25.0) # Default penalty if no stop or error
                    else:
                        r_df['Avg_Pit_Time'] = 0
                except:
                    r_df['Avg_Pit_Time'] = 0

                # --- 2B. CHAOS FACTORS (SC / VSC) ---
                # Scan track status for '4' (SC) or '6' (VSC)
                # This is a race-wide metric
                status_codes = race.track_status['Status'].unique()
                has_sc = '4' in status_codes
                has_vsc = '6' in status_codes or '7' in status_codes
                
                # Lap 1 Chaos: How many people changed position by > 3 spots on Lap 1?
                # (Requires loop, skipping for speed, using simple DNF rate instead)
                dnf_count = len(r_df[~r_df['Status'].isin(['Finished', '+1 Lap', '+2 Laps'])])
                dnf_rate = dnf_count / len(r_df)

                # ==========================================================
                # PHASE 3: MERGING & CALCULATED FEATURES
                # ==========================================================
                
                # Merge Quali Pace
                r_df = pd.merge(r_df, q_data, on='Driver', how='left')
                r_df['Quali_Gap_Pct'] = r_df['Quali_Gap_Pct'].fillna(0.05) # Default 5% gap if no time
                
                # Track Characteristics
                t_type, t_downforce, t_overtake = get_track_features(race_name)
                r_df['Track_Type'] = t_type
                r_df['Track_Downforce'] = t_downforce
                r_df['Track_Overtake'] = t_overtake
                
                # Environmental
                is_rain = False
                if not race.weather_data.empty and 'Rainfall' in race.weather_data.columns:
                    is_rain = race.weather_data['Rainfall'].any()
                
                r_df['Is_Rain'] = 1 if is_rain else 0
                r_df['SC_Deployed'] = 1 if has_sc else 0
                r_df['VSC_Deployed'] = 1 if has_vsc else 0
                r_df['Race_DNF_Rate'] = dnf_rate
                
                # Team Context
                team_best = r_df.groupby('TeamName')['Grid_Pos'].min().reset_index().rename(columns={'Grid_Pos': 'Team_Best_Grid'})
                team_avg = r_df.groupby('TeamName')['Grid_Pos'].mean().reset_index().rename(columns={'Grid_Pos': 'Team_Avg_Grid'})
                
                r_df = pd.merge(r_df, team_best, on='TeamName', how='left')
                r_df = pd.merge(r_df, team_avg, on='TeamName', how='left')
                
                r_df['Car_Potential'] = r_df['Grid_Pos'] - r_df['Team_Best_Grid']
                r_df['Teammate_Delta_Grid'] = r_df['Grid_Pos'] - r_df['Team_Avg_Grid']
                
                # Metadata
                r_df['RoundNumber'] = round_num
                r_df['Year'] = year
                r_df['EventName'] = race_name

                # Append
                if master_df.empty: master_df = r_df
                else: master_df = pd.concat([master_df, r_df], ignore_index=True)
                
                # Save Checkpoint
                # (We calculate global metrics at the end to keep the loop fast)
                master_df.to_parquet(DATA_FILE, index=False)
                existing_rounds.add((year, round_num))
                print(f"      ✅ Saved {race_name}")

            except Exception as e:
                print(f"      ⚠️ Error: {e}")
                continue

    # --- FINAL PASS: GLOBAL METRICS ---
    if not master_df.empty:
        master_df = calculate_metrics(master_df)
        master_df.to_parquet(DATA_FILE, index=False)
        print("🏁 Master Feature Store Built Successfully.")

if __name__ == "__main__":
    get_data(force_rebuild=False)
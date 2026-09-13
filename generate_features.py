import fastf1
import pandas as pd
import os
import numpy as np
import time

# Create cache
if not os.path.exists('f1_cache'): os.makedirs('f1_cache')
fastf1.Cache.enable_cache('f1_cache')

DATA_FILE = 'data/quali_training_data.parquet'

# FastF1 self-limits to 200-500 calls/h per process. A weekly run only ever
# needs 1 new round; this cap just keeps a multi-round catch-up from blowing
# through that budget and failing every round outright.
MAX_NEW_ROUNDS_PER_RUN = 2

# --- 1. CONFIG: TRACK TYPES ---
TRACK_CONFIG = {
    'Monaco Grand Prix': 'Street', 'Singapore Grand Prix': 'Street', 
    'Azerbaijan Grand Prix': 'Street', 'Saudi Arabian Grand Prix': 'Street',
    'Las Vegas Grand Prix': 'Street', 'Miami Grand Prix': 'Street',
    'Australian Grand Prix': 'Street', 'Canadian Grand Prix': 'Hybrid',
    'Italian Grand Prix': 'Power', 'Belgian Grand Prix': 'Power'
}

def calculate_advanced_features(df):
    """
    Runs post-processing on the full dataset to generate history-based features.
    """
    df = df.sort_values(['Year', 'RoundNumber'])
    
    # A. DRIVER FORM (Using MEDIAN to ignore outliers/crashes)
    df['Form_Last3'] = df.groupby('Driver')['Quali_Pos'] \
                         .transform(lambda x: x.shift(1).rolling(3, min_periods=1).median())
    df['Form_Last3'] = df['Form_Last3'].fillna(10) # Default to mid-field

    # B. TRACK SPECIFIC HISTORY
    df['Driver_Track_Avg'] = df.groupby(['Driver', 'EventName'])['Quali_Pos'] \
                               .transform(lambda x: x.shift(1).expanding().mean())
    
    # Fill NaN (First time at track) with their current season average
    df['Season_Avg'] = df.groupby(['Year', 'Driver'])['Quali_Pos'].transform('mean')
    df['Driver_Track_Avg'] = df['Driver_Track_Avg'].fillna(df['Season_Avg']).fillna(10)
    
    df = df.drop(columns=['Season_Avg'], errors='ignore')
    return df

def load_session_safe(year, round_num, session_type, retries=3, **load_kwargs):
    """Loads a session, backing off on FastF1/Ergast rate-limit (429) responses."""
    for attempt in range(retries):
        try:
            s = fastf1.get_session(year, round_num, session_type)
            s.load(**load_kwargs)
            return s
        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                print(f"      ⚠️ Rate limited. Cooling down 60s (attempt {attempt+1}/{retries})...")
                time.sleep(60)
            else:
                raise

def get_data(years=None, force_rebuild=False):
    if years is None:
        years = list(range(2023, pd.Timestamp.now().year + 1))
    # 1. Load Existing Data
    existing_df = pd.DataFrame()
    existing_rounds = set()
    
    if os.path.exists(DATA_FILE) and not force_rebuild:
        try:
            existing_df = pd.read_parquet(DATA_FILE)
            if not existing_df.empty:
                existing_rounds = set(zip(existing_df['Year'], existing_df['RoundNumber']))
                print(f"📂 Resuming... Loaded {len(existing_df)} rows. Skipping {len(existing_rounds)} completed races.")
        except: 
            print("⚠️ File error. Starting fresh.")
    
    new_rounds_fetched = 0
    for year in years:
        try:
            time.sleep(0.5)
            schedule = fastf1.get_event_schedule(year, include_testing=False)
            completed = schedule[schedule['EventDate'] < pd.Timestamp.now()]
        except: continue

        print(f"📅 Checking Season {year}...")

        for i, row in completed.iterrows():
            race_name = row['EventName']
            round_num = row['RoundNumber']

            if (year, round_num) in existing_rounds:
                continue

            if new_rounds_fetched >= MAX_NEW_ROUNDS_PER_RUN:
                print(f"   ⏸️ Reached cap of {MAX_NEW_ROUNDS_PER_RUN} new rounds this run. Remaining rounds will be picked up next run.")
                break

            new_rounds_fetched += 1
            print(f"   📍 FETCHING R{round_num}: {race_name}")

            try:
                # --- A. TARGET: QUALI ---
                time.sleep(1) # Safety Delay
                qs = load_session_safe(year, round_num, 'Q', telemetry=False, messages=False, weather=False)

                if not hasattr(qs, 'results') or qs.results.empty: continue

                q_df = qs.results[['DriverNumber', 'Abbreviation', 'TeamName', 'Position']].copy()
                q_df = q_df.rename(columns={'Position': 'Quali_Pos', 'Abbreviation': 'Driver'})
                q_df['DriverNumber'] = q_df['DriverNumber'].astype(str).str.strip()

                # --- B. SIGNAL: FP3 (or FP1) ---
                try:
                    time.sleep(1) # Safety Delay
                    fp = load_session_safe(year, round_num, 'FP3', telemetry=False, messages=False, weather=False)
                except Exception:
                    time.sleep(2)
                    fp = load_session_safe(year, round_num, 'FP1', telemetry=False, messages=False, weather=False)

                # --- CRITICAL FIX: FORCE CALCULATE RANK FROM TIME ---
                # Official 'Position' is often NaN in practice. We calculate it ourselves.
                if 'Time' in fp.results.columns:
                    # Rank drivers by fastest time (method='min' means ties share rank)
                    fp.results['Calculated_Pos'] = fp.results['Time'].rank(method='min')
                    
                    # Fill official Position with our calculated one
                    fp.results['Position'] = fp.results['Position'].fillna(fp.results['Calculated_Pos'])
                    
                    # Calculate Gaps
                    leader_time = fp.results['Time'].min()
                    fp.results['Gap_To_Leader'] = (fp.results['Time'] - leader_time).dt.total_seconds()
                else:
                    fp.results['Gap_To_Leader'] = np.nan

                # Grab Columns
                fp_df = fp.results[['DriverNumber', 'Position', 'Gap_To_Leader']].copy()
                fp_df = fp_df.rename(columns={'Position': 'FP_Pos', 'Gap_To_Leader': 'FP_Gap'})
                fp_df['DriverNumber'] = fp_df['DriverNumber'].astype(str).str.strip()
                
                # Fallback for Drivers with No Time (Crashes/Garage)
                # If they have no rank, put them at the back (P20)
                fp_df['FP_Pos'] = fp_df['FP_Pos'].fillna(20)
                
                # Fallback for Gap
                mask = fp_df['FP_Gap'].isna()
                fp_df.loc[mask, 'FP_Gap'] = fp_df.loc[mask, 'FP_Pos'] * 0.1

                # --- D. MERGE & PROCESS ---
                merged = pd.merge(q_df, fp_df, on='DriverNumber', how='inner')
                merged['RoundNumber'] = round_num
                merged['Year'] = year
                merged['EventName'] = race_name
                merged['Track_Type'] = merged['EventName'].map(TRACK_CONFIG).fillna('Circuit')

                # Team Context
                merged['FP_Gap'] = pd.to_numeric(merged['FP_Gap'], errors='coerce').fillna(2.0)
                team_stats = merged.groupby('TeamName')['FP_Gap'].mean().reset_index()
                team_stats = team_stats.rename(columns={'FP_Gap': 'Team_Avg_Gap'})
                
                merged = pd.merge(merged, team_stats, on='TeamName', how='left')
                merged['Teammate_Delta_Gap'] = merged['FP_Gap'] - merged['Team_Avg_Gap']
                
                # --- E. SAVE IMMEDIATELY ---
                if existing_df.empty:
                    existing_df = merged
                else:
                    existing_df = pd.concat([existing_df, merged], ignore_index=True)
                
                # Recalculate advanced stats
                existing_df = calculate_advanced_features(existing_df)
                
                save_df = existing_df.dropna(subset=['Quali_Pos', 'FP_Pos'])
                save_df.to_parquet(DATA_FILE, index=False)
                
                existing_rounds.add((year, round_num))
                print(f"      ✅ Saved. (Total Database: {len(save_df)} rows)")
                
            except Exception as e:
                print(f"      ⚠️ Error: {e}")
                time.sleep(2)
                continue

        if new_rounds_fetched >= MAX_NEW_ROUNDS_PER_RUN:
            break

    print(f"🏁 Update Complete.")

if __name__ == "__main__":
    get_data()
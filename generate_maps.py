import fastf1
import pandas as pd
import numpy as np
import os

# Setup Cache
if not os.path.exists('f1_cache'):
    os.makedirs('f1_cache')
fastf1.Cache.enable_cache('f1_cache')

def generate_track_maps():
    print("🗺️ Generating Track Map Database...")
    
    # 1. Get list of all tracks we need
    try:
        df = pd.read_parquet('data/processed_f1_data.parquet')
        all_tracks = df['EventName'].unique()
    except:
        print("❌ Could not read processed_f1_data.parquet. Run etl_process.py first.")
        return

    map_data = []

    for track in all_tracks:
        print(f"  📍 Processing: {track}...")
        
        # Find the most recent year this track was raced
        track_history = df[df['EventName'] == track]
        # We prefer newer years for better telemetry, but check up to 2024
        recent_year = track_history['Year'].max()
        
        # Telemetry only reliable after ~2018. If latest race is 2013, we might fail.
        if recent_year < 2018:
            print(f"     ⚠️ Track too old ({recent_year}), skipping map.")
            continue
            
        try:
            # Load Session
            session = fastf1.get_session(recent_year, track, 'Q')
            session.load(telemetry=True, weather=False, messages=False)
            
            # A. Get Track Shape (The Line)
            lap = session.laps.pick_fastest()
            tel = lap.get_telemetry()
            
            # Downsample: We don't need 10,000 points for a small map. 1 in 5 is enough.
            # This keeps file size tiny.
            x_coords = tel['X'].values[::5].tolist()
            y_coords = tel['Y'].values[::5].tolist()
            
            # B. Get Corner Markers (The "CircuitInfo" you mentioned)
            # Useful for annotating the map later if we want
            circuit_info = session.get_circuit_info()
            if circuit_info is not None:
                corners = circuit_info.corners
                c_x = corners['X'].tolist()
                c_y = corners['Y'].tolist()
                c_num = corners['Number'].tolist()
            else:
                c_x, c_y, c_num = [], [], []

            map_data.append({
                'EventName': track,
                'Year': int(recent_year),
                'X_Data': x_coords,       # Store as list/array
                'Y_Data': y_coords,
                'Corner_X': c_x,
                'Corner_Y': c_y,
                'Corner_Num': c_num
            })
            
        except Exception as e:
            print(f"     ❌ Failed: {e}")
            continue

    # Save to Parquet
    if map_data:
        map_df = pd.DataFrame(map_data)
        output_path = 'data/track_maps.parquet'
        
        if not os.path.exists('data'):
            os.makedirs('data')
            
        map_df.to_parquet(output_path, index=False)
        print(f"✅ Success! Saved {len(map_df)} track maps to {output_path}")
    else:
        print("❌ No maps generated.")

if __name__ == "__main__":
    generate_track_maps()
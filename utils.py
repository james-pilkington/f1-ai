import streamlit as st
import pandas as pd
import numpy as np
import fastf1
import fastf1.plotting
import pickle
import os
from datetime import datetime, timedelta

# --- 1. SHARED CONFIGURATION ---
def setup_app():
    # Page Config (Call this at start of every page)
    st.set_page_config(
        page_title="F1 AI Strategist",
        page_icon="🏎️",
        layout="wide"
    )
    # Cache & Plotting Setup
    if not os.path.exists('f1_cache'):
        os.makedirs('f1_cache')
    fastf1.Cache.enable_cache('f1_cache') 
    fastf1.plotting.setup_mpl(misc_mpl_mods=False)

# --- 2. CONSTANTS ---
TEAM_COLORS = {
    'Mercedes': '#00D7B6', 'Red Bull Racing': '#4781D7', 'Red Bull': '#4781D7',
    'Ferrari': '#ED1131', 'McLaren': '#F47600', 'Alpine': '#00A1E8', 'Renault': '#00A1E8',
    'Racing Bulls': '#6C98FF', 'RB': '#6C98FF', 'AlphaTauri': '#2B4562', 'Toro Rosso': '#469BFF',
    'Aston Martin': '#229971', 'Aston Martin Aramco': '#229971', 'Racing Point': '#F596C8',
    'Williams': '#1868DB', 'Kick Sauber': '#01C00E', 'Sauber': '#01C00E',
    'Alfa Romeo': '#900000', 'Haas F1 Team': '#9C9FA2', 'Haas': '#9C9FA2',
}

# --- TEAM LINEAGE MAPPING ---
# Maps current team names to their historical identities for long-term analysis.
TEAM_LINEAGE = {
    'Mercedes': ['Mercedes', 'Brawn', 'Honda', 'BAR'],
    'Red Bull Racing': ['Red Bull', 'Jaguar', 'Stewart'],
    'Ferrari': ['Ferrari'],
    'McLaren': ['McLaren'],
    'Alpine': ['Alpine', 'Renault', 'Lotus F1', 'Benetton'],
    'Aston Martin': ['Aston Martin', 'Racing Point', 'Force India', 'Spyker', 'Midland', 'Jordan'],
    'RB': ['RB', 'Racing Bulls', 'AlphaTauri', 'Toro Rosso', 'Minardi'],
    'Racing Bulls': ['RB', 'Racing Bulls', 'AlphaTauri', 'Toro Rosso', 'Minardi'], # Handle variations
    'Kick Sauber': ['Kick Sauber', 'Sauber', 'Alfa Romeo', 'BMW Sauber'],
    'Haas F1 Team': ['Haas F1 Team', 'Haas'],
    'Williams': ['Williams']
}

def get_team_logo_url(team_name, season=2025):
    """
    Attempts to construct the official F1 CDN URL for a team logo.
    """
    # 1. Normalize Team Name to F1's URL format
    # e.g. "Red Bull Racing" -> "red-bull-racing"
    slug = team_name.lower().replace(" ", "-")
    
    # 2. Handle Edge Cases (Teams with complex names)
    overrides = {
        "rb": "rb",
        "racing-bulls": "rb",
        "alphatauri": "alphatauri",
        "kick-sauber": "kick-sauber",
        "sauber": "kick-sauber", # Current name
        "alfa-romeo": "alfa-romeo-racing", # Historic
        "haas": "haas-f1-team",
        "haas-f1-team": "haas-f1-team",
        "aston-martin": "aston-martin",
        "aston-martin-aramco": "aston-martin"
    }
    
    # Apply override if exists, otherwise use the slug
    final_slug = overrides.get(slug, slug)
    
    # 3. Construct URL
    # F1 uses this specific pattern for their content management system (AEM)
    return f"https://media.formula1.com/content/dam/fom-website/teams/{season}/{final_slug}-logo.png.transform/2col/image.png"

# --- 3. DATA LOADERS ---
@st.cache_data(ttl=3600)
def get_schedule(year):
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        date_cols = ['EventDate', 'Session1Date', 'Session2Date', 'Session3Date', 'Session4Date', 'Session5Date']
        for col in date_cols:
            if col in schedule.columns:
                schedule[col] = pd.to_datetime(schedule[col], errors='coerce')
        return schedule
    except:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_next_race():
    now = pd.Timestamp.now()
    year = now.year
    try:
        schedule = get_schedule(year)
        if not schedule.empty and schedule['EventDate'].dt.tz is not None:
            now = now.tz_localize(schedule['EventDate'].dt.tz)
        
        future = schedule[schedule['EventDate'] > now]
        if not future.empty:
            return future.iloc[0], year
        # Check next year
        schedule_next = get_schedule(year + 1)
        if not schedule_next.empty:
            return schedule_next.iloc[0], year + 1
    except:
        pass
    return None, None

@st.cache_data(ttl=3600)
def get_weekend_status(year, round_num):
    status = {}
    try:
        schedule = get_schedule(year)
        event = schedule[schedule['RoundNumber'] == round_num].iloc[0]
        now = pd.Timestamp.now()
        if event['Session5Date'].tzinfo is not None:
            now = now.tz_localize(event['Session5Date'].tzinfo)

        sessions = [
            ('FP1', event['Session1Date'], 'Practice 1'),
            ('FP2', event['Session2Date'], 'Practice 2'),
            ('FP3', event['Session3Date'], 'Practice 3'),
            ('Quali', event['Session4Date'], 'Qualifying'),
            ('Race', event['Session5Date'], 'Race')
        ]
        
        for short, date_obj, full in sessions:
            if pd.isna(date_obj):
                status[short] = {'state': 'N/A', 'full': full}
            elif now > (date_obj + timedelta(hours=2)):
                status[short] = {'state': 'Complete', 'full': full, 'date': date_obj}
            else:
                status[short] = {'state': 'TBD', 'full': full, 'date': date_obj}
        return status, event['EventName']
    except:
        return {}, "Unknown"

@st.cache_data
def load_data():
    data = {}
    try:
        df = pd.read_parquet('data/processed_f1_data.parquet')
        for c in ['Position', 'GridPosition', 'Points', 'RoundNumber', 'Year']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        data['history'] = df
    except: data['history'] = pd.DataFrame()
        
    try: data['training'] = pd.read_parquet('data/quali_training_data.parquet')
    except: data['training'] = pd.DataFrame()
    
    try: data['maps'] = pd.read_parquet('data/track_maps.parquet')
    except: data['maps'] = pd.DataFrame()
    return data

@st.cache_resource
def load_model():
    try:
        with open('data/quali_model.pkl', 'rb') as f:
            return pickle.load(f)
    except: return None


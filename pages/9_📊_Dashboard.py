import streamlit as st
import pandas as pd
import json
import streamlit.components.v1 as components
from utils import setup_app, load_data, get_schedule

setup_app()
st.title("📊 Pit Wall")
st.caption("Interactive season dashboard — standings, pace and reliability, built live from local race data.")

data_store = load_data()
df = data_store['history']

if df.empty:
    st.error("No data found. Run the ETL process first.")
    st.stop()

TEAM_NORM = {
    'Red Bull': 'Red Bull Racing', 'Red Bull Racing': 'Red Bull Racing',
    'Alpine': 'Alpine', 'Alpine F1 Team': 'Alpine', 'Renault': 'Alpine', 'Lotus F1': 'Alpine',
    'Racing Bulls': 'Racing Bulls', 'RB F1 Team': 'Racing Bulls', 'RB': 'Racing Bulls',
    'AlphaTauri': 'Racing Bulls', 'Toro Rosso': 'Racing Bulls',
    'Cadillac': 'Cadillac', 'Cadillac F1 Team': 'Cadillac',
    'Haas F1 Team': 'Haas', 'Haas': 'Haas',
    'Ferrari': 'Ferrari', 'McLaren': 'McLaren', 'Mercedes': 'Mercedes',
    'Williams': 'Williams',
    'Aston Martin': 'Aston Martin', 'Aston Martin Aramco': 'Aston Martin',
    'Racing Point': 'Aston Martin', 'Force India': 'Aston Martin',
    'Audi': 'Audi', 'Kick Sauber': 'Audi', 'Sauber': 'Audi', 'Alfa Romeo': 'Audi',
}

TEAM_SWATCH = {
    'Mercedes': '#00D7B6', 'Ferrari': '#ED1131', 'McLaren': '#F47600',
    'Red Bull Racing': '#4781D7', 'Alpine': '#00A1E8', 'Racing Bulls': '#6C98FF',
    'Aston Martin': '#229971', 'Williams': '#1868DB', 'Audi': '#900000',
    'Haas': '#9C9FA2', 'Cadillac': '#7A5CFF',
}


def build_dashboard_data(history_df, year):
    d = history_df[history_df['Year'] == year].copy()
    d['TeamName'] = d['TeamName'].map(lambda t: TEAM_NORM.get(t, t))
    race = d[d['SessionType'] == 'Race'].copy()
    sprint = d[d['SessionType'] == 'Sprint'].copy()
    race['Position'] = pd.to_numeric(race['Position'], errors='coerce')
    race['GridPosition'] = pd.to_numeric(race['GridPosition'], errors='coerce')

    if race.empty:
        return None

    n_rounds = int(race['RoundNumber'].max())
    event_names = race.drop_duplicates('RoundNumber').sort_values('RoundNumber').set_index('RoundNumber')['EventName'].to_dict()
    latest_team = race.sort_values(['RoundNumber']).groupby('Driver')['TeamName'].last().to_dict()

    pts_all = pd.concat([
        race[['Driver', 'TeamName', 'Points', 'RoundNumber']],
        sprint[['Driver', 'TeamName', 'Points', 'RoundNumber']],
    ])
    driver_points = pts_all.groupby('Driver')['Points'].sum()
    team_points = pts_all.groupby('TeamName')['Points'].sum().sort_values(ascending=False)

    wins = race[race['Position'] == 1].groupby('Driver').size()
    podiums = race[race['Position'] <= 3].groupby('Driver').size()
    gridp1 = race[race['GridPosition'] == 1].groupby('Driver').size()
    starts = race.groupby('Driver').size()
    dnf_mask = ~race['Status'].isin(['Finished', 'Lapped'])
    dnfs = race[dnf_mask].groupby('Driver').size()
    avg_finish = race.groupby('Driver')['Position'].mean()
    avg_grid = race.groupby('Driver')['GridPosition'].mean()
    avg_delta = avg_grid - avg_finish

    drivers_sorted = driver_points.sort_values(ascending=False).index.tolist()
    drivers_sorted = [dv for dv in drivers_sorted if starts.get(dv, 0) > 0]

    driver_rows = []
    for dv in drivers_sorted:
        driver_rows.append({
            'driver': dv,
            'team': latest_team.get(dv, 'Unknown'),
            'points': float(driver_points.get(dv, 0)),
            'wins': int(wins.get(dv, 0)),
            'podiums': int(podiums.get(dv, 0)),
            'gridP1': int(gridp1.get(dv, 0)),
            'starts': int(starts.get(dv, 0)),
            'dnfs': int(dnfs.get(dv, 0)),
            'avgFinish': round(float(avg_finish.get(dv, 0)), 2),
            'avgGrid': round(float(avg_grid.get(dv, 0)), 2),
            'avgDelta': round(float(avg_delta.get(dv, 0)), 2),
        })

    team_rows = []
    for tm, pts in team_points.items():
        team_rows.append({'team': tm, 'points': float(pts), 'color': TEAM_SWATCH.get(tm, '#888888')})

    top8 = drivers_sorted[:8]
    prog = {dv: [0.0] * (n_rounds + 1) for dv in top8}
    cum = {dv: 0.0 for dv in drivers_sorted}
    for rnd in range(1, n_rounds + 1):
        round_pts = pts_all[pts_all['RoundNumber'] == rnd].groupby('Driver')['Points'].sum()
        for dv in drivers_sorted:
            cum[dv] = cum.get(dv, 0.0) + float(round_pts.get(dv, 0.0))
        for dv in top8:
            prog[dv][rnd] = cum[dv]

    chaos = []
    for rnd in range(1, n_rounds + 1):
        r = race[race['RoundNumber'] == rnd]
        total = len(r)
        dnf_ct = int((~r['Status'].isin(['Finished', 'Lapped'])).sum())
        chaos.append({
            'round': rnd,
            'event': event_names.get(rnd, ''),
            'dnfRate': round(dnf_ct / total, 3) if total else 0,
            'dnfCount': dnf_ct,
            'field': total,
        })

    return {
        'season': int(year),
        'roundsCompleted': n_rounds,
        'drivers': driver_rows,
        'teams': team_rows,
        'progression': {'rounds': list(range(0, n_rounds + 1)), 'eventNames': event_names, 'series': prog},
        'chaos': chaos,
        'teamColors': TEAM_SWATCH,
    }


years = sorted(df['Year'].dropna().unique().astype(int).tolist(), reverse=True)
selected_year = st.selectbox("Season", years, index=0)

dash_data = build_dashboard_data(df, selected_year)

if dash_data is None:
    st.warning(f"No completed races found for {selected_year} yet.")
    st.stop()

try:
    schedule = get_schedule(selected_year)
    if not schedule.empty:
        total_rounds = int((schedule['EventFormat'] != 'testing').sum())
    else:
        total_rounds = dash_data['roundsCompleted']
except Exception:
    total_rounds = dash_data['roundsCompleted']

dash_data['totalRounds'] = max(total_rounds, dash_data['roundsCompleted'])

HTML_TEMPLATE = r"""
<style>
  :root {
    color-scheme: light;
    --surface-0: #eef1f4;
    --surface-1: #ffffff;
    --surface-2: #f4f6f8;
    --border: #dde2e8;
    --border-strong: #c7ced7;
    --text-primary: #12151b;
    --text-secondary: #545e6b;
    --text-muted: #63707e;
    --accent: #6a46e0;
    --accent-ink: #ffffff;
    --track: #e7eaee;
    --gain: #008300;
    --loss: #c92a2a;
    --zero: #66707d;
    --chaos-1: #fde9c8;
    --chaos-2: #f7c96b;
    --chaos-3: #f2994a;
    --chaos-4: #e2572b;
    --chaos-5: #b3261e;
    --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100;
    --s5: #e87ba4; --s6: #008300; --s7: #4a3aa7; --s8: #e34948;
    --shadow: 0 1px 2px rgba(20,24,32,0.04), 0 8px 24px -12px rgba(20,24,32,0.14);
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      color-scheme: dark;
      --surface-0: #0c0f14;
      --surface-1: #151a21;
      --surface-2: #1b212a;
      --border: #262e39;
      --border-strong: #333d4a;
      --text-primary: #edf0f4;
      --text-secondary: #a3adba;
      --text-muted: #6d7683;
      --accent: #9c7dff;
      --accent-ink: #10121a;
      --track: #1f2530;
      --gain: #35b04a;
      --loss: #ef6a68;
      --zero: #6d7683;
      --chaos-1: #3a2f1c; --chaos-2: #6b4a1f; --chaos-3: #c9772f;
      --chaos-4: #e2622f; --chaos-5: #ff6e52;
      --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500;
      --s5: #d55181; --s6: #2fae3f; --s7: #9085e9; --s8: #e66767;
      --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 12px 32px -14px rgba(0,0,0,0.6);
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --surface-0: #0c0f14;
    --surface-1: #151a21;
    --surface-2: #1b212a;
    --border: #262e39;
    --border-strong: #333d4a;
    --text-primary: #edf0f4;
    --text-secondary: #a3adba;
    --text-muted: #6d7683;
    --accent: #9c7dff;
    --accent-ink: #10121a;
    --track: #1f2530;
    --gain: #35b04a;
    --loss: #ef6a68;
    --zero: #6d7683;
    --chaos-1: #3a2f1c; --chaos-2: #6b4a1f; --chaos-3: #c9772f;
    --chaos-4: #e2622f; --chaos-5: #ff6e52;
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500;
    --s5: #d55181; --s6: #2fae3f; --s7: #9085e9; --s8: #e66767;
    --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 12px 32px -14px rgba(0,0,0,0.6);
  }

  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; }
  body {
    background: var(--surface-0);
    color: var(--text-primary);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  .mono {
    font-family: ui-monospace, "SF Mono", "Cascadia Code", "JetBrains Mono", Consolas, "Roboto Mono", monospace;
    font-variant-numeric: tabular-nums;
  }
  a { color: inherit; }

  .page {
    max-width: 1180px;
    margin: 0 auto;
    padding: 20px 20px 40px;
    display: flex;
    flex-direction: column;
    gap: 22px;
  }

  header.top {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 16px;
    padding-bottom: 4px;
    border-bottom: 1px solid var(--border);
  }
  .eyebrow {
    font-family: ui-monospace, "SF Mono", Consolas, monospace;
    font-size: 11.5px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--accent);
    font-weight: 700;
    margin: 0 0 6px;
  }
  h1.title {
    font-family: ui-monospace, "SF Mono", Consolas, monospace;
    font-weight: 800;
    font-size: clamp(30px, 5vw, 44px);
    letter-spacing: -0.01em;
    margin: 0 0 6px;
    text-wrap: balance;
  }
  .subtitle {
    color: var(--text-secondary);
    font-size: 14.5px;
    margin: 0 0 18px;
    max-width: 56ch;
  }
  .theme-toggle {
    display: flex;
    gap: 2px;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 3px;
    flex-shrink: 0;
  }
  .theme-toggle button {
    border: none;
    background: transparent;
    color: var(--text-muted);
    font-size: 12px;
    font-weight: 600;
    padding: 6px 12px;
    border-radius: 999px;
    cursor: pointer;
    font-family: inherit;
  }
  .theme-toggle button.active {
    background: var(--surface-1);
    color: var(--text-primary);
    box-shadow: var(--shadow);
  }
  .theme-toggle button:focus-visible, button:focus-visible, [tabindex]:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
  }

  .kpi-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 10px;
  }
  .kpi {
    position: relative;
    background: var(--surface-1);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 16px 14px 18px;
    box-shadow: var(--shadow);
    overflow: hidden;
  }
  .kpi::before {
    content: "";
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 4px;
    background: var(--kpi-accent, var(--accent));
  }
  .kpi .label {
    font-size: 10.5px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-muted);
    font-weight: 700;
    margin-bottom: 6px;
  }
  .kpi .value {
    font-size: 24px;
    font-weight: 700;
    line-height: 1.1;
    margin-bottom: 4px;
  }
  .kpi .caption {
    font-size: 12.5px;
    color: var(--text-secondary);
  }

  .panel {
    background: var(--surface-1);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px 20px 18px;
    box-shadow: var(--shadow);
  }
  .panel-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 14px;
  }
  .panel-head h2 {
    font-size: 15.5px;
    font-weight: 700;
    margin: 0;
  }
  .panel-head .desc {
    font-size: 12.5px;
    color: var(--text-muted);
    margin-top: 2px;
  }
  .tabs {
    display: flex;
    gap: 2px;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 9px;
    padding: 3px;
  }
  .tabs button {
    border: none;
    background: transparent;
    color: var(--text-secondary);
    font-size: 12.5px;
    font-weight: 600;
    padding: 6px 13px;
    border-radius: 6px;
    cursor: pointer;
    font-family: inherit;
  }
  .tabs button.active {
    background: var(--surface-1);
    color: var(--text-primary);
    box-shadow: var(--shadow);
  }

  .bar-list { display: flex; flex-direction: column; gap: 5px; }
  .bar-row {
    display: grid;
    grid-template-columns: 22px 16px 116px 1fr 56px;
    align-items: center;
    gap: 10px;
    padding: 5px 6px;
    border-radius: 8px;
    cursor: default;
  }
  .bar-row:hover { background: var(--surface-2); }
  .bar-row .rank { color: var(--text-muted); font-size: 12px; text-align: right; }
  .swatch { width: 10px; height: 10px; border-radius: 3px; box-shadow: 0 0 0 1px var(--border-strong) inset; }
  .bar-row .who { min-width: 0; }
  .bar-row .who .name { font-weight: 700; font-size: 13.5px; }
  .bar-row .who .team { font-size: 11px; color: var(--text-muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .track { position: relative; height: 20px; background: var(--track); border-radius: 6px; overflow: hidden; }
  .fill { position: absolute; top: 0; bottom: 0; left: 0; border-radius: 6px; transition: width 0.5s cubic-bezier(.2,.8,.2,1); }
  .bar-row .val { font-size: 13px; font-weight: 700; text-align: right; }

  .legend { display: flex; flex-wrap: wrap; gap: 4px 14px; margin-bottom: 10px; }
  .legend-item {
    display: flex; align-items: center; gap: 6px;
    font-size: 12px; color: var(--text-secondary); cursor: pointer;
    padding: 3px 6px; border-radius: 6px; user-select: none;
  }
  .legend-item:hover { background: var(--surface-2); }
  .legend-item .dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
  .legend-item.dim { opacity: 0.32; }
  .legend-item .lname { font-weight: 700; color: var(--text-primary); }

  .chart-wrap { position: relative; }
  svg.chart { display: block; width: 100%; height: auto; overflow: visible; }
  .gridline { stroke: var(--border); stroke-width: 1; }
  .axis-label { fill: var(--text-muted); font-size: 10.5px; font-family: ui-monospace, monospace; }
  .series-line { fill: none; stroke-width: 2.25; stroke-linecap: round; stroke-linejoin: round; transition: opacity 0.15s; }
  .series-dot { transition: opacity 0.15s; }
  .crosshair { stroke: var(--text-muted); stroke-width: 1; stroke-dasharray: 3 3; pointer-events: none; opacity: 0; }

  .tooltip {
    position: fixed;
    z-index: 50;
    background: var(--surface-1);
    border: 1px solid var(--border-strong);
    border-radius: 10px;
    padding: 10px 12px;
    font-size: 12px;
    box-shadow: var(--shadow);
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.1s;
    max-width: 240px;
  }
  .tooltip.show { opacity: 1; }
  .tooltip .t-title { font-weight: 700; margin-bottom: 6px; font-size: 12.5px; }
  .tooltip .t-row { display: flex; justify-content: space-between; gap: 14px; padding: 1.5px 0; color: var(--text-secondary); }
  .tooltip .t-row b { color: var(--text-primary); font-weight: 700; }
  .tooltip .t-row .swatch { width: 7px; height: 7px; margin-right: 5px; }

  .perf-list { display: flex; flex-direction: column; gap: 4px; }
  .perf-row { display: grid; grid-template-columns: 96px 1fr 54px; align-items: center; gap: 10px; padding: 4px 6px; border-radius: 8px; }
  .perf-row:hover { background: var(--surface-2); }
  .perf-row .who { font-size: 12.5px; font-weight: 700; text-align: right; }
  .perf-track { position: relative; height: 16px; background: var(--track); border-radius: 5px; }
  .perf-zero { position: absolute; top: -3px; bottom: -3px; left: 50%; width: 1px; background: var(--border-strong); }
  .perf-fill { position: absolute; top: 0; bottom: 0; border-radius: 4px; }
  .perf-row .delta { font-size: 12.5px; font-weight: 700; }

  .chaos-strip { display: flex; align-items: flex-end; gap: 6px; height: 108px; padding: 0 2px; }
  .chaos-bar-wrap { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: flex-end; height: 100%; gap: 6px; cursor: default; }
  .chaos-bar { width: 100%; max-width: 30px; border-radius: 4px 4px 2px 2px; min-height: 3px; }
  .chaos-round { font-size: 9.5px; color: var(--text-muted); font-family: ui-monospace, monospace; }

  .table-wrap { overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 12.5px; min-width: 720px; }
  thead th {
    text-align: right;
    font-size: 10.5px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--text-muted);
    font-weight: 700;
    padding: 8px 10px;
    border-bottom: 1px solid var(--border-strong);
    cursor: pointer;
    white-space: nowrap;
    user-select: none;
  }
  thead th:hover { color: var(--text-primary); }
  thead th.col-name, thead th.col-team { text-align: left; }
  thead th .arrow { opacity: 0.4; margin-left: 2px; }
  thead th.sorted .arrow { opacity: 1; color: var(--accent); }
  tbody td { padding: 8px 10px; border-bottom: 1px solid var(--border); text-align: right; white-space: nowrap; }
  tbody td.col-name, tbody td.col-team { text-align: left; }
  tbody tr:hover { background: var(--surface-2); }
  tbody td.col-name .name-cell { display: flex; align-items: center; gap: 8px; font-weight: 700; }
  .delta-pos { color: var(--gain); }
  .delta-neg { color: var(--loss); }
  .delta-zero { color: var(--zero); }

  footer.foot {
    text-align: center;
    color: var(--text-muted);
    font-size: 11.5px;
    padding-top: 6px;
  }

  @media (max-width: 640px) {
    .bar-row { grid-template-columns: 18px 14px 84px 1fr 46px; }
    .perf-row { grid-template-columns: 68px 1fr 44px; }
  }
</style>

<div class="page">
  <header class="top">
    <div>
      <p class="eyebrow" id="eyebrowText">Loading&hellip;</p>
      <h1 class="title">PIT WALL</h1>
      <p class="subtitle" id="subtitleText">Standings, pace and reliability &mdash; built live from local race-by-race timing data.</p>
    </div>
    <div class="theme-toggle" role="group" aria-label="Theme">
      <button data-theme-btn="light" type="button">Light</button>
      <button data-theme-btn="dark" type="button">Dark</button>
      <button data-theme-btn="system" type="button">Auto</button>
    </div>
  </header>

  <section class="kpi-row" id="kpiRow"></section>

  <section class="panel">
    <div class="panel-head">
      <div>
        <h2 id="standingsTitle">Driver Championship</h2>
        <div class="desc" id="standingsDesc">Hover a row for the full breakdown</div>
      </div>
      <div class="tabs" id="standingsTabs">
        <button data-mode="drivers" class="active" type="button">Drivers</button>
        <button data-mode="teams" type="button">Constructors</button>
      </div>
    </div>
    <div class="bar-list" id="standingsList"></div>
  </section>

  <section class="panel">
    <div class="panel-head">
      <div>
        <h2>Championship Race</h2>
        <div class="desc">Cumulative points by round, top 8 &middot; click a name to isolate it</div>
      </div>
    </div>
    <div class="legend" id="progLegend"></div>
    <div class="chart-wrap">
      <svg class="chart" id="progChart" viewBox="0 0 1000 340" preserveAspectRatio="none"></svg>
    </div>
  </section>

  <section class="panel">
    <div class="panel-head">
      <div>
        <h2>Race-Day Performers</h2>
        <div class="desc">Average grid &rarr; finish position change &middot; positive = gains places on race day &middot; min. 5 starts</div>
      </div>
    </div>
    <div class="perf-list" id="perfList"></div>
  </section>

  <section class="panel">
    <div class="panel-head">
      <div>
        <h2>Season Chaos Index</h2>
        <div class="desc">Share of the field retired or DNS, by round</div>
      </div>
    </div>
    <div class="chaos-strip" id="chaosStrip"></div>
  </section>

  <section class="panel">
    <div class="panel-head">
      <div>
        <h2>Full Grid</h2>
        <div class="desc">Every classified driver this season &middot; click a column to sort</div>
      </div>
    </div>
    <div class="table-wrap">
      <table id="gridTable">
        <thead>
          <tr>
            <th class="col-rank" data-key="rank">#</th>
            <th class="col-name" data-key="driver">Driver</th>
            <th class="col-team" data-key="team">Team</th>
            <th data-key="points">Pts <span class="arrow">&#9662;</span></th>
            <th data-key="wins">W</th>
            <th data-key="podiums">Pod</th>
            <th data-key="gridP1">P1</th>
            <th data-key="avgGrid">Avg Grid</th>
            <th data-key="avgFinish">Avg Fin</th>
            <th data-key="avgDelta">&Delta;</th>
            <th data-key="dnfs">DNF</th>
          </tr>
        </thead>
        <tbody id="gridTbody"></tbody>
      </table>
    </div>
  </section>

  <footer class="foot" id="footerText">Generated from local race-by-race data</footer>
</div>

<div class="tooltip" id="tooltip"></div>

<script id="data" type="application/json">
__DATA_JSON__
</script>

<script>
(function () {
  var DATA = JSON.parse(document.getElementById('data').textContent);
  var SLOTS = ['s1','s2','s3','s4','s5','s6','s7','s8'];
  var cssv = function (name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); };
  var tooltip = document.getElementById('tooltip');

  function showTooltip(x, y, html) {
    tooltip.innerHTML = html;
    tooltip.classList.add('show');
    var pad = 16;
    var rect = tooltip.getBoundingClientRect();
    var left = x + 14, top = y + 14;
    if (left + rect.width + pad > window.innerWidth) left = x - rect.width - 14;
    if (top + rect.height + pad > window.innerHeight) top = y - rect.height - 14;
    tooltip.style.left = left + 'px';
    tooltip.style.top = top + 'px';
  }
  function hideTooltip() { tooltip.classList.remove('show'); }

  function reportHeight() {
    try {
      var h = document.documentElement.scrollHeight;
      window.parent.postMessage({ type: 'streamlit:setFrameHeight', height: h }, '*');
    } catch (e) {}
  }

  /* ---------------- theme ---------------- */
  function applyTheme(mode) {
    if (mode === 'system') { document.documentElement.removeAttribute('data-theme'); }
    else { document.documentElement.setAttribute('data-theme', mode); }
    document.querySelectorAll('[data-theme-btn]').forEach(function (b) {
      b.classList.toggle('active', b.getAttribute('data-theme-btn') === mode);
    });
    try { localStorage.setItem('pitwall-theme', mode); } catch (e) {}
  }
  var savedTheme = 'system';
  try { savedTheme = localStorage.getItem('pitwall-theme') || 'system'; } catch (e) {}
  applyTheme(savedTheme);
  document.querySelectorAll('[data-theme-btn]').forEach(function (b) {
    b.addEventListener('click', function () { applyTheme(b.getAttribute('data-theme-btn')); renderAll(); });
  });

  /* ---------------- header ---------------- */
  function renderHeader() {
    document.getElementById('eyebrowText').textContent =
      DATA.season + ' Season · Round ' + DATA.roundsCompleted + ' of ' + DATA.totalRounds;
    document.getElementById('subtitleText').textContent =
      'Standings, pace and reliability from the ' + DATA.season + ' F1 season — built live from local race-by-race timing data.';
    document.getElementById('footerText').textContent =
      'Generated from local race-by-race data · ' + DATA.season + ' season, Rounds 1–' + DATA.roundsCompleted + ' · F1 AI Strategist project';
  }

  /* ---------------- KPIs ---------------- */
  function renderKpis() {
    var drivers = DATA.drivers.slice().sort(function (a, b) { return b.points - a.points; });
    var leader = drivers[0], second = drivers[1] || drivers[0];
    var wins = drivers.slice().sort(function (a, b) { return b.wins - a.wins; })[0];
    var clean = drivers.filter(function (d) { return d.starts >= Math.max(3, DATA.roundsCompleted - 3); })
      .sort(function (a, b) { return a.dnfs - b.dnfs || b.points - a.points; })[0] || drivers[0];
    var avgChaos = DATA.chaos.length ? DATA.chaos.reduce(function (s, c) { return s + c.dnfRate; }, 0) / DATA.chaos.length : 0;
    var wildest = DATA.chaos.slice().sort(function (a, b) { return b.dnfRate - a.dnfRate; })[0];
    var teamColor = function (t) { return DATA.teamColors[t] || cssv('--accent'); };

    var tiles = [
      { accent: teamColor(leader.team), label: 'Championship Leader', value: leader.driver, caption: leader.team + ' &middot; ' + leader.points.toFixed(0) + ' pts, +' + (leader.points - second.points).toFixed(0) + ' over ' + second.driver },
      { accent: teamColor(wins.team), label: 'Most Wins', value: wins.wins + ' &mdash; ' + wins.driver, caption: wins.team + ' &middot; ' + wins.podiums + ' podiums this season' },
      { accent: teamColor(clean.team), label: 'Cleanest Season', value: clean.driver, caption: clean.dnfs + ' DNF' + (clean.dnfs === 1 ? '' : 's') + ' in ' + clean.starts + ' starts' },
    ];
    if (wildest) {
      tiles.push({ accent: 'var(--chaos-4)', label: 'Season Chaos Index', value: (avgChaos * 100).toFixed(0) + '%', caption: 'avg. retirement rate &middot; peak ' + wildest.event.replace(' Grand Prix', ' GP') + ' (' + (wildest.dnfRate * 100).toFixed(0) + '%)' });
    }
    tiles.push({ accent: 'var(--accent)', label: 'Rounds Complete', value: DATA.roundsCompleted + ' / ' + DATA.totalRounds, caption: Math.max(DATA.totalRounds - DATA.roundsCompleted, 0) + ' races remaining this season' });

    document.getElementById('kpiRow').innerHTML = tiles.map(function (t) {
      return '<div class="kpi" style="--kpi-accent:' + t.accent + '"><div class="label">' + t.label + '</div><div class="value">' + t.value + '</div><div class="caption">' + t.caption + '</div></div>';
    }).join('');
  }

  /* ---------------- standings bars ---------------- */
  var standingsMode = 'drivers';
  function renderStandings() {
    var listEl = document.getElementById('standingsList');
    var titleEl = document.getElementById('standingsTitle');
    var descEl = document.getElementById('standingsDesc');
    if (standingsMode === 'drivers') {
      titleEl.textContent = 'Driver Championship';
      descEl.textContent = 'Points after Round ' + DATA.roundsCompleted + ' · hover a row for the full breakdown';
      var rows = DATA.drivers.slice().sort(function (a, b) { return b.points - a.points; });
      var max = rows[0].points || 1;
      listEl.innerHTML = rows.map(function (d, i) {
        var color = DATA.teamColors[d.team] || '#888';
        var pct = Math.max((d.points / max) * 100, d.points > 0 ? 2 : 0);
        return '<div class="bar-row" data-tip="drv" data-idx="' + i + '">' +
          '<div class="rank mono">' + (i + 1) + '</div>' +
          '<div class="swatch" style="background:' + color + '"></div>' +
          '<div class="who"><div class="name">' + d.driver + '</div><div class="team">' + d.team + '</div></div>' +
          '<div class="track"><div class="fill" style="width:' + pct + '%;background:' + color + '"></div></div>' +
          '<div class="val mono">' + d.points.toFixed(0) + '</div>' +
          '</div>';
      }).join('');
      listEl.querySelectorAll('.bar-row').forEach(function (el) {
        el.addEventListener('mousemove', function (e) {
          var d = rows[+el.getAttribute('data-idx')];
          showTooltip(e.clientX, e.clientY,
            '<div class="t-title">' + d.driver + ' — ' + d.team + '</div>' +
            '<div class="t-row"><span>Points</span><b>' + d.points.toFixed(0) + '</b></div>' +
            '<div class="t-row"><span>Wins / Podiums</span><b>' + d.wins + ' / ' + d.podiums + '</b></div>' +
            '<div class="t-row"><span>Grid P1s</span><b>' + d.gridP1 + '</b></div>' +
            '<div class="t-row"><span>Avg grid → finish</span><b>' + d.avgGrid.toFixed(1) + ' → ' + d.avgFinish.toFixed(1) + '</b></div>' +
            '<div class="t-row"><span>DNFs</span><b>' + d.dnfs + ' / ' + d.starts + '</b></div>');
        });
        el.addEventListener('mouseleave', hideTooltip);
      });
    } else {
      titleEl.textContent = 'Constructor Championship';
      descEl.textContent = 'Combined team points after Round ' + DATA.roundsCompleted;
      var trows = DATA.teams.slice().sort(function (a, b) { return b.points - a.points; });
      var tmax = trows[0].points || 1;
      var byTeam = {};
      DATA.drivers.forEach(function (d) { (byTeam[d.team] = byTeam[d.team] || []).push(d); });
      listEl.innerHTML = trows.map(function (t, i) {
        var pct = Math.max((t.points / tmax) * 100, t.points > 0 ? 2 : 0);
        var lineup = (byTeam[t.team] || []).map(function (d) { return d.driver; }).join(' / ');
        return '<div class="bar-row" data-tip="team" data-idx="' + i + '">' +
          '<div class="rank mono">' + (i + 1) + '</div>' +
          '<div class="swatch" style="background:' + t.color + '"></div>' +
          '<div class="who"><div class="name">' + t.team + '</div><div class="team">' + lineup + '</div></div>' +
          '<div class="track"><div class="fill" style="width:' + pct + '%;background:' + t.color + '"></div></div>' +
          '<div class="val mono">' + t.points.toFixed(0) + '</div>' +
          '</div>';
      }).join('');
      listEl.querySelectorAll('.bar-row').forEach(function (el) {
        el.addEventListener('mousemove', function (e) {
          var t = trows[+el.getAttribute('data-idx')];
          var lineup = byTeam[t.team] || [];
          showTooltip(e.clientX, e.clientY,
            '<div class="t-title">' + t.team + '</div>' +
            '<div class="t-row"><span>Points</span><b>' + t.points.toFixed(0) + '</b></div>' +
            lineup.map(function (d) { return '<div class="t-row"><span>' + d.driver + '</span><b>' + d.points.toFixed(0) + ' pts</b></div>'; }).join(''));
        });
        el.addEventListener('mouseleave', hideTooltip);
      });
    }
  }
  document.getElementById('standingsTabs').addEventListener('click', function (e) {
    var btn = e.target.closest('button[data-mode]');
    if (!btn) return;
    standingsMode = btn.getAttribute('data-mode');
    document.querySelectorAll('#standingsTabs button').forEach(function (b) { b.classList.toggle('active', b === btn); });
    renderStandings();
    reportHeight();
  });

  /* ---------------- progression line chart ---------------- */
  var isolated = null;
  function renderProgression() {
    var series = DATA.progression.series;
    var rounds = DATA.progression.rounds;
    var names = Object.keys(series);
    if (!names.length || rounds.length < 2) {
      document.getElementById('progChart').innerHTML = '';
      document.getElementById('progLegend').innerHTML = '';
      return;
    }
    var W = 1000, H = 340, padL = 40, padR = 14, padT = 14, padB = 30;
    var innerW = W - padL - padR, innerH = H - padT - padB;
    var maxY = 0;
    names.forEach(function (n) { series[n].forEach(function (v) { if (v > maxY) maxY = v; }); });
    maxY = Math.ceil(maxY / 25) * 25 || 25;
    var x = function (i) { return padL + (i / (rounds.length - 1)) * innerW; };
    var y = function (v) { return padT + innerH - (v / maxY) * innerH; };

    var legend = document.getElementById('progLegend');
    legend.innerHTML = names.map(function (n, i) {
      var d = DATA.drivers.filter(function (dd) { return dd.driver === n; })[0];
      return '<div class="legend-item' + (isolated && isolated !== n ? ' dim' : '') + '" data-name="' + n + '"><span class="dot" style="background:var(--' + SLOTS[i % 8] + ')"></span><span class="lname">' + n + '</span><span>' + (d ? d.team : '') + '</span></div>';
    }).join('');
    legend.querySelectorAll('.legend-item').forEach(function (el) {
      el.addEventListener('click', function () {
        var n = el.getAttribute('data-name');
        isolated = (isolated === n) ? null : n;
        renderProgression();
      });
    });

    var gridlines = '', yLabels = '';
    for (var g = 0; g <= 4; g++) {
      var gv = (maxY / 4) * g;
      var gy = y(gv);
      gridlines += '<line class="gridline" x1="' + padL + '" x2="' + (W - padR) + '" y1="' + gy + '" y2="' + gy + '"></line>';
      yLabels += '<text class="axis-label" x="' + (padL - 8) + '" y="' + (gy + 3) + '" text-anchor="end">' + gv.toFixed(0) + '</text>';
    }
    var xLabels = '';
    rounds.forEach(function (r, i) {
      if (r === 0) return;
      if (r % 2 !== 0 && rounds.length - 1 > 10) return;
      xLabels += '<text class="axis-label" x="' + x(i) + '" y="' + (H - 8) + '" text-anchor="middle">R' + r + '</text>';
    });

    var paths = '', dots = '';
    names.forEach(function (n, i) {
      var vals = series[n];
      var d = vals.map(function (v, idx) { return (idx === 0 ? 'M' : 'L') + x(idx).toFixed(1) + ',' + y(v).toFixed(1); }).join(' ');
      var dim = isolated && isolated !== n;
      paths += '<path class="series-line" data-name="' + n + '" d="' + d + '" stroke="var(--' + SLOTS[i % 8] + ')" style="opacity:' + (dim ? 0.14 : 1) + '"></path>';
      var lastIdx = vals.length - 1;
      dots += '<circle class="series-dot" data-name="' + n + '" cx="' + x(lastIdx) + '" cy="' + y(vals[lastIdx]) + '" r="3.5" fill="var(--' + SLOTS[i % 8] + ')" style="opacity:' + (dim ? 0.14 : 1) + '"></circle>';
    });

    var svg = document.getElementById('progChart');
    svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
    svg.innerHTML = gridlines + yLabels + xLabels + paths + dots +
      '<line class="crosshair" id="progCrosshair" x1="0" x2="0" y1="' + padT + '" y2="' + (H - padB) + '"></line>' +
      '<rect id="progHit" x="' + padL + '" y="' + padT + '" width="' + innerW + '" height="' + innerH + '" fill="transparent"></rect>';

    var hit = document.getElementById('progHit');
    var crosshair = document.getElementById('progCrosshair');
    hit.addEventListener('mousemove', function (e) {
      var rect = svg.getBoundingClientRect();
      var svgX = ((e.clientX - rect.left) / rect.width) * W;
      var idx = Math.round(((svgX - padL) / innerW) * (rounds.length - 1));
      idx = Math.max(0, Math.min(rounds.length - 1, idx));
      var cx = x(idx);
      crosshair.setAttribute('x1', cx); crosshair.setAttribute('x2', cx);
      crosshair.style.opacity = 1;
      var rows = names.map(function (n, i) { return { n: n, v: series[n][idx], color: 'var(--' + SLOTS[i % 8] + ')' }; })
        .sort(function (a, b) { return b.v - a.v; });
      var evName = idx > 0 ? DATA.progression.eventNames[idx] : 'Season start';
      showTooltip(e.clientX, e.clientY,
        '<div class="t-title">' + (idx > 0 ? 'R' + idx + ' · ' : '') + evName + '</div>' +
        rows.map(function (r) { return '<div class="t-row"><span><span class="swatch" style="background:' + r.color + '"></span>' + r.n + '</span><b>' + r.v.toFixed(0) + '</b></div>'; }).join(''));
    });
    hit.addEventListener('mouseleave', function () { crosshair.style.opacity = 0; hideTooltip(); });
  }

  /* ---------------- race-day performers ---------------- */
  function renderPerformers() {
    var rows = DATA.drivers.filter(function (d) { return d.starts >= 5; })
      .slice().sort(function (a, b) { return b.avgDelta - a.avgDelta; });
    if (!rows.length) rows = DATA.drivers.slice().sort(function (a, b) { return b.avgDelta - a.avgDelta; });
    var maxAbs = Math.max.apply(null, rows.map(function (d) { return Math.abs(d.avgDelta); }).concat([1]));
    document.getElementById('perfList').innerHTML = rows.map(function (d, i) {
      var pct = (Math.abs(d.avgDelta) / maxAbs) * 50;
      var color = d.avgDelta > 0.05 ? 'var(--gain)' : (d.avgDelta < -0.05 ? 'var(--loss)' : 'var(--zero)');
      var side = d.avgDelta >= 0 ? 'left:50%;' : 'right:50%;';
      var sign = d.avgDelta > 0 ? '+' : '';
      return '<div class="perf-row" data-idx="' + i + '">' +
        '<div class="who">' + d.driver + '</div>' +
        '<div class="perf-track"><div class="perf-zero"></div><div class="perf-fill" style="' + side + 'width:' + pct + '%;background:' + color + '"></div></div>' +
        '<div class="delta mono" style="color:' + color + '">' + sign + d.avgDelta.toFixed(2) + '</div>' +
        '</div>';
    }).join('');
    document.querySelectorAll('#perfList .perf-row').forEach(function (el) {
      el.addEventListener('mousemove', function (e) {
        var d = rows[+el.getAttribute('data-idx')];
        showTooltip(e.clientX, e.clientY,
          '<div class="t-title">' + d.driver + ' — ' + d.team + '</div>' +
          '<div class="t-row"><span>Avg grid</span><b>' + d.avgGrid.toFixed(1) + '</b></div>' +
          '<div class="t-row"><span>Avg finish</span><b>' + d.avgFinish.toFixed(1) + '</b></div>' +
          '<div class="t-row"><span>Net change / race</span><b>' + (d.avgDelta > 0 ? '+' : '') + d.avgDelta.toFixed(2) + '</b></div>');
      });
      el.addEventListener('mouseleave', hideTooltip);
    });
  }

  /* ---------------- chaos strip ---------------- */
  function renderChaos() {
    var maxRate = Math.max.apply(null, DATA.chaos.map(function (c) { return c.dnfRate; }).concat([0.001]));
    function chaosColor(rate) {
      var t = maxRate ? rate / maxRate : 0;
      var steps = ['--chaos-1', '--chaos-2', '--chaos-3', '--chaos-4', '--chaos-5'];
      var idx = Math.min(steps.length - 1, Math.floor(t * steps.length));
      return 'var(' + steps[idx] + ')';
    }
    document.getElementById('chaosStrip').innerHTML = DATA.chaos.map(function (c, i) {
      var h = Math.max(6, (c.dnfRate / (maxRate || 1)) * 84);
      return '<div class="chaos-bar-wrap" data-idx="' + i + '">' +
        '<div class="chaos-bar" style="height:' + h + 'px;background:' + chaosColor(c.dnfRate) + '"></div>' +
        '<div class="chaos-round">R' + c.round + '</div>' +
        '</div>';
    }).join('');
    document.querySelectorAll('#chaosStrip .chaos-bar-wrap').forEach(function (el) {
      el.addEventListener('mousemove', function (e) {
        var c = DATA.chaos[+el.getAttribute('data-idx')];
        showTooltip(e.clientX, e.clientY,
          '<div class="t-title">R' + c.round + ' · ' + c.event + '</div>' +
          '<div class="t-row"><span>Retired / DNS</span><b>' + c.dnfCount + ' of ' + c.field + '</b></div>' +
          '<div class="t-row"><span>Rate</span><b>' + (c.dnfRate * 100).toFixed(0) + '%</b></div>');
      });
      el.addEventListener('mouseleave', hideTooltip);
    });
  }

  /* ---------------- table ---------------- */
  var sortState = { key: 'points', dir: -1 };
  function renderTable() {
    var rows = DATA.drivers.slice().sort(function (a, b) { return b.points - a.points; })
      .map(function (d, i) { return Object.assign({ rank: i + 1 }, d); });
    var key = sortState.key, dir = sortState.dir;
    rows.sort(function (a, b) {
      var av = a[key], bv = b[key];
      if (typeof av === 'string') return dir * av.localeCompare(bv);
      return dir * (av - bv);
    });
    document.getElementById('gridTbody').innerHTML = rows.map(function (d) {
      var color = DATA.teamColors[d.team] || '#888';
      var dcls = d.avgDelta > 0.05 ? 'delta-pos' : (d.avgDelta < -0.05 ? 'delta-neg' : 'delta-zero');
      return '<tr>' +
        '<td class="col-rank mono">' + d.rank + '</td>' +
        '<td class="col-name"><div class="name-cell"><span class="swatch" style="background:' + color + '"></span>' + d.driver + '</div></td>' +
        '<td class="col-team">' + d.team + '</td>' +
        '<td class="mono">' + d.points.toFixed(0) + '</td>' +
        '<td class="mono">' + d.wins + '</td>' +
        '<td class="mono">' + d.podiums + '</td>' +
        '<td class="mono">' + d.gridP1 + '</td>' +
        '<td class="mono">' + d.avgGrid.toFixed(1) + '</td>' +
        '<td class="mono">' + d.avgFinish.toFixed(1) + '</td>' +
        '<td class="mono ' + dcls + '">' + (d.avgDelta > 0 ? '+' : '') + d.avgDelta.toFixed(2) + '</td>' +
        '<td class="mono">' + d.dnfs + '</td>' +
        '</tr>';
    }).join('');
    document.querySelectorAll('#gridTable thead th').forEach(function (th) {
      th.classList.toggle('sorted', th.getAttribute('data-key') === key);
      var arrow = th.querySelector('.arrow');
      if (th.getAttribute('data-key') === key && arrow) arrow.innerHTML = dir === 1 ? '&#9652;' : '&#9662;';
    });
  }
  document.getElementById('gridTable').querySelector('thead').addEventListener('click', function (e) {
    var th = e.target.closest('th');
    if (!th) return;
    var key = th.getAttribute('data-key');
    if (sortState.key === key) sortState.dir *= -1;
    else { sortState.key = key; sortState.dir = (key === 'driver' || key === 'team') ? 1 : -1; }
    renderTable();
  });

  function renderAll() {
    renderHeader();
    renderKpis();
    renderStandings();
    renderProgression();
    renderPerformers();
    renderChaos();
    renderTable();
    reportHeight();
    setTimeout(reportHeight, 150);
    setTimeout(reportHeight, 500);
  }
  renderAll();
  window.addEventListener('resize', function () { renderProgression(); reportHeight(); });
})();
</script>
"""

html = HTML_TEMPLATE.replace("__DATA_JSON__", json.dumps(dash_data))
components.html(html, height=2700, scrolling=True)

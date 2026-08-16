import streamlit as st
import pandas as pd
import numpy as np
import html
from datetime import date
import h_utils
import p_utils
import unicodedata

from utils import TEAM_OPTIONS, LEAGUES


st.set_page_config(page_title="Player Ranks", layout="wide", page_icon="⚾")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
    [data-testid="stToolbar"] {visibility: hidden;}
    [data-testid="stDecoration"] {display: none;}
    [data-testid="stStatusWidget"] {display: none;}
    .viewerBadge_link__qRi_k {display: none;}
    .stSelectbox div[data-baseweb="select"],
    .stNumberInput > div { max-width: 200px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("""
<style>
    @media only screen and (max-width: 600px) {
        [data-testid="stAppViewContainer"] h1 {
            font-size: 1.8rem !important;
        }
        .mobile-meta {
            font-size: 0.8rem !important;
            padding-top: 0.3rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

title_col, meta_col = st.columns([3, 1])
with title_col:
    st.title("Player Ranks")
with meta_col:
    st.markdown(
        """
        <div class="mobile-meta" style="text-align: right; font-size: 1rem; padding-top: 0.6rem;">
            Built by <a href="https://twitter.com/Sox_Savant" target="_blank">@Sox_Savant</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

type_mode = st.radio("Type", ["Hitting", "Pitching"], horizontal=True, key="pr_mode", label_visibility="collapsed")
is_hitting = (type_mode == "Hitting")
U = h_utils if is_hitting else p_utils
prefix = "hpr" if is_hitting else "ppr"

current_year = date.today().year
last_updated = U.get_last_updated(current_year)
st.caption(f"{current_year} data last updated: {last_updated}")

# NOTE: league scope is resolved via utils.LEAGUES (assumed shape: {"AL": [team
# abbrevs...], "NL": [...]}), same as your other AL/NL-filtered apps. If LEAGUES is
# shaped differently (e.g. team -> league), adjust team_in_league() below.
STAT_PRESETS = U.STAT_PRESETS_RANKS
STAT_DISPLAY_NAMES = U.STAT_DISPLAY_NAMES
STAT_ALLOWLIST = U.STAT_ALLOWLIST
TRUTHY_STRINGS = U.TRUTHY_STRINGS
get_headshot = U.get_headshot
label_map = U.label_map
lower_better = U.lower_better
DATA_START_YEAR = U.start_year
format_stat = U.format_stat
load_final_year = U.load_final_year
get_team_display = U.get_team_display
get_player_id_by_name = U.get_player_id_by_name
RATE_STATS = U.RATE_STATS

from utils import get_dynamic_min_ip, get_dynamic_min_pa

DEFAULT_PRESET = "Default"


def display_stat_name(stat) -> str:
    if stat is None:
        return ""
    return STAT_DISPLAY_NAMES.get(str(stat), str(stat))


def ordinal(n: int) -> str:
    n = int(n)
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def team_in_league(team, league) -> bool:
    teams = LEAGUES.get(league, [])
    t = str(team).strip()
    for c in getattr(U, "normalize_team", lambda x: x)(t), t:
        if c in teams:
            return True
    return False


MIN_LEAGUE_YEAR = 2013  # league splits disabled before this, matching other apps


left_col, right_col = st.columns([1, 2])

years_desc = list(range(current_year, DATA_START_YEAR - 1, -1))
default_name = "Pete Crow-Armstrong" if is_hitting else "Jacob Misiorowski"

for key, default in [
    (f"{prefix}_player",   default_name),
    ("pr_player_id",       ""),
    ("pr_player_mode",     "Name"),
    ("pr_year",            current_year),
    ("pr_scope",           "MLB"),
]:
    if key not in st.session_state:
        st.session_state[key] = default

STAT_STATE_KEY    = f"{prefix}_stat_config"
STAT_PRESET_KEY   = f"{prefix}_stat_preset"
STAT_VERSION_KEY  = f"{prefix}_stat_version"
ADD_SELECT_KEY    = f"{prefix}_add_stat_select"
REMOVE_SELECT_KEY = f"{prefix}_remove_stat_select"
ADD_RESET_KEY     = f"{prefix}_reset_add_select"
REMOVE_RESET_KEY  = f"{prefix}_reset_remove_select"

with left_col:
    mode_val = st.selectbox("Player Input", ["Name", "FanGraphs ID"], key="pr_player_mode")
    if mode_val == "Name":
        name_input = st.text_input("Player", key=f"{prefix}_player")
        id_input = st.session_state.get("pr_player_id", "")
    else:
        id_input = st.text_input("FanGraphs ID", key="pr_player_id")
        name_input = st.session_state.get(f"{prefix}_player", "")

    st.selectbox("Year", options=years_desc, key="pr_year")
    year = st.session_state["pr_year"]

    scope_options = ["MLB", "AL", "NL"] if year >= MIN_LEAGUE_YEAR else ["MLB"]
    if st.session_state.get("pr_scope") not in scope_options:
        st.session_state["pr_scope"] = "MLB"
    st.selectbox("Scope", options=scope_options, key="pr_scope")
    scope = st.session_state["pr_scope"]
    if year < MIN_LEAGUE_YEAR:
        st.caption("League splits are only available for 2013+ seasons.")

    if is_hitting:
        min_thresh = st.number_input(
            "Min PA", min_value=0, max_value=700,
            value=get_dynamic_min_pa(year), step=25,
        )
    else:
        min_thresh = st.number_input(
            "Min IP", min_value=0, max_value=700,
            value=int(get_dynamic_min_ip(year)), step=25,
        )
    thresh_col = "PA" if is_hitting else "IP"

    # ── Resolve player ──────────────────────────
    if mode_val == "Name":
        if not name_input.strip():
            st.warning("Enter a player name.")
            st.stop()
        player_id = get_player_id_by_name(name_input.strip(), year)
        if not player_id:
            st.error(f"Could not find '{name_input}' in {year}. Check spelling or use FanGraphs ID.")
            st.stop()
    else:
        if not id_input.strip():
            st.warning("Enter a FanGraphs ID.")
            st.stop()
        try:
            player_id = int(id_input.strip())
        except Exception:
            st.error("FanGraphs ID must be a positive integer.")
            st.stop()

    df_year = load_final_year(year)
    if df_year is None or df_year.empty:
        st.error(f"No data found for {year}.")
        st.stop()

    player_rows = df_year[df_year["PlayerId"] == player_id]
    if player_rows.empty:
        st.error(f"No data found for this player in {year}.")
        st.stop()
    player_row = player_rows.iloc[0]

    # ── Build ranking populations ────────────────
    # Counting stats rank against everyone in scope; rate stats additionally
    # require the min PA/IP threshold so small samples don't skew the ranks.
    scope_pool = df_year.copy()
    if scope != "MLB" and "Team" in scope_pool.columns:
        scope_pool = scope_pool[scope_pool["Team"].apply(lambda t: team_in_league(t, scope))]
    if player_id not in scope_pool["PlayerId"].values:
        scope_pool = pd.concat([scope_pool, player_rows], ignore_index=True)

    rate_pool = scope_pool.copy()
    if thresh_col in rate_pool.columns:
        rate_pool = rate_pool[pd.to_numeric(rate_pool[thresh_col], errors="coerce") >= min_thresh]
    # always include the player being viewed, even if under the threshold
    if player_id not in rate_pool["PlayerId"].values:
        rate_pool = pd.concat([rate_pool, player_rows], ignore_index=True)

    # ── Build stat options ───────────────────────
    stat_exclusions = {"Season", "PlayerId", "MLBAMID"}
    numeric_pool = {c for c in scope_pool.columns if pd.notna(pd.to_numeric(scope_pool[c], errors="coerce")).any()}
    numeric_player = {c for c in player_row.index if pd.notna(pd.to_numeric(player_row[c], errors="coerce"))}
    numeric_stats_set = (numeric_pool & numeric_player) - stat_exclusions

    preferred = [s for s in STAT_ALLOWLIST if s in numeric_stats_set]
    other = [s for s in numeric_stats_set if s not in preferred]
    stat_options = preferred + other
    allowed_add_stats = preferred if preferred else stat_options.copy()

    if not stat_options:
        st.error("No comparable numeric stats for this player/pool.")
        st.stop()

    # ── Stat builder helpers ─────────────────────

    def bump_version():
        st.session_state[STAT_VERSION_KEY] = st.session_state.get(STAT_VERSION_KEY, 0) + 1

    def normalize_stat_rows(rows, fallback):
        cleaned, seen = [], set()
        for row in rows:
            if not isinstance(row, dict):
                continue
            stat_name = row.get("Stat")
            if not stat_name or stat_name not in stat_options or stat_name in seen:
                continue
            show_val = row.get("Show", True)
            if pd.isna(show_val) if not isinstance(show_val, str) else False:
                show_bool = True
            elif isinstance(show_val, str):
                show_bool = show_val.strip().lower() in TRUTHY_STRINGS
            else:
                show_bool = bool(show_val)
            cleaned.append({"Stat": stat_name, "Show": show_bool})
            seen.add(stat_name)
        return cleaned if cleaned else [r.copy() for r in fallback]

    def get_preset_base(preset_name):
        preset_stats = STAT_PRESETS.get(preset_name, [])
        filtered = [s for s in preset_stats if s in stat_options]
        if not filtered and stat_options:
            filtered = [stat_options[0]]
        return [{"Stat": s, "Show": True} for s in filtered]

    def add_stat_callback():
        choice = st.session_state.get(ADD_SELECT_KEY)
        if not choice or choice == sentinel_add:
            return
        fallback = get_preset_base(st.session_state.get(STAT_PRESET_KEY, DEFAULT_PRESET))
        config = normalize_stat_rows(st.session_state.get(STAT_STATE_KEY, fallback), fallback)
        if not any(r["Stat"] == choice for r in config):
            config.append({"Stat": choice, "Show": True})
        st.session_state[STAT_STATE_KEY] = config
        bump_version()
        st.session_state[ADD_RESET_KEY] = True

    def remove_stat_callback():
        choice = st.session_state.get(REMOVE_SELECT_KEY)
        if not choice or choice == sentinel_remove:
            return
        fallback = get_preset_base(st.session_state.get(STAT_PRESET_KEY, DEFAULT_PRESET))
        config = normalize_stat_rows(st.session_state.get(STAT_STATE_KEY, fallback), fallback)
        new_config = [r for r in config if r.get("Stat") != choice]
        st.session_state[STAT_STATE_KEY] = new_config or [r.copy() for r in fallback]
        bump_version()
        st.session_state[REMOVE_RESET_KEY] = True

    def preset_callback():
        preset_name = st.session_state.get(STAT_PRESET_KEY, DEFAULT_PRESET)
        st.session_state[STAT_STATE_KEY] = get_preset_base(preset_name)
        bump_version()
        st.session_state[ADD_RESET_KEY] = True
        st.session_state[REMOVE_RESET_KEY] = True

    def move_stat_row(delta, index, fallback):
        rows = normalize_stat_rows(st.session_state.get(STAT_STATE_KEY, fallback), fallback)
        target = index + delta
        if 0 <= target < len(rows):
            rows[index], rows[target] = rows[target], rows[index]
            st.session_state[STAT_STATE_KEY] = rows
            bump_version()

    # ── Initialize stat state ────────────────────
    if STAT_STATE_KEY not in st.session_state:
        st.session_state[STAT_PRESET_KEY] = DEFAULT_PRESET
        st.session_state[STAT_STATE_KEY] = get_preset_base(DEFAULT_PRESET)
        st.session_state[STAT_VERSION_KEY] = 0

    preset_base_config = get_preset_base(st.session_state.get(STAT_PRESET_KEY, DEFAULT_PRESET))
    current_stat_config = normalize_stat_rows(
        st.session_state.get(STAT_STATE_KEY, preset_base_config), preset_base_config
    )

    preset_options = list(STAT_PRESETS.keys())
    prior_preset = st.session_state.get(STAT_PRESET_KEY, DEFAULT_PRESET)
    preset_index = preset_options.index(prior_preset) if prior_preset in preset_options else 0
    st.selectbox(
        "Stat Preset", preset_options, index=preset_index,
        key=STAT_PRESET_KEY, on_change=preset_callback,
    )

    st.markdown("### Customize stats")
    st.markdown(
        "<div style='margin-bottom: -0.25rem; font-size: 0.9rem;'>"
        "Use the dropdowns to add or remove stats and the arrows to reorder them."
        "</div>",
        unsafe_allow_html=True,
    )

    stats_in_config = [r.get("Stat") for r in current_stat_config if r.get("Stat")]
    available_pool = allowed_add_stats if allowed_add_stats else stat_options
    available_stats = [s for s in available_pool if s not in stats_in_config]

    sentinel_add = "Add stat"
    sentinel_remove = "Remove stat"
    add_options = [sentinel_add] + available_stats
    remove_options = [sentinel_remove] + stats_in_config

    if st.session_state.get(ADD_SELECT_KEY) not in add_options:
        st.session_state[ADD_SELECT_KEY] = sentinel_add
    if st.session_state.pop(ADD_RESET_KEY, False):
        st.session_state[ADD_SELECT_KEY] = sentinel_add
    if st.session_state.get(REMOVE_SELECT_KEY) not in remove_options:
        st.session_state[REMOVE_SELECT_KEY] = sentinel_remove
    if st.session_state.pop(REMOVE_RESET_KEY, False):
        st.session_state[REMOVE_SELECT_KEY] = sentinel_remove

    add_col, remove_col = st.columns(2)
    with add_col:
        st.selectbox(
            "Add stat", add_options, label_visibility="hidden",
            format_func=display_stat_name, key=ADD_SELECT_KEY, on_change=add_stat_callback,
        )
    with remove_col:
        st.selectbox(
            "Remove stat", remove_options, label_visibility="hidden",
            format_func=display_stat_name, key=REMOVE_SELECT_KEY, on_change=remove_stat_callback,
        )

    current_stat_config = normalize_stat_rows(
        st.session_state.get(STAT_STATE_KEY, preset_base_config), preset_base_config
    )

    config_df = pd.DataFrame([
        {"Stat": STAT_DISPLAY_NAMES.get(row["Stat"], row["Stat"]), "Show": bool(row.get("Show", True))}
        for row in current_stat_config
    ])

    edited = st.data_editor(
        config_df,
        column_config={
            "Stat": st.column_config.TextColumn("Stat", disabled=True),
            "Show": st.column_config.CheckboxColumn("Show"),
        },
        hide_index=True,
        width="stretch",
        num_rows="fixed",
        key="pr_stat_editor",
    )

    if edited is not None:
        new_config = []
        for i, erow in edited.iterrows():
            if i < len(current_stat_config):
                new_config.append({"Stat": current_stat_config[i]["Stat"], "Show": bool(erow["Show"])})
        if new_config:
            st.session_state[STAT_STATE_KEY] = new_config
            bump_version()
            current_stat_config = normalize_stat_rows(st.session_state[STAT_STATE_KEY], preset_base_config)

    move_col, up_col, down_col = st.columns([3, 1, 1])
    with move_col:
        move_stat = st.selectbox(
            "Reorder",
            [""] + [STAT_DISPLAY_NAMES.get(r["Stat"], r["Stat"]) for r in current_stat_config],
            label_visibility="collapsed", key="pr_stat_reorder_select",
        )
    idx = next((i for i, r in enumerate(current_stat_config)
                if STAT_DISPLAY_NAMES.get(r["Stat"], r["Stat"]) == move_stat), None)
    with up_col:
        st.button("▲", key="pr_move_up", disabled=not move_stat or idx == 0,
                  on_click=move_stat_row, args=(-1, idx if idx is not None else 0, preset_base_config))
    with down_col:
        st.button("▼", key="pr_move_down", disabled=not move_stat or idx == len(current_stat_config) - 1,
                  on_click=move_stat_row, args=(1, idx if idx is not None else 0, preset_base_config))

    st.session_state[STAT_STATE_KEY] = normalize_stat_rows(
        st.session_state.get(STAT_STATE_KEY, current_stat_config), preset_base_config
    )

stats_order = [r["Stat"] for r in st.session_state[STAT_STATE_KEY] if r.get("Show", True)]
if not stats_order:
    with right_col:
        st.info("Add at least one stat and mark it as shown.")
    st.stop()

# ── Compute ranks ─────────────────────────────
display_name = str(player_row.get("Name", name_input)).strip()
team_raw = str(player_row.get("Team", "N/A"))
team_display = get_team_display(team_raw)
headshot_url = get_headshot(player_row)
esc = html.escape

table_rows = []
for stat in stats_order:
    pool = rate_pool if stat in RATE_STATS else scope_pool
    if stat not in pool.columns:
        continue
    vals = pd.to_numeric(pool[stat], errors="coerce")
    ids = pool["PlayerId"]
    valid_mask = vals.notna()
    vals = vals[valid_mask]
    ids = ids[valid_mask]
    if player_id not in ids.values:
        continue

    pool_size = len(vals)
    ascending = stat in lower_better  # lower value = better rank when True
    ranks = vals.rank(method="min", ascending=ascending)
    player_positions = ids[ids == player_id].index
    rank_val = int(ranks.loc[player_positions[0]])

    player_stat_val = pd.to_numeric(player_row.get(stat, np.nan), errors="coerce")
    value_display = format_stat(stat, player_stat_val) if pd.notna(player_stat_val) else "—"

    percentile = round((pool_size - rank_val) / max(pool_size - 1, 1) * 100)
    if percentile >= 67:
        rank_class = "rank-good"
    elif percentile <= 33:
        rank_class = "rank-bad"
    else:
        rank_class = ""

    rank_display = ordinal(rank_val)

    table_rows.append({
        "stat_label": label_map.get(stat, stat),
        "value_display": value_display,
        "rank_display": rank_display,
        "rank_class": rank_class,
    })

if not table_rows:
    with right_col:
        st.info("None of the selected stats had enough valid data to rank.")
    st.stop()

headshot_html = (
    f'<img src="{esc(headshot_url)}" class="headshot-img" alt="{esc(display_name)}" />'
    if headshot_url else ""
)

rows_html = []
for tr in table_rows:
    rows_html.append(f"""
    <div class="stat-row">
      <div class="stat-label">{esc(tr["stat_label"])}</div>
      <div class="val-mid">{esc(tr["value_display"])}</div>
      <div class="val-rank {esc(tr["rank_class"])}">{esc(tr["rank_display"])}</div>
    </div>""")

rows_html_str = "\n".join(rows_html)
scope_label = "MLB" if scope == "MLB" else scope
thresh_label = f"Min {min_thresh} {thresh_col} for rate stats"

card_html = f"""
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link href="https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600;700;800;900&display=swap" rel="stylesheet">
<style>
  html, body {{
    margin: 0; padding: 0 0 24px 0;
    background: transparent;
    font-family: Source Sans Pro, sans-serif;
  }}
  .card {{
    background: #ffffff;
    border: 1px solid #d0d0d0;
    border-radius: 12px;
    padding: 1.5rem 1rem 1.5rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    box-sizing: border-box;
    width: 100%;
    max-width: 675px;
    margin: 0 auto;
  }}
  .player-header {{
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 14px;
    margin-bottom: 1rem;
  }}
  .headshot-img {{
    width: 150px;
    height: 150px;
    object-fit: cover;
    border-radius: 100px;
    border: 2px solid #dedede;
    background: #f6f6f6;
  }}
  .player-name {{
    font-weight: 700;
    font-size: 1.8rem;
    line-height: 1.1;
    margin: 0;
    text-align: center;
  }}
  .player-meta {{
    color: #666;
    font-size: 1.1rem;
    margin: 0.2rem 0 0 0rem;
    text-align: center;
  }}
  .player-meta-sub {{
    color: #999;
    font-size: 0.9rem;
    margin: 0.4rem 0 0 0rem;
    text-align: center;
  }}
  .player-info {{
    display: block;
    margin-left: 1rem;
  }}
  .col-header {{
    display: grid;
    grid-template-columns: 1.5fr 1fr 0.8fr;
    border-top: 2px solid #c9cdd4;
    border-bottom: 2px solid #c9cdd4;
    padding: 8px 8px;
    font-size: 1rem;
    font-weight: 800;
    color: #590505;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    text-align: center;
  }}
  .stat-row {{
    display: grid;
    grid-template-columns: 1.5fr 1fr 0.8fr;
    border-bottom: 1px solid #efefef;
    align-items: center;
    padding: 5.5px 6px;
    gap: 4px;
  }}
  .stat-row:last-child {{ border-bottom: none; }}
  .stat-label {{
    font-weight: 900;
    font-size: 1rem;
    color: #111;
    text-align: center;
  }}
  .val-mid {{
    font-size: 1.1rem;
    font-weight: 500;
    color: #333;
    text-align: center;
    white-space: nowrap;
  }}
  .val-rank {{
    font-size: 1.1rem;
    font-weight: 800;
    color: #555;
    text-align: center;
    white-space: nowrap;
  }}
  .rank-good {{ color: #1a7a3c; }}
  .rank-bad  {{ color: #c0392b; }}
  .footer {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin: 1.3rem -1rem 0 -1rem;
    padding: 0 1rem;
  }}
  .footer p {{
    margin: 0;
    font-size: 0.75rem;
    color: #666;
    white-space: nowrap;
  }}

  @media (max-width: 600px) {{
    .card {{ padding: 1rem; max-width: 550px; }}
    .player-header {{ gap: 10px; margin-bottom: 0.75rem; }}
    .headshot-img {{ width: 75px; height: 75px; border-width: 1px; }}
    .player-info {{ margin-left: 0.25rem; }}
    .player-name {{ font-size: 1.3rem; }}
    .player-meta {{ font-size: 0.85rem; }}
    .player-meta-sub {{ font-size: 0.7rem; }}
    .col-header {{ padding: 5px 6px; font-size: 0.8rem; }}
    .stat-row {{ padding: 4px 4px; gap: 2px; }}
    .stat-label {{ font-size: 0.8rem; }}
    .val-mid {{ font-size: 0.85rem; }}
    .val-rank {{ font-size: 0.85rem; }}
    .footer p {{ font-size: 0.6rem; }}
    .footer {{ padding: 0 0.5rem; margin-top: 0.8rem; }}
  }}
</style>
</head>
<body>
<div class="card">

  <div class="player-header">
    {headshot_html}
    <div class="player-info">
    <div class="player-name">{esc(display_name)}</div>
    <div class="player-meta">{esc(team_display)} &nbsp;|&nbsp; {int(year)}</div>
    <div class="player-meta-sub">{esc(thresh_label)}</div>
    </div>
  </div>

 <div class="col-header">
  <span>Stat</span><span>Value</span><span>{esc(scope_label)} Rank</span>
</div>

  {rows_html_str}

  <div class="footer">
    <p>By: Sox_Savant</p>
    <p>Data: FanGraphs • Baseball Reference • Baseball Savant</p>
  </div>

</div>
</body>
</html>
"""

with right_col:
    row_count = len(rows_html)
    card_height = 275 + row_count * 40
    st.iframe(card_html, height=card_height)
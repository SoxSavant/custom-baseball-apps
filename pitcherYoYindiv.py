import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import html
import unicodedata
from datetime import date

st.set_page_config(page_title="Individual Pitcher Year-over-Year", layout="wide", page_icon="⚾")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 1rem !important;
    }
        [data-testid="stToolbar"] {visibility: hidden;}
        [data-testid="stDecoration"] {display: none;}
        [data-testid="stStatusWidget"] {display: none;}
        .viewerBadge_link__qRi_k {display: none;}
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
    st.title("Individual Pitcher Year-over-Year")
with meta_col:
    st.markdown(
        """
        <div class="mobile-meta" style="text-align: right; font-size: 1rem; padding-top: 0.6rem;">
            Built by <a href="https://twitter.com/Sox_Savant" target="_blank">@Sox_Savant</a>
        </div>
        """,
        unsafe_allow_html=True,
    )
from p_utils import get_last_updated
current_year = date.today().year
last_updated = get_last_updated(current_year)
st.caption(f"{current_year} data last updated: {last_updated}")
from p_utils import (
    STAT_ALLOWLIST, STAT_PRESETS_YOY, STAT_DISPLAY_NAMES, TRUTHY_STRINGS,
    start_year as DATA_START_YEAR, get_headshot, label_map, lower_better,
    format_stat, format_stat_yoy, load_final_year, get_team_display,
)

current_year = date.today().year

def normalize_name(raw: str) -> str:
    if not raw or not isinstance(raw, str):
        return ""
    cleaned = raw.replace("\xa0", " ").strip()
    try:
        cleaned = unicodedata.normalize("NFKD", cleaned).encode("ascii", "ignore").decode()
    except Exception:
        pass
    return " ".join(cleaned.split()).lower()


def display_stat_name(stat) -> str:
    if stat is None:
        return ""
    return STAT_DISPLAY_NAMES.get(str(stat), str(stat))


def get_player_id_by_name(name: str, year: int) -> int | None:
    df = load_final_year(year)
    if df is None or df.empty or "Name" not in df.columns:
        return None
    match = df[df["Name"].str.strip() == name.strip()]
    if match.empty:
        match = df[df["Name"].str.lower().str.strip() == name.lower().strip()]
    if match.empty:
        return None
    ids = match["PlayerId"].dropna()
    return int(ids.iloc[0]) if not ids.empty else None


def resolve_player_id(name: str, y1: int, y2: int) -> int | None:
    for year in range(y2, y1 - 1, -1):
        pid = get_player_id_by_name(name, year)
        if pid is not None:
            return pid
    return None


@st.cache_data(show_spinner=False, ttl=3600)
def load_player_year(player_id: int, year: int) -> pd.Series | None:
    df = load_final_year(year)
    if df is None or df.empty:
        return None
    match = df[df["PlayerId"] == player_id]
    if match.empty:
        return None
    return match.iloc[0]


left_col, right_col = st.columns([1, 2])

years_desc = list(range(current_year, DATA_START_YEAR - 1, -1))


for key, default in [
    ("iyoy_player",         "Mason Miller"),
    ("iyoy_player_id",      ""),
    ("iyoy_player_mode",    "Name"),
    ("iyoy_start_year",     current_year - 1),
    ("iyoy_end_year",       current_year),
]:
    if key not in st.session_state:
        st.session_state[key] = default


STAT_STATE_KEY       = "iyoy_stat_config"
STAT_PRESET_KEY      = "iyoy_stat_preset"
STAT_VERSION_KEY     = "iyoy_stat_version"
MANUAL_UPDATE_KEY    = "iyoy_stat_manual_update"
ADD_SELECT_KEY       = "iyoy_add_stat_select"
REMOVE_SELECT_KEY    = "iyoy_remove_stat_select"
ADD_RESET_KEY        = "iyoy_reset_add_select"
REMOVE_RESET_KEY     = "iyoy_reset_remove_select"

DEFAULT_PRESET = "Default"


with left_col:
    # Player input
    mode_val = st.selectbox("Player Input", ["Name", "FanGraphs ID"], key="iyoy_player_mode")
    if mode_val == "Name":
        name_input = st.text_input("Player", key="iyoy_player")
        id_input = st.session_state.get("iyoy_player_id", "")
    else:
        id_input = st.text_input("FanGraphs ID", key="iyoy_player_id")
        name_input = st.session_state.get("iyoy_player", "")

    # Year selectors
    col_y1, col_y2 = st.columns(2)
    with col_y1:
        st.selectbox("Start Year", options=list(range(current_year - 1, DATA_START_YEAR - 1, -1)), key="iyoy_start_year")
    with col_y2:
        st.selectbox("End Year", options=list(range(current_year, DATA_START_YEAR - 1, -1)), key="iyoy_end_year")

    start_yr = st.session_state["iyoy_start_year"]
    end_yr   = st.session_state["iyoy_end_year"]

    if end_yr <= start_yr:
        st.warning("End Year must be greater than Start Year.")
        st.stop()

    # ── Resolve player ──────────────────────────
    if mode_val == "Name":
        if not name_input.strip():
            st.warning("Enter a player name.")
            st.stop()
        player_id = resolve_player_id(name_input.strip(), start_yr, end_yr)
        if not player_id:
            st.error(f"Could not find '{name_input}'. Check spelling or use FanGraphs ID.")
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

    row_start = load_player_year(player_id, start_yr)
    row_end   = load_player_year(player_id, end_yr)

    if row_start is None:
        st.error(f"No data found for this player in {start_yr}.")
        st.stop()
    if row_end is None:
        st.error(f"No data found for this player in {end_yr}.")
        st.stop()

    # ── Build stat options ───────────────────────
    stat_exclusions = {"Season", "PlayerId", "MLBAMID"}
    numeric_start = {c for c in row_start.index if pd.notna(pd.to_numeric(row_start[c], errors="coerce"))}
    numeric_end   = {c for c in row_end.index   if pd.notna(pd.to_numeric(row_end[c],   errors="coerce"))}
    numeric_stats_set = (numeric_start & numeric_end) - stat_exclusions

    preferred = [s for s in STAT_ALLOWLIST if s in numeric_stats_set]
    other     = [s for s in numeric_stats_set if s not in preferred]
    stat_options = preferred + other
    allowed_add_stats = preferred if preferred else stat_options.copy()

    if not stat_options:
        st.error("No comparable numeric stats between these two seasons.")
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
        preset_stats = STAT_PRESETS_YOY.get(preset_name, [])
        filtered = [s for s in preset_stats if s in stat_options]
        if not filtered and stat_options:
            filtered = [stat_options[0]]
        return [{"Stat": s, "Show": True} for s in filtered]

    def compute_direction_preset(want_improvement: bool) -> list[dict]:
        """Return stat config for all stats that improved (or regressed)."""
        matched = []
        for stat in STAT_ALLOWLIST:
            if stat not in stat_options:
                continue
            s_val = pd.to_numeric(row_start.get(stat, np.nan), errors="coerce")
            e_val = pd.to_numeric(row_end.get(stat,   np.nan), errors="coerce")
            if pd.isna(s_val) or pd.isna(e_val):
                continue
            delta = float(e_val - s_val)
            if delta == 0:
                continue
            is_improvement = (delta < 0) if (stat in lower_better) else (delta > 0)
            if is_improvement == want_improvement:
                matched.append(stat)
        return [{"Stat": s, "Show": True} for s in matched] if matched else get_preset_base(DEFAULT_PRESET)

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
        st.session_state[MANUAL_UPDATE_KEY] = True
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
        st.session_state[MANUAL_UPDATE_KEY] = True
        st.session_state[REMOVE_RESET_KEY] = True

    def preset_callback():
        preset_name = st.session_state.get(STAT_PRESET_KEY, DEFAULT_PRESET)

        if preset_name == "Only Improvements":
            new_config = compute_direction_preset(want_improvement=True)
        elif preset_name == "Only Regressions":
            new_config = compute_direction_preset(want_improvement=False)
        else:
            new_config = get_preset_base(preset_name)

        st.session_state[STAT_STATE_KEY] = new_config
        bump_version()
        st.session_state[MANUAL_UPDATE_KEY] = True
        st.session_state[ADD_RESET_KEY] = True
        st.session_state[REMOVE_RESET_KEY] = True

    def move_stat_row(delta, index, fallback):
        rows = normalize_stat_rows(st.session_state.get(STAT_STATE_KEY, fallback), fallback)
        target = index + delta
        if 0 <= target < len(rows):
            rows[index], rows[target] = rows[target], rows[index]
            st.session_state[STAT_STATE_KEY] = rows
            bump_version()
            st.session_state[MANUAL_UPDATE_KEY] = True

    def toggle_stat_show(index, check_key, fallback):
        rows = normalize_stat_rows(st.session_state.get(STAT_STATE_KEY, fallback), fallback)
        if 0 <= index < len(rows):
            rows[index]["Show"] = bool(st.session_state.get(check_key, True))
            st.session_state[STAT_STATE_KEY] = rows
            bump_version()
            st.session_state[MANUAL_UPDATE_KEY] = True

    # ── Initialize stat state ────────────────────
    if STAT_STATE_KEY not in st.session_state:
        st.session_state[STAT_PRESET_KEY] = DEFAULT_PRESET
        st.session_state[STAT_STATE_KEY]  = get_preset_base(DEFAULT_PRESET)
        st.session_state[STAT_VERSION_KEY] = 0

    preset_base_config = get_preset_base(st.session_state.get(STAT_PRESET_KEY, DEFAULT_PRESET))
    current_stat_config = normalize_stat_rows(
        st.session_state.get(STAT_STATE_KEY, preset_base_config), preset_base_config
    )

    # ── Stat preset selector ─────────────────────
    preset_options = list(STAT_PRESETS_YOY.keys())
    prior_preset = st.session_state.get(STAT_PRESET_KEY, DEFAULT_PRESET)
    preset_index = preset_options.index(prior_preset) if prior_preset in preset_options else 0
    st.selectbox(
        "Stat Preset",
        preset_options,
        index=preset_index,
        key=STAT_PRESET_KEY,
        on_change=preset_callback,
    )

    # ── Add / remove ─────────────────────────────
    st.markdown("### Customize stats")
    st.markdown(
        "<div style='margin-bottom: -0.25rem; font-size: 0.9rem;'>"
        "Use the dropdowns to add or remove stats and the arrows to reorder them."
        "</div>",
        unsafe_allow_html=True,
    )

    stats_in_config = [r.get("Stat") for r in current_stat_config if r.get("Stat")]
    available_pool  = allowed_add_stats if allowed_add_stats else stat_options
    available_stats = [s for s in available_pool if s not in stats_in_config]

    sentinel_add    = "Add stat"
    sentinel_remove = "Remove stat"
    add_options     = [sentinel_add]    + available_stats
    remove_options  = [sentinel_remove] + stats_in_config

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
            format_func=display_stat_name,
            key=ADD_SELECT_KEY, on_change=add_stat_callback,
        )
    with remove_col:
        st.selectbox(
            "Remove stat", remove_options, label_visibility="hidden",
            format_func=display_stat_name,
            key=REMOVE_SELECT_KEY, on_change=remove_stat_callback,
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
        width='stretch',
        num_rows="fixed",
        key="stat_editor",
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
            label_visibility="collapsed",
            key="stat_reorder_select",
        )
    idx = next((i for i, r in enumerate(current_stat_config) if STAT_DISPLAY_NAMES.get(r["Stat"], r["Stat"]) == move_stat), None)
    with up_col:
        st.button("▲", key="move_up", disabled=not move_stat or idx == 0,
                  on_click=move_stat_row, args=(-1, idx if idx is not None else 0, preset_base_config))
    with down_col:
        st.button("▼", key="move_down", disabled=not move_stat or idx == len(current_stat_config) - 1,
                  on_click=move_stat_row, args=(1, idx if idx is not None else 0, preset_base_config))

    st.session_state[STAT_STATE_KEY] = normalize_stat_rows(
        st.session_state.get(STAT_STATE_KEY, current_stat_config), preset_base_config
    )

stats_order = [r["Stat"] for r in st.session_state[STAT_STATE_KEY] if r.get("Show", True)]
if not stats_order:
    with right_col:
        st.info("Add at least one stat and mark it as shown.")
    st.stop()

# Player meta
display_name = str(row_end.get("Name", name_input)).strip()
team_raw     = str(row_end.get("Team", "N/A"))
team_display = get_team_display(team_raw)
headshot_url = get_headshot(row_end)
esc = html.escape

# Build table rows
table_rows = []
for stat in stats_order:
    s_val = pd.to_numeric(row_start.get(stat, np.nan), errors="coerce")
    e_val = pd.to_numeric(row_end.get(stat,   np.nan), errors="coerce")
    if pd.isna(s_val) and pd.isna(e_val):
        continue

    start_display = format_stat(stat, s_val) if pd.notna(s_val) else "—"
    end_display   = format_stat(stat, e_val) if pd.notna(e_val) else "—"

    # Compute delta from display-rounded values so "10.1 → 10.1" never shows a non-zero delta.
    # For % stats format_stat scales raw decimals (0.237) to 23.7%, so we must
    # subtract the scaled display values and pass the scaled delta to format_stat_yoy.
    def _parse_display(s):
        try:
            return float(s.replace("%", "").replace(",", "").strip())
        except Exception:
            return np.nan

    def _is_pct_stat(stat):
        return (
            "Barrel" in stat or "Hard" in stat or "K%" in stat
            or "Swing" in stat or "Whiff" in stat or "%" in stat
        )

    if pd.notna(s_val) and pd.notna(e_val):
        s_rounded = _parse_display(start_display)
        e_rounded = _parse_display(end_display)
        if not np.isnan(s_rounded) and not np.isnan(e_rounded):
            raw_delta = e_rounded - s_rounded
            # If it's a % stat, display values are already in pct-point scale (e.g. 23.7).
            # Convert back to raw scale so format_stat_yoy formats correctly,
            # unless format_stat_yoy also expects the scaled value — keep scaled.
            if _is_pct_stat(stat):
                delta = raw_delta  # already in display pct-point units; pass directly
                delta_is_scaled = True
            else:
                delta = raw_delta
                delta_is_scaled = False
        else:
            delta = np.nan
            delta_is_scaled = False
    else:
        delta = np.nan
        delta_is_scaled = False

    if pd.isna(delta):
        delta_display = "—"
        delta_class   = ""
    else:
        is_positive = float(delta) > 0
        # For % stats the delta is already in display pct-point units (e.g. +3.2),
        # so format it directly instead of through format_stat_yoy which would
        # re-scale a raw decimal.
        if delta_is_scaled:
            sign = "+" if is_positive else ""
            delta_display = f"{sign}{delta:.1f}%"
        else:
            delta_display = format_stat_yoy(stat, delta, show_sign=is_positive)
        if stat in lower_better:
            improvement = float(delta) < 0
        else:
            improvement = float(delta) > 0
        delta_class = "delta-good" if improvement else ("delta-bad" if float(delta) != 0 else "")

    table_rows.append({
        "stat_label":     label_map.get(stat, stat),
        "start_display":  start_display,
        "end_display":    end_display,
        "delta_display":  delta_display,
        "delta_class":    delta_class,
    })

# ─────────────────────────────────────────────
#  Render HTML card
# ─────────────────────────────────────────────

headshot_html = f'<img src="{esc(headshot_url)}" class="headshot-img" alt="{esc(display_name)}" />' if headshot_url else ""

def stat_block(tr, side="left"):
    border_style = "border-right: 2px solid #e8e8e8;" if side == "left" else ""
    delta_cls = tr["delta_class"]
    return f"""
      <div class="stat-block" style="{border_style}">
        <div class="stat-label">{esc(tr["stat_label"])}</div>
        <div class="val-start">{esc(tr["start_display"])}</div>
        <div class="val-delta {esc(delta_cls)}">{esc(tr["delta_display"])}</div>
        <div class="val-end">{esc(tr["end_display"])}</div>
      </div>"""

rows_html = []
for tr in table_rows:
    delta_cls = tr["delta_class"]
    rows_html.append(f"""
    <div class="stat-row">
      <div class="stat-label">{esc(tr["stat_label"])}</div>
      <div class="val-start">{esc(tr["start_display"])}</div>
      <div class="val-delta {esc(delta_cls)}">{esc(tr["delta_display"])}</div>
      <div class="val-end">{esc(tr["end_display"])}</div>
    </div>""")

rows_html_str = "\n".join(rows_html)

card_html = f"""
<html>
<head>
<meta charset="utf-8"/>
<link href="https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600;700;800;900&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700;900&display=swap" rel="stylesheet">
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
    padding: 1.5rem 1.5rem 1.5rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    box-sizing: border-box;
    width: 100%;
    max-width: 700px;
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
  .player-info {{
    display: block;
    margin-left: 1rem;
  }}
 .col-header {{
    display: grid;
    grid-template-columns: 1fr 1fr 1fr 1fr;
    border-top: 2px solid #c9cdd4;
    border-bottom: 2px solid #c9cdd4;
    padding: 8px 14px;
    font-size: 1.1rem;
    font-weight: 800;
    color: #590505;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    text-align: center;
}}
.stat-row {{
    display: grid;
    grid-template-columns: 1fr 1fr 1fr 1fr;
    border-bottom: 1px solid #efefef;
    align-items: center;
    padding: 5.5px 10px;
    gap: 6px;
    
}}
  .stat-row:last-child {{ 
    border-bottom: none; 
    }}
  .stat-block {{
    display: grid;
    grid-template-columns: 1fr 1fr 1fr 1fr;
    align-items: center;
    padding: 4px 10px;
    gap: 2px;
  }}
  .stat-label {{
    font-weight: 900;
    font-size: 1rem;
    color: #111;
    text-align: center;
  }}
  .val-start, .val-end {{
    font-size: 1.1rem;
    font-weight: 500;
    color: #333;
    text-align: center;
    white-space: nowrap;
  }}
  .val-delta {{
    font-size: 1.2rem;
    font-weight: 800;
    white-space: nowrap;
    color: #555;
    text-align: center;
  }}
  .delta-good {{ 
    color: #1a7a3c; 
    }}
  .delta-bad  {{ 
    color: #c0392b; 
    }}
.footer {{
display: flex;
justify-content: space-between;
align-items: center;
margin: 1.3rem -1rem 0 -1rem;
padding: 0 3rem;
}}

.footer p {{
margin: 0;
font-size: 1rem;
color: #666;
white-space: nowrap;
}}
  .signs {{
    color: #635c5b;
  }}
  @media (max-width: 600px) {{
    .card {{
        padding: 1rem; /* Tightens inner frame spacing */
    }}
    .player-header {{
        gap: 10px;
        margin-bottom: 0.75rem;
    }}
    .headshot-img {{
        width: 75px; /* Cut photo dimension in half */
        height: 75px;
        border-width: 1px;
    }}
    .player-info {{
        margin-left: 0.25rem;
    }}
    .player-name {{
        font-size: 1.3rem; /* Brings down name scaling */
    }}
    .player-meta {{
        font-size: 0.85rem;
    }}
    .col-header {{
        padding: 5px 6px;
        font-size: 0.8rem; /* Keeps label line clean without stacking */
    }}
    .stat-row {{
        padding: 4px 4px;
        gap: 2px;
    }}
    .stat-label {{
        font-size: 0.8rem;
    }}
    .val-start, .val-end {{
        font-size: 0.85rem;
    }}
    .val-delta {{
        font-size: 0.9rem;
    }}
   .footer p {{
font-size: 0.65rem;
}}

.footer {{
padding: 0 2rem;
margin-top: 0.8rem;
}}
  }}
</style>
</head>
<body>
<div class="card">

  <div class="player-header">
    {headshot_html}
    <div class = "player-info">
    <div class="player-name">{esc(display_name)}</div>
    <div class="player-meta">{esc(team_display)} &nbsp;|&nbsp; {int(start_yr)} → {int(end_yr)}</div>
    </div>
  </div>

 <div class="col-header">
  <span>Stat</span><span>{int(start_yr)}</span><span class="signs">+/-</span><span>{int(end_yr)}</span>
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
    # ── Original card ────────────────────────────────────────────────────
    row_count = len(rows_html)
    card_height = 295 + row_count * 40
    components.html(card_html, height=card_height)
    st.caption("Screenshot to save")
    st.caption("Find a player's FanGraphs ID in their FanGraphs profile URL")

    # ── Leaderboard ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Find Most Improved / Regressed")

    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns([2, 1, 1, 1])
    with col_btn1:
        run_leaderboard = st.button(
            f"Find Players",
            help="Scans every player present in both years using the current stat preset"
        )
    with col_btn2:
        top_n = st.number_input("Show top N", min_value=5, max_value=50, value=15, step=5)
    with col_btn3:
        min_pa_start = st.number_input(f"Min IP ({int(start_yr)})", min_value=0, max_value=700, value=100, step=25)
    with col_btn4:
        min_pa_end = st.number_input(f"Min IP ({int(end_yr)})", min_value=0, max_value=700, value=40, step=25)

    if run_leaderboard:
        scan_stats = stats_order

        with st.spinner(f"Scanning all players for {int(start_yr)}→{int(end_yr)}..."):
            df_start = load_final_year(start_yr)
            df_end   = load_final_year(end_yr)

        if df_start is None or df_end is None:
            st.error("Could not load data for one or both years.")
        else:
            # Apply PA filters before merging
            if "IP" in df_start.columns:
                df_start = df_start[pd.to_numeric(df_start["IP"], errors="coerce") >= min_pa_start]
            if "IP" in df_end.columns:
                df_end = df_end[pd.to_numeric(df_end["IP"], errors="coerce") >= min_pa_end]

            merged = df_start.merge(df_end, on="PlayerId", suffixes=("_s", "_e"))

            records = []
            for _, row in merged.iterrows():
                improved = 0
                regressed = 0
                total = 0
                for stat in scan_stats:
                    s_col = f"{stat}_s"
                    e_col = f"{stat}_e"
                    if s_col not in row or e_col not in row:
                        continue
                    s_val = pd.to_numeric(row[s_col], errors="coerce")
                    e_val = pd.to_numeric(row[e_col], errors="coerce")
                    if pd.isna(s_val) or pd.isna(e_val):
                        continue
                    delta = float(e_val - s_val)
                    if delta == 0:
                        continue
                    total += 1
                    is_improvement = (delta < 0) if (stat in lower_better) else (delta > 0)
                    if is_improvement:
                        improved += 1
                    else:
                        regressed += 1

                if total == 0:
                    continue

                name = str(row.get("Name_e", row.get("Name_s", "Unknown"))).strip()
                team = str(row.get("Team_e", "N/A"))
                records.append({
                    "Name":      name,
                    "Team":      get_team_display(team),
                    "Improved":  improved,
                    "Regressed": regressed,
                    "Total":     total,
                    "Pct":       round(improved / total * 100, 1),
                    "PlayerId":  int(row["PlayerId"]),
                })

            if not records:
                st.info("No players matched the IP filters.")
            else:
                df_scores = pd.DataFrame(records)

                most_improved  = df_scores.sort_values(["Improved",  "Pct"], ascending=[False, False]).head(int(top_n)).reset_index(drop=True)
                most_regressed = df_scores.sort_values(["Regressed", "Pct"], ascending=[False, True]).head(int(top_n)).reset_index(drop=True)

                tab1, tab2 = st.tabs(["🟢 Most Improved", "🔴 Most Regressed"])

                with tab1:
                    display_imp = most_improved[["Name", "Team", "Improved", "Total", "Pct"]].copy()
                    display_imp.columns = ["Name", "Team", "# Improved", "# Tracked", "% Improved"]
                    display_imp.index += 1
                    st.dataframe(
                        display_imp,
                        width='stretch',
                        column_config={
                            "% Improved": st.column_config.ProgressColumn(
                                "% Improved", min_value=0, max_value=100, format="%.1f%%"
                            )
                        }
                    )

                with tab2:
                    most_regressed["% Regressed"] = round(most_regressed["Regressed"] / most_regressed["Total"] * 100, 1)
                    display_reg = most_regressed[["Name", "Team", "Regressed", "Total", "% Regressed"]].copy()
                    display_reg.columns = ["Name", "Team", "# Regressed", "# Tracked", "% Regressed"]
                    display_reg.index += 1
                    st.dataframe(
                        display_reg,
                        width='stretch',
                        column_config={
                            "% Regressed": st.column_config.ProgressColumn(
                                "% Regressed", min_value=0, max_value=100, format="%.1f%%"
                            )
                        }
                    )

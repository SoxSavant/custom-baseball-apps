import streamlit as st
import pandas as pd
from datetime import date
from p_utils import (
    load_final_year,
    TEAM_OPTIONS, normalize_team,
    STAT_ALLOWLIST, STAT_ROUND,  PCT_STATS, aggregate_player_group,
    STAT_PRESETS_DATABASE, STAT_DISPLAY_NAMES,
     get_last_updated,
    TRUTHY_STRINGS,
)
from utils import get_dynamic_min_ip

st.set_page_config(page_title="Pitching Database", layout="wide", page_icon="⚾")

st.markdown("""
<style>
[data-testid="stToolbar"]      { visibility: hidden; }
[data-testeid="stDecoration"]   { display: none; }
[data-testid="stStatusWidget"] { display: none; }
.viewerBadge_link__qRi_k       { display: none; }
.block-container { padding-top: 1rem !important; padding-bottom: 3rem !important; }
.stSelectbox div[data-baseweb="select"],
.stNumberInput > div { max-width: 200px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    @media only screen and (max-width: 600px) {
        [data-testid="stAppViewContainer"] h1 { font-size: 1.8rem !important; }
        .mobile-meta { font-size: 0.8rem !important; padding-top: 0.3rem !important; }
    }
</style>
""", unsafe_allow_html=True)

title_col, meta_col = st.columns([3, 1])
with title_col:
    st.title("Pitching Database")
with meta_col:
    st.markdown(
        """
        <div class="mobile-meta" style="text-align: right; font-size: 1rem; padding-top: 0.6rem;">
            Built by <a href="https://twitter.com/Sox_Savant" target="_blank">@Sox_Savant</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

current_year = date.today().year
START_YEAR   = 1901


def load_year_range(start: int, end: int) -> pd.DataFrame:
    frames = []
    for yr in range(start, end + 1):
        df = load_final_year(yr)
        if df is not None and not df.empty:
            df = df.copy()
            df["Year"] = yr
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


STAT_STATE_KEY    = "db_stat_config"
STAT_PRESET_KEY   = "db_stat_preset"
STAT_VERSION_KEY  = "db_stat_version"
STAT_MANUAL_KEY   = "db_stat_manual_update"
ADD_SELECT_KEY    = "db_add_stat_select"
REMOVE_SELECT_KEY = "db_remove_stat_select"
ADD_RESET_KEY     = "db_add_reset"
REMOVE_RESET_KEY  = "db_remove_reset"

stat_options        = list(STAT_ALLOWLIST)
default_preset_name = list(STAT_PRESETS_DATABASE.keys())[0]
preset_options      = list(STAT_PRESETS_DATABASE.keys())


def normalize_stat_rows(rows, fallback):
    cleaned = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = row.get("Stat")
        if not name or name not in stat_options or name in seen:
            continue
        show_val = row.get("Show", True)
        if pd.isna(show_val) if not isinstance(show_val, str) else False:
            show_bool = True
        elif isinstance(show_val, str):
            show_bool = show_val.strip().lower() in TRUTHY_STRINGS
        else:
            show_bool = bool(show_val)
        cleaned.append({"Stat": name, "Show": show_bool})
        seen.add(name)
    return cleaned if cleaned else [r.copy() for r in fallback]


def bump_version():
    st.session_state[STAT_VERSION_KEY] = st.session_state.get(STAT_VERSION_KEY, 0) + 1


def _preset_base(preset_name):
    candidates = [s for s in STAT_PRESETS_DATABASE.get(preset_name, []) if s in stat_options]
    if not candidates and stat_options:
        candidates = [stat_options[0]]
    return [{"Stat": s, "Show": True} for s in candidates]


def add_stat_callback():
    choice = st.session_state.get(ADD_SELECT_KEY)
    if not choice or choice == _sentinel_add:
        return
    current_preset = st.session_state.get(STAT_PRESET_KEY, default_preset_name)
    base = _preset_base(current_preset)
    config = normalize_stat_rows(st.session_state.get(STAT_STATE_KEY, base), base)
    if not any(r["Stat"] == choice for r in config):
        config.append({"Stat": choice, "Show": True})
    st.session_state[STAT_STATE_KEY] = config
    bump_version()
    st.session_state[STAT_MANUAL_KEY] = True
    st.session_state[ADD_RESET_KEY] = True


def remove_stat_callback():
    choice = st.session_state.get(REMOVE_SELECT_KEY)
    if not choice or choice == _sentinel_remove:
        return
    current_preset = st.session_state.get(STAT_PRESET_KEY, default_preset_name)
    base = _preset_base(current_preset)
    config = normalize_stat_rows(st.session_state.get(STAT_STATE_KEY, base), base)
    config = [r for r in config if r.get("Stat") != choice] or [r.copy() for r in base]
    st.session_state[STAT_STATE_KEY] = config
    bump_version()
    st.session_state[STAT_MANUAL_KEY] = True
    st.session_state[REMOVE_RESET_KEY] = True


def preset_callback():
    preset_name = st.session_state.get(STAT_PRESET_KEY, default_preset_name)
    candidates = [s for s in STAT_PRESETS_DATABASE.get(preset_name, []) if s in stat_options]
    if not candidates and stat_options:
        candidates = [stat_options[0]]
    st.session_state[STAT_STATE_KEY] = [{"Stat": s, "Show": True} for s in candidates]
    bump_version()
    st.session_state[STAT_MANUAL_KEY] = True
    st.session_state[ADD_RESET_KEY] = True
    st.session_state[REMOVE_RESET_KEY] = True


def move_stat_row(delta, index, fallback):
    rows = normalize_stat_rows(st.session_state.get(STAT_STATE_KEY, fallback), fallback)
    target = index + delta
    if 0 <= target < len(rows):
        rows[index], rows[target] = rows[target], rows[index]
        st.session_state[STAT_STATE_KEY] = rows
        bump_version()
        st.session_state[STAT_MANUAL_KEY] = True


if STAT_STATE_KEY not in st.session_state:
    st.session_state[STAT_PRESET_KEY] = default_preset_name
    st.session_state[STAT_STATE_KEY]  = _preset_base(default_preset_name)
    st.session_state[STAT_VERSION_KEY] = 0

current_preset     = st.session_state.get(STAT_PRESET_KEY, default_preset_name)
preset_base_config = _preset_base(current_preset)
current_stat_config = normalize_stat_rows(
    st.session_state.get(STAT_STATE_KEY, preset_base_config), preset_base_config
)

_sentinel_add    = "Add stat"
_sentinel_remove = "Remove stat"

last_updated = get_last_updated(current_year)
st.caption(f"{current_year} data last updated: {last_updated}")

left_col, right_col = st.columns([1, 2.8])

with left_col:
    st.markdown("### Filters")

    year_mode = st.selectbox("Year Mode", ["Single Season", "Multi-Year Span", "Split Season"])

    if year_mode == "Single Season":
        year_single = st.selectbox("Year", list(range(current_year, START_YEAR - 1, -1)))
        year_start = year_end = year_single
    else:
        year_start = st.selectbox("From", list(range(current_year, START_YEAR - 1, -1)), index=4)
        valid_ends = list(range(current_year, year_start - 1, -1))
        year_end   = st.selectbox("To", valid_ends, index=0)

    team = st.selectbox(
        "Team",
        options=list(TEAM_OPTIONS.keys()),
        format_func=lambda x: TEAM_OPTIONS[x],
    )
    min_ip = st.number_input(
        "Min IP", min_value=0, max_value=5000,
        value=int(get_dynamic_min_ip(current_year)), step=5
    )
    search = st.text_input("Search players (separate by commas)")

    st.divider()
    st.markdown("### Stats")

    prior_preset = st.session_state.get(STAT_PRESET_KEY, default_preset_name)
    preset_index = preset_options.index(prior_preset) if prior_preset in preset_options else 0
    st.selectbox(
        "Stat Preset", preset_options, index=preset_index,
        key=STAT_PRESET_KEY, on_change=preset_callback,
    )

    stats_in_config  = [r["Stat"] for r in current_stat_config]
    available_to_add = [s for s in stat_options if s not in stats_in_config]

    add_options    = [_sentinel_add] + available_to_add
    remove_options = [_sentinel_remove] + stats_in_config

    if st.session_state.get(ADD_SELECT_KEY) not in add_options:
        st.session_state[ADD_SELECT_KEY] = _sentinel_add
    if st.session_state.pop(ADD_RESET_KEY, False):
        st.session_state[ADD_SELECT_KEY] = _sentinel_add
    if st.session_state.get(REMOVE_SELECT_KEY) not in remove_options:
        st.session_state[REMOVE_SELECT_KEY] = _sentinel_remove
    if st.session_state.pop(REMOVE_RESET_KEY, False):
        st.session_state[REMOVE_SELECT_KEY] = _sentinel_remove

    add_col, remove_col = st.columns(2)
    with add_col:
        st.selectbox(
            "Add stat", add_options, label_visibility="hidden",
            key=ADD_SELECT_KEY, on_change=add_stat_callback,
        )
    with remove_col:
        st.selectbox(
            "Remove stat", remove_options, label_visibility="hidden",
            key=REMOVE_SELECT_KEY, on_change=remove_stat_callback,
        )

    current_stat_config = normalize_stat_rows(
        st.session_state.get(STAT_STATE_KEY, preset_base_config), preset_base_config
    )

    config_df = pd.DataFrame([
        {"Stat": STAT_DISPLAY_NAMES.get(r["Stat"], r["Stat"]), "_key": r["Stat"], "Show": bool(r.get("Show", True))}
        for r in current_stat_config
    ])

    edited = st.data_editor(
        config_df[["Stat", "Show"]],
        column_config={
            "Stat": st.column_config.TextColumn("Stat", disabled=True),
            "Show": st.column_config.CheckboxColumn("Show"),
        },
        hide_index=True,
        width="stretch",
        num_rows="fixed",
        key="db_stat_editor",
    )

    if edited is not None:
        new_config = []
        for i, erow in edited.iterrows():
            if i < len(current_stat_config):
                new_config.append({
                    "Stat": current_stat_config[i]["Stat"],
                    "Show": bool(erow["Show"]),
                })
        if new_config:
            st.session_state[STAT_STATE_KEY] = new_config
            bump_version()
            current_stat_config = normalize_stat_rows(
                st.session_state[STAT_STATE_KEY], preset_base_config
            )

    move_col, up_col, down_col = st.columns([3, 1, 1])
    with move_col:
        move_stat = st.selectbox(
            "Reorder", [""] + [STAT_DISPLAY_NAMES.get(r["Stat"], r["Stat"]) for r in current_stat_config],
            label_visibility="collapsed",
            key="db_stat_reorder_select",
        )
    display_to_key = {STAT_DISPLAY_NAMES.get(r["Stat"], r["Stat"]): r["Stat"] for r in current_stat_config}
    move_key = display_to_key.get(move_stat)
    idx = next((i for i, r in enumerate(current_stat_config) if r["Stat"] == move_key), None)
    with up_col:
        st.button(
            "▲", key="db_move_up",
            disabled=not move_stat or idx == 0,
            on_click=move_stat_row, args=(-1, idx if idx is not None else 0, preset_base_config),
        )
    with down_col:
        st.button(
            "▼", key="db_move_down",
            disabled=not move_stat or idx == len(current_stat_config) - 1,
            on_click=move_stat_row, args=(1, idx if idx is not None else 0, preset_base_config),
        )

    st.session_state[STAT_STATE_KEY] = normalize_stat_rows(
        st.session_state.get(STAT_STATE_KEY, current_stat_config), preset_base_config
    )

selected_stats = [
    r["Stat"] for r in st.session_state.get(STAT_STATE_KEY, current_stat_config)
    if r.get("Show", True)
]

with right_col:
    if not selected_stats:
        st.warning("Select at least one stat.")
        st.stop()

    if year_start == year_end:
        df = load_final_year(year_start)
        if df is not None and not df.empty:
            df["Year"] = year_start
    else:
        df = load_year_range(year_start, year_end)

    if df is None or df.empty:
        st.error("No data found for the selected year(s).")
        st.stop()

    multi_year = (year_start != year_end) and (year_mode == "Multi-Year Span")

    if team != "all" and "Team" in df.columns:
        target = normalize_team(team)
        df = df[df["Team"].astype(str).apply(lambda t: normalize_team(t) == target)]

    if multi_year:
        df = aggregate_player_group(df)

    if min_ip > 0 and "IP" in df.columns:
        df = df[pd.to_numeric(df["IP"], errors="coerce").fillna(0) >= min_ip]

    if search.strip():
        terms = [t.strip() for t in search.split(",") if t.strip()]
        if terms:
            mask = pd.Series(False, index=df.index)
            for term in terms:
                mask |= df["Name"].astype(str).str.contains(term, case=False, na=False)
            df = df[mask]

    if year_mode == "Split Season":
        base_cols = [c for c in ["Name", "Team", "Year"] if c in df.columns]
    else:
        base_cols = [c for c in ["Name", "Team"] if c in df.columns]

    default_sort = next((s for s in selected_stats if s in df.columns), None)

    seen = set()
    deduped_stats = []
    for s in selected_stats:
        if s not in seen:
            seen.add(s)
            deduped_stats.append(s)

    stat_cols  = [s for s in deduped_stats if s in df.columns]
    avail_base = [c for c in base_cols if c in df.columns]
    stat_cols  = [s for s in stat_cols if s not in avail_base]
    display    = df[avail_base + stat_cols].copy()

    if default_sort and default_sort in display.columns:
        display = display.sort_values(default_sort, ascending=False, na_position="last")

    display = display.reset_index(drop=True)
    display.index += 1

    PCT_STAT_SET = PCT_STATS | {s for s in stat_cols if "%" in s}

    for stat in stat_cols:
        if stat in display.columns:
            if stat not in PCT_STAT_SET:
                decimals = STAT_ROUND.get(stat, 1)
                display[stat] = display[stat].apply(
                    lambda v, d=decimals: "" if pd.isna(v) else f"{v:.{d}f}"
                )

    col_config: dict = {}
    rename_map = {stat: STAT_DISPLAY_NAMES.get(stat, stat) for stat in stat_cols}
    display    = display.rename(columns=rename_map)
    for stat in stat_cols:
        label = STAT_DISPLAY_NAMES.get(stat, stat)
        decimals = STAT_ROUND.get(stat, 1)
        if stat in PCT_STAT_SET:
            col_config[label] = st.column_config.NumberColumn(label=label, format=f"%.{decimals}f%%")
        else:
            decimals = STAT_ROUND.get(stat, 1)
            col_config[label] = st.column_config.NumberColumn(label=label, format=f"%.{decimals}f")

    year_label = str(year_start) if year_mode == "Single Season" else f"{year_start}–{year_end}"
    if year_mode == "Single Season":
        mode_label = "Season"
    elif year_mode == "Split Season":
        mode_label = "Split Seasons"
    else:
        mode_label = "Multi-Year Span"

    st.caption(f" {mode_label} – {year_label}")

    st.dataframe(display, width="stretch", height=700, column_config=col_config)

    st.markdown(
        "<div style='text-align:center; color:#888; font-size:1rem; margin-top:1rem;'>"
        "Data: Baseball Reference · FanGraphs · Baseball Savant"
        "</div>",
        unsafe_allow_html=True,
    )
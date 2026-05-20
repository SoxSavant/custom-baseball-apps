import streamlit as st
import pandas as pd
import numpy as np
import html
import unicodedata
from datetime import date

st.set_page_config(page_title="Custom Pitcher Comparison", layout="wide", page_icon="⚾")

st.markdown("""
    <style>
        [data-testid="stMarkdownContainer"] > div {
            overflow-x: auto;
        }
        .compare-card {
            min-width: 320px;
        }
    </style>
""", unsafe_allow_html=True)

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

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
        :root {
            --stat-col-width: 120px;
            --headshot-col-width: 220px;
            --headshot-img-width: 200px;
            --player-name-size: 1.35rem;
            --player-meta-size: 1.3rem;
        }
        [data-testid="stToolbar"] {visibility: hidden;}
        [data-testid="stDecoration"] {display: none;}
        [data-testid="stStatusWidget"] {display: none;}
        .viewerBadge_link__qRi_k {display: none;}
        .compare-card {
            background: #ffffff;
            border: 1px solid #d0d0d0;
            border-radius: 10px;
            padding: 1.25rem 1.5rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.12);
            color: #111111;
            max-width: 100%;
            margin: 0 auto;
        }
        .compare-card .headshot-row {
            display: grid;
            grid-auto-flow: column;
            grid-auto-columns: 1fr;
            grid-template-columns: var(--stat-col-width) 1fr 1fr;
            align-items: center;
            justify-items: center;
            width: 100%;
            max-width: 100%;
            overflow: hidden;
            margin-bottom: .2rem;
            gap: 0;
        }
        .compare-card .headshot-spacer { width: var(--stat-col-width); }
        .compare-card .headshot-col {
            flex: 1 1 auto;
            width: auto;
            max-width: var(--headshot-col-width);
            min-width: 0;
            text-align: center;
            padding-top: .1rem;
        }
        .compare-card .headshot-col img {
            border: 1px solid #d0d0d0;
            background: #f2f2f2;
            border-radius: 4px;
            padding: 4px;
            width: 100%;
            max-width: var(--headshot-img-width);
            max-height: var(--headshot-img-width);
            height: auto;
            object-fit: contain;
        }
        .compare-card .player-name {
            font-size: var(--player-name-size);
            font-weight: 800;
            line-height: 1.2;
            margin: .2rem 0 0 0;
        }
        .compare-card .player-meta {
            color: #555;
            margin: 0 0 0.3rem 0;
            font-size: var(--player-meta-size);
        }
        .compare-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
            table-layout: fixed;
            line-height: 1.5;
        }
        .compare-table td { width: auto; }
        .compare-table th, .compare-table td {
            border: 1px solid #d0d0d0;
            padding: 3px 3px;
            text-align: center;
            background: #ffffff;
            color: #111111;
        }
        .compare-table th {
            background: #f1f1f1;
            font-weight: 800;
            color: #7b0d0d;
            font-size: 15px;
            line-height: 1.2;
        }
        .compare-table .overall-row th {
            background: #f1f1f1;
            color: #7b0d0d;
            font-weight: 800;
            font-size: 15px;
            padding: 5px 0 3px 0;
            border: 1px solid #d0d0d0;
        }
        .compare-table .stat-col {
            font-weight: 700;
            background: #fafafa;
            color: #111;
            width: var(--stat-col-width);
        }
        .compare-table col.col-stat { width: var(--stat-col-width); }
        .compare-table col.col-player { width: auto; }
        .compare-table .best {
            background: #E5F1E4;
            font-weight: 800;
            color: #111111;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

title_col, meta_col = st.columns([3, 1])
with title_col:
    st.title("Custom Pitcher Comparison")
with meta_col:
    st.markdown(
        """
        <div class="mobile-meta" style="text-align: right; font-size: 1rem; padding-top: 0.6rem;">
            Built by <a href="https://twitter.com/Sox_Savant" target="_blank">@Sox_Savant</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


from p_utils import (STAT_ALLOWLIST, TRUTHY_STRINGS, STAT_PRESETS,
get_headshot, label_map, lower_better, start_year, format_stat, load_final_year,
get_player_id_by_name, aggregate_player_group)


def normalize_name(raw: str) -> str:
    if not raw or not isinstance(raw, str):
        return ""
    cleaned = raw.replace("\xa0", " ").strip()
    try:
        cleaned = unicodedata.normalize("NFKD", cleaned).encode("ascii", "ignore").decode()
    except Exception:
        pass
    return " ".join(cleaned.split()).lower()


def build_player_profile(player_id: int, start_year: int, end_year: int) -> pd.Series | None:
    frames = []
    for year in range(start_year, end_year + 1):
        df = load_final_year(year)
        if df is None or df.empty:
            continue
        match = df[df["PlayerId"] == player_id]
        if not match.empty:
            frames.append(match)
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    agg = aggregate_player_group(combined, start_year=start_year)
    if not agg:
        return None
    return pd.Series(agg)


def resolve_player_id(name: str, start_year: int, end_year: int) -> int | None:
    for year in range(end_year, start_year - 1, -1):
        pid = get_player_id_by_name(name, year)
        if pid is not None:
            return pid
    return None

player_mode_options = ["2 players", "3 players", "4 players", "5 players"]
player_mode = st.radio("", player_mode_options, index=0, horizontal=True)
player_count = int(player_mode.split()[0])
column_weights_map = {
    "2 players": [1, 1], "3 players": [1, 1.5],
    "4 players": [1, 2], "5 players": [1, 2.5],
}
column_weights = column_weights_map.get(player_mode, [1, 1])

left_col, right_col = st.columns(column_weights)
with left_col:
    controls_container = st.container()
    stat_builder_container = st.container()

current_year = date.today().year
years_desc = list(range(current_year, start_year-1, -1))
MAX_PLAYERS = 5
default_names = ["Cam Schlittler", "José Soriano", "", "", ""]

prev_count = st.session_state.get("comp_prev_player_count", 2)
if player_count > prev_count:
    for idx in range(prev_count, player_count):
        st.session_state[f"comp_single_year_{idx}"] = True
st.session_state["comp_prev_player_count"] = player_count

for idx in range(MAX_PLAYERS):
    for key, default in [
        (f"comp_player_{idx}", default_names[idx] if idx < len(default_names) else ""),
        (f"comp_player_{idx}_id", ""),
        (f"comp_player_{idx}_mode", "Name"),
        (f"comp_single_year_{idx}", True),
        (f"comp_year_{idx}_single", years_desc[0]),
        (f"comp_year_{idx}_start", years_desc[0]),
        (f"comp_year_{idx}_end", years_desc[0]),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

with controls_container:
    year_cols = st.columns(player_count)
    year_ranges: list[tuple[int, int]] = []
    for idx in range(player_count):
        label = chr(ord("A") + idx)
        with year_cols[idx]:
            single = st.checkbox(f"Single season (Player {label})", key=f"comp_single_year_{idx}")
            if single:
                year_single = st.selectbox(f"Season (Player {label})", years_desc, index=0, key=f"comp_year_{idx}_single")
                year_start = year_single
                year_end = year_single
            else:
                year_start = st.selectbox(f"Season Start (Player {label})", years_desc, index=0, key=f"comp_year_{idx}_start")
                year_end = st.selectbox(f"Season End (Player {label})", years_desc, index=0, key=f"comp_year_{idx}_end")
        year_ranges.append((min(year_start, year_end), max(year_start, year_end)))

    input_cols = st.columns(player_count)
    player_inputs = []
    for idx in range(player_count):
        label = chr(ord("A") + idx)
        with input_cols[idx]:
            mode_val = st.selectbox(f"Player {label} Input", ["Name", "FanGraphs ID"], key=f"comp_player_{idx}_mode")
            if mode_val == "Name":
                name_input = st.text_input(f"Player {label}", key=f"comp_player_{idx}")
                id_input = st.session_state.get(f"comp_player_{idx}_id", "")
            else:
                id_input = st.text_input(f"Player {label} FanGraphs ID", key=f"comp_player_{idx}_id")
                name_input = st.session_state.get(f"comp_player_{idx}", "")
        player_inputs.append({
            "mode": mode_val,
            "name_input": name_input.strip(),
            "id_input": str(id_input).strip(),
            "years": year_ranges[idx],
        })

players_data = []
for idx, cfg in enumerate(player_inputs):
    label = chr(ord("A") + idx)
    start_year, end_year = cfg["years"]

    if cfg["mode"] == "Name":
        if not cfg["name_input"]:
            st.warning(f"Enter a name for Player {label} or switch to FanGraphs ID input.")
            st.stop()
        player_id = resolve_player_id(cfg["name_input"], start_year, end_year)
        if not player_id:
            st.error(f"Could not find '{cfg['name_input']}' in the dataset. Check spelling or use FanGraphs ID.")
            st.stop()
    else:
        if not cfg["id_input"]:
            st.warning(f"Enter a FanGraphs ID for Player {label}.")
            st.stop()
        try:
            player_id = int(cfg["id_input"])
        except Exception:
            st.error(f"Player {label} FanGraphs ID must be a positive integer.")
            st.stop()

    player_row = build_player_profile(player_id, start_year, end_year)
    if player_row is None:
        st.error(f"Could not load data for Player {label} (ID: {player_id}).")
        st.stop()

    display_name = str(player_row.get("Name", "")).strip()
    if not display_name:
        display_name = cfg["name_input"] if cfg["mode"] == "Name" else f"FG#{player_id}"

    team_display = str(player_row.get("Team", "N/A"))
    year_label = f"{start_year}" if start_year == end_year else f"{start_year}-{end_year}"
    df = pd.DataFrame([player_row])

    players_data.append({
        "player_id": player_id,
        "display_name": display_name,
        "team": team_display,
        "year_label": year_label,
        "df": df,
        "row": player_row,
        "label_char": label,
    })

seen_labels: set[str] = set()
for pdata in players_data:
    base = pdata["display_name"]
    label = base
    if label in seen_labels and pdata["year_label"]:
        label = f"{base} ({pdata['year_label']})"
    if label in seen_labels:
        label = f"{base} (Player {pdata['label_char']})"
    seen_labels.add(label)
    pdata["col_label"] = label

dfs = [p["df"] for p in players_data]

# ─────────────────────────────────────────────
#  Stat options
# ─────────────────────────────────────────────

stat_exclusions = {"Season", "PlayerId", "MLBAMID", "W", "L"}
numeric_sets = []
for df in dfs:
    numeric_sets.append({col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])})
if not numeric_sets:
    st.error("No numeric stats available to display.")
    st.stop()

numeric_stats = list((numeric_sets[0] if len(numeric_sets) == 1 else set.intersection(*numeric_sets)) - stat_exclusions)

if any("bWAR" in df.columns for df in dfs) and "bWAR" not in numeric_stats:
    numeric_stats.append("bWAR")

preferred_stats = [stat for stat in STAT_ALLOWLIST if stat in numeric_stats]
other_stats = [stat for stat in numeric_stats if stat not in preferred_stats]
stat_options = preferred_stats + other_stats
allowed_add_stats = preferred_stats if preferred_stats else stat_options.copy()

has_record = all("W" in df.columns and "L" in df.columns for df in dfs)
if has_record:
    if "W-L" not in stat_options:
        stat_options = ["W-L"] + stat_options
    if "W-L" not in allowed_add_stats:
        allowed_add_stats = ["W-L"] + allowed_add_stats

if not stat_options:
    st.error("No numeric stats available to display.")
    st.stop()

default_preset_name = "Default"
stat_preset_key = "comp_stat_preset_select"
preset_options = list(STAT_PRESETS.keys())
stat_state_key = "comp_stat_config"
manual_stat_update_key = "comp_stat_config_manual_update"
add_select_key = "comp_add_stat_select"
remove_select_key = "comp_remove_stat_select"
add_reset_key = "comp_reset_add_select"
remove_reset_key = "comp_reset_remove_select"
stat_version_key = "comp_stat_config_version"


def bump_stat_config_version():
    st.session_state[stat_version_key] = st.session_state.get(stat_version_key, 0) + 1


def normalize_stat_rows(rows, fallback):
    cleaned = []
    seen_stats: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        stat_name = row.get("Stat")
        if not stat_name or stat_name not in stat_options or stat_name in seen_stats:
            continue
        show_val = row.get("Show", True)
        if pd.isna(show_val):
            show_bool = True
        elif isinstance(show_val, str):
            show_bool = show_val.strip().lower() in TRUTHY_STRINGS
        else:
            show_bool = bool(show_val)
        cleaned.append({"Stat": stat_name, "Show": show_bool})
        seen_stats.add(stat_name)
    if not cleaned:
        cleaned = [row.copy() for row in fallback]
    return cleaned


def add_stat_callback(stat_key, select_key, reset_key, sentinel):
    choice = st.session_state.get(select_key)
    if not choice or choice == sentinel:
        return
    current_preset = st.session_state.get(stat_preset_key, default_preset_name)
    base = [{"Stat": s, "Show": True} for s in STAT_PRESETS[current_preset] if s in stat_options] or [{"Stat": stat_options[0], "Show": True}]
    config = normalize_stat_rows(st.session_state.get(stat_key, base), base)
    if not any(r["Stat"] == choice for r in config):
        config.append({"Stat": choice, "Show": True})
    st.session_state[stat_key] = config
    bump_stat_config_version()
    st.session_state[manual_stat_update_key] = True
    st.session_state[reset_key] = True


def remove_stat_callback(stat_key, select_key, reset_key, sentinel):
    choice = st.session_state.get(select_key)
    if not choice or choice == sentinel:
        return
    current_preset = st.session_state.get(stat_preset_key, default_preset_name)
    base = [{"Stat": s, "Show": True} for s in STAT_PRESETS[current_preset] if s in stat_options] or [{"Stat": stat_options[0], "Show": True}]
    config = normalize_stat_rows(st.session_state.get(stat_key, base), base)
    config = [r for r in config if r.get("Stat") != choice] or [r.copy() for r in base]
    st.session_state[stat_key] = config
    bump_stat_config_version()
    st.session_state[manual_stat_update_key] = True
    st.session_state[reset_key] = True


def stat_preset_callback(preset_key, stat_key, available_stats):
    preset_name = st.session_state.get(preset_key, default_preset_name)

    def compute_leads(pidx):
        leads = []
        if len(players_data) < 2:
            return leads
        for stat in available_stats:
            if stat == "W-L" or stat not in STAT_ALLOWLIST:
                continue
            if any(stat not in p["df"].columns for p in players_data):
                continue
            vals = []
            for p in players_data:
                v = p["row"].get(stat, np.nan)
                try:
                    vals.append(float(v) if pd.notna(v) else np.nan)
                except Exception:
                    vals.append(np.nan)
            if pd.isna(vals[pidx]):
                continue
            is_lb = stat in lower_better
            better = True
            for i, other in enumerate(vals):
                if i == pidx or pd.isna(other):
                    continue
                if is_lb:
                    if not (vals[pidx] + 1e-9 < other):
                        better = False; break
                else:
                    if not (vals[pidx] > other + 1e-9):
                        better = False; break
            if better:
                leads.append(stat)
        return leads

    if preset_name.startswith("Player ") and preset_name.endswith(" leads"):
        try:
            letter = preset_name.split()[1]
            pidx = ord(letter.upper()) - ord("A") if len(letter) == 1 and letter.isalpha() else None
        except Exception:
            pidx = None
        if pidx is not None and 0 <= pidx < len(players_data):
            leads = compute_leads(pidx)
            if leads:
                st.session_state[stat_key] = [{"Stat": s, "Show": True} for s in leads]
                bump_stat_config_version()
                st.session_state[manual_stat_update_key] = True
                st.session_state[add_reset_key] = True
                st.session_state[remove_reset_key] = True
                return

    preset_stats = [s for s in STAT_PRESETS.get(preset_name, []) if s in available_stats]
    if not preset_stats and available_stats:
        preset_stats = [available_stats[0]]
    if not preset_stats:
        return
    st.session_state[stat_key] = [{"Stat": s, "Show": True} for s in preset_stats]
    bump_stat_config_version()
    st.session_state[manual_stat_update_key] = True
    st.session_state[add_reset_key] = True
    st.session_state[remove_reset_key] = True


def move_stat_row(delta, index, fallback):
    rows = normalize_stat_rows(st.session_state.get(stat_state_key, fallback), fallback)
    target = index + delta
    if 0 <= target < len(rows):
        rows[index], rows[target] = rows[target], rows[index]
        st.session_state[stat_state_key] = rows
        bump_stat_config_version()
        st.session_state[manual_stat_update_key] = True


if stat_state_key not in st.session_state:
    st.session_state[stat_preset_key] = default_preset_name
    candidates = [s for s in STAT_PRESETS[default_preset_name] if s in stat_options] or [stat_options[0]]
    st.session_state[stat_state_key] = [{"Stat": s, "Show": True} for s in candidates]
    st.session_state[stat_version_key] = 0
elif stat_version_key not in st.session_state:
    st.session_state[stat_version_key] = 0

current_preset = st.session_state.get(stat_preset_key, default_preset_name)
preset_base_candidates = [s for s in STAT_PRESETS[current_preset] if s in stat_options] or [stat_options[0]]
preset_base_config = [{"Stat": s, "Show": True} for s in preset_base_candidates]
current_stat_config = normalize_stat_rows(st.session_state.get(stat_state_key, preset_base_config), preset_base_config)

# ─────────────────────────────────────────────
#  Stat builder UI
# ─────────────────────────────────────────────

with stat_builder_container:
    prior_preset = st.session_state.get(stat_preset_key, default_preset_name)
    preset_index = preset_options.index(prior_preset) if prior_preset in preset_options else 0
    st.selectbox(
        "Stat Preset", preset_options, index=preset_index,
        key=stat_preset_key, on_change=stat_preset_callback,
        args=(stat_preset_key, stat_state_key, stat_options),
    )

    st.markdown("### Customize stats")
    st.markdown(
        "<div style='margin-bottom: -0.25rem; color: inherit; font-size: 0.9rem;'>"
        "Use the drop downs to add or remove stats</div>",
        unsafe_allow_html=True,
    )

    stats_in_config = [r.get("Stat") for r in current_stat_config if r.get("Stat")]
    available_pool = allowed_add_stats or stat_options
    available_stats = [s for s in available_pool if s not in stats_in_config]

    sentinel_add = "Select stat to add"
    sentinel_remove = "Select stat to remove"
    add_options = [sentinel_add] + available_stats
    remove_options = [sentinel_remove] + stats_in_config

    if st.session_state.get(add_select_key) not in add_options:
        st.session_state[add_select_key] = sentinel_add
    if st.session_state.pop(add_reset_key, False):
        st.session_state[add_select_key] = sentinel_add
    if st.session_state.get(remove_select_key) not in remove_options:
        st.session_state[remove_select_key] = sentinel_remove
    if st.session_state.pop(remove_reset_key, False):
        st.session_state[remove_select_key] = sentinel_remove

    add_col, remove_col = st.columns(2)
    with add_col:
        st.selectbox("Add stat", add_options, label_visibility="hidden",
            key=add_select_key, on_change=add_stat_callback,
            args=(stat_state_key, add_select_key, add_reset_key, sentinel_add))
    with remove_col:
        st.selectbox("Remove stat", remove_options, label_visibility="hidden",
            key=remove_select_key, on_change=remove_stat_callback,
            args=(stat_state_key, remove_select_key, remove_reset_key, sentinel_remove))

    current_stat_config = normalize_stat_rows(
        st.session_state.get(stat_state_key, preset_base_config), preset_base_config
    )

    config_df = pd.DataFrame([
        {"Stat": row["Stat"], "Show": bool(row.get("Show", True))}
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
            st.session_state[stat_state_key] = new_config
            bump_stat_config_version()
            current_stat_config = normalize_stat_rows(st.session_state[stat_state_key], preset_base_config)

    move_col, up_col, down_col = st.columns([3, 1, 1])
    with move_col:
        move_stat = st.selectbox(
            "Reorder",
            [""] + [r["Stat"] for r in current_stat_config],
            label_visibility="collapsed",
            key="stat_reorder_select",
        )
    idx = next((i for i, r in enumerate(current_stat_config) if r["Stat"] == move_stat), None)
    with up_col:
        st.button("▲", key="move_up", disabled=not move_stat or idx == 0,
                  on_click=move_stat_row, args=(-1, idx if idx is not None else 0, preset_base_config))
    with down_col:
        st.button("▼", key="move_down", disabled=not move_stat or idx == len(current_stat_config) - 1,
                  on_click=move_stat_row, args=(1, idx if idx is not None else 0, preset_base_config))

    st.session_state[stat_state_key] = normalize_stat_rows(
        st.session_state.get(stat_state_key, current_stat_config), preset_base_config
    )

# ─────────────────────────────────────────────
#  Build comparison table
# ─────────────────────────────────────────────

stats_order = [r["Stat"] for r in st.session_state[stat_state_key] if r.get("Show", True)]
if not stats_order:
    st.info("Add at least one stat and mark it as shown to build the comparison.")
    st.stop()

comparison_rows = []
winner_map: dict[str, set[str]] = {}
col_order = [p["col_label"] for p in players_data]

for stat in stats_order:
    if stat == "W-L":
        if any("W" not in p["df"].columns or "L" not in p["df"].columns for p in players_data):
            continue
        values, ratios = [], []
        for pdata in players_data:
            w = pd.to_numeric(pdata["row"].get("W", np.nan), errors="coerce")
            l = pd.to_numeric(pdata["row"].get("L", np.nan), errors="coerce")
            if pd.isna(w) or pd.isna(l):
                values.append(""); ratios.append(np.nan)
            else:
                values.append(f"{int(round(w))}-{int(round(l))}")
                total = w + l
                ratios.append(np.nan if total <= 0 else w / total)
        winners = set()
        cands = [v for v in ratios if not pd.isna(v)]
        if cands:
            best = max(cands)
            winners = {col_order[i] for i, v in enumerate(ratios) if not pd.isna(v) and abs(v - best) < 1e-9}
        row_dict = {"Stat": "W-L"}
        for i, pdata in enumerate(players_data):
            row_dict[pdata["col_label"]] = values[i]
        comparison_rows.append(row_dict)
        winner_map["W-L"] = winners
        continue

    if any(stat not in p["df"].columns for p in players_data):
        continue

    raw_label = label_map.get(stat, stat)
    values, numeric_vals = [], []
    has_non_numeric = False

    for pdata in players_data:
        val = pdata["row"].get(stat, np.nan)
        values.append(val)
        if pd.isna(val):
            numeric_vals.append(np.nan)
        else:
            try:
                numeric_vals.append(float(val))
            except Exception:
                has_non_numeric = True
                numeric_vals.append(np.nan)

    winners: set[str] = set()
    cands = [v for v in numeric_vals if not pd.isna(v)]
    if cands and not has_non_numeric:
        best = min(cands) if stat in lower_better else max(cands)
        winners = {col_order[i] for i, v in enumerate(numeric_vals) if not pd.isna(v) and abs(v - best) < 1e-9}

    row_dict = {"Stat": raw_label}
    for i, pdata in enumerate(players_data):
        row_dict[pdata["col_label"]] = format_stat(stat, values[i])
    comparison_rows.append(row_dict)
    winner_map[raw_label] = winners

table_df = pd.DataFrame(comparison_rows, columns=["Stat"] + col_order)

for pdata in players_data:
    pdata["headshot"] = get_headshot(pdata["row"])

esc = html.escape

if player_count == 2:
    stat_col_width = "calc(100% / 3)"
    player_col_width = "calc(100% / 3)"
    grid_template = "1fr 1fr 1fr"
    headshot_width = 200
    headshot_col_width = 220
    player_name_size = "1.35rem"
    player_meta_size = "1.3rem"
else:
    shared = f"calc(100% / {player_count + 1})"
    stat_col_width = player_col_width = shared
    grid_template = " ".join(["1fr"] * (player_count + 1))
    headshot_width = f"clamp(110px, calc(80vw / {player_count + 1}), 140px)"
    headshot_col_width = f"clamp(125px, calc(84vw / {player_count + 1}), 160px)"
    player_name_size = ".9rem"
    player_meta_size = ".95rem"

name_style_attr = f' style="font-size:{player_name_size}; line-height:1.1;"' if player_count > 2 else ""
hw_suffix = "px" if isinstance(headshot_width, int) else ""
hcw_suffix = "px" if isinstance(headshot_col_width, int) else ""

with right_col:
    if table_df.empty:
        st.warning("No stats available to compare.")
    else:
        rows = [
            f'<div class="compare-card" style="'
            f'--stat-col-width: {stat_col_width}; '
            f'--headshot-col-width: {headshot_col_width}{hcw_suffix}; '
            f'--headshot-img-width: {headshot_width}{hw_suffix}; '
            f'--player-name-size: {player_name_size}; '
            f'--player-meta-size: {player_meta_size};">',
            f'  <div class="headshot-row" style="grid-template-columns: {grid_template};">',
        ]

        if player_count == 2:
            for i, pdata in enumerate([players_data[0], None, players_data[1]]):
                if pdata is None:
                    rows.append('    <div class="headshot-spacer"></div>')
                else:
                    img = f'<img src="{esc(pdata["headshot"])}" width="{headshot_width}" />' if pdata["headshot"] else ""
                    rows.extend([
                        '    <div class="headshot-col">',
                        f'      <div class="player-meta">{esc(str(pdata["year_label"]))} | {esc(str(pdata["team"]))}</div>',
                        f'      {img}',
                        f'      <div class="player-name"{name_style_attr}>{esc(pdata["display_name"])}</div>',
                        '    </div>',
                    ])
        else:
            rows.append('    <div class="headshot-spacer"></div>')
            for pdata in players_data:
                img = f'<img src="{esc(pdata["headshot"])}" width="{headshot_width}" />' if pdata["headshot"] else ""
                rows.extend([
                    '    <div class="headshot-col">',
                    f'      <div class="player-meta">{esc(str(pdata["year_label"]))} | {esc(str(pdata["team"]))}</div>',
                    f'      {img}',
                    f'      <div class="player-name"{name_style_attr}>{esc(pdata["display_name"])}</div>',
                    '    </div>',
                ])

        rows.extend(['  </div>', '  <table class="compare-table">', '    <colgroup>'])

        if player_count == 2:
            rows += [
                f'      <col class="col-player" style="width: {player_col_width};" />',
                f'      <col class="col-stat" style="width: {stat_col_width};" />',
                f'      <col class="col-player" style="width: {player_col_width};" />',
            ]
            render_cols = [players_data[0]["col_label"], "__STAT__", players_data[1]["col_label"]]
        else:
            rows.append(f'      <col class="col-stat" style="width: {stat_col_width};" />')
            for _ in players_data:
                rows.append(f'      <col class="col-player" style="width: {player_col_width};" />')
            render_cols = ["__STAT__"] + [p["col_label"] for p in players_data]

        rows += [
            '    </colgroup>', '    <thead>',
            f'      <tr class="overall-row"><th colspan="{player_count + 1}">Overall Stats</th></tr>',
            '    </thead>', '    <tbody>',
        ]

        for row in comparison_rows:
            stat_label = esc(str(row["Stat"]))
            winners = winner_map.get(str(row["Stat"]), set())
            rows.append("      <tr>")
            for col_id in render_cols:
                if col_id == "__STAT__":
                    rows.append(f'        <td class="stat-col">{stat_label}</td>')
                else:
                    val = esc(str(row.get(col_id, "")))
                    cell_class = "best" if col_id in winners else ""
                    rows.append(f'        <td class="{cell_class}">{val}</td>')
            rows.append("      </tr>")

        rows += [
            '    </tbody>', '  </table>',
            '  <div style="display:flex; justify-content:space-between; margin-top:0.35rem; color:#555; font-size:0.9rem;">',
            '    <div>By: Sox_Savant</div>',
            '    <div>Data: FanGraphs, Bref</div>',
            '  </div>',
            '</div>',
        ]

        st.markdown("\n".join(rows), unsafe_allow_html=True)
        st.caption("Screenshot to save")
        st.caption("Find a player's FanGraphs ID in their FanGraphs profile URL")
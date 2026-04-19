import unicodedata
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from io import BytesIO
from datetime import date

st.set_page_config(page_title="Custom Team Pitching Savant Page", layout="wide", page_icon="⚾")

st.markdown(
    """
    <style>
        [data-testid="stToolbar"] {visibility: hidden;}
        [data-testid="stDecoration"] {display: none;}
        [data-testid="stStatusWidget"] {display: none;}
        .viewerBadge_link__qRi_k {display: none;}
    </style>
    """,
    unsafe_allow_html=True,
)

title_col, meta_col = st.columns([3, 1])
with title_col:
    st.title("Custom Team Pitcher Savant Page")
with meta_col:
    st.markdown(
        """
        <div style="text-align: right; font-size: 1rem; padding-top: 0.6rem;">
            Built by <a href="https://twitter.com/Sox_Savant" target="_blank">@Sox_Savant</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────
from p_utils import (STAT_ALLOWLIST, format_stat, TEAMS, TRUTHY_STRINGS,
 label_map, lower_better,  EVERY_STAT_PRESET,
normalize_team, start_year,load_final_year)

STAT_PRESETS = {
    "Statcast": [
        "fWAR", "bWAR", "xERA", "EV", "Chase%", "Whiff%",
        "K%", "BB%", "Barrel%", "HardHit%", "GB%",
    ],
    "Standard": [
        "bWAR", "fWAR", "GS", "IP", "ERA", "FIP", "WHIP",
        "K/9", "BB/9", "HR/9", "K%", "BB%",
    ],
    "Miscellaneous": [
        "K-BB%", "WPA", "Clutch", "GB%", "HR/FB",
    ],
    "Every Stat": EVERY_STAT_PRESET,
}



# ─────────────────────────────────────────────
#  Team helpers
# ─────────────────────────────────────────────



def get_teams_for_year(season: int) -> dict[str, str]:
    key = "ATH" if season >= 2025 else "OAK"
    return {k: v for k, v in TEAMS.items() if k not in {"OAK", "ATH"} or k == key}


def get_team_nickname(full_name: str) -> str:
    for city in ["Kansas City", "Los Angeles", "New York", "San Diego",
                 "San Francisco", "St. Louis", "Tampa Bay"]:
        if full_name.startswith(f"{city} "):
            return full_name[len(city) + 1:]
    return full_name.split(" ", 1)[-1]


def normalize_name(raw: str) -> str:
    if not raw or not isinstance(raw, str):
        return ""
    cleaned = raw.replace("\xa0", " ").strip()
    try:
        cleaned = unicodedata.normalize("NFKD", cleaned).encode("ascii", "ignore").decode()
    except Exception:
        pass
    return " ".join(cleaned.split()).lower()

current_year = date.today().year

# ─────────────────────────────────────────────
#  Controls
# ─────────────────────────────────────────────

from utils import get_dynamic_min_ip

min_ip = get_dynamic_min_ip(current_year)

left_col, right_col = st.columns([1, 1.3])

with left_col:
    year = st.selectbox("Select Year", list(range(current_year, start_year-1, -1)))
    teams_for_year = get_teams_for_year(year)
    team_options = list(teams_for_year.keys())
    team_select_key = "team_abbr_select"
    preferred = st.session_state.get(team_select_key, "LAA" if "LAA" in team_options else team_options[0])
    if preferred not in team_options:
        preferred = team_options[0]
    team_abbr = st.selectbox(
        "Team", team_options,
        index=team_options.index(preferred),
        key=team_select_key,
    )
    # ── Minimum IP input (with dynamic default)
    if "pitcher_team_min_ip" not in st.session_state:
        st.session_state.pitcher_team_min_ip = min_ip 

    if "pitcher_team_last_year" not in st.session_state:
        st.session_state.pitcher_team_last_year = year

    if year != st.session_state.pitcher_team_last_year:
        st.session_state.pitcher_team_min_ip = get_dynamic_min_ip(year)
        st.session_state.pitcher_team_last_year = year

    min_ip = st.number_input(
        "Minimum IP",
        min_value=0,
        max_value=250,
        key="pitcher_team_min_ip"
    )

stat_builder_container = left_col.container()

# ─────────────────────────────────────────────
#  Load data
# ─────────────────────────────────────────────

team_full_name = TEAMS.get(team_abbr, team_abbr)
nickname = get_team_nickname(team_full_name)
logo_dir = Path(__file__).parent / "logos"
logo_path = logo_dir / f"{nickname}.png"
logo_img = None
if logo_path.exists():
    try:
        logo_img = mpimg.imread(logo_path)
    except Exception:
        pass

df = load_final_year(year)

if df is None or df.empty:
    st.error(f"No data found for {year}.")
    st.stop()

df["IP"] = pd.to_numeric(df.get("IP", 0), errors="coerce")

if "Team" in df.columns:
    df["Team"] = df["Team"].astype(str).str.strip()


# League for percentiles
from utils import get_percentile_min_ip
PCT_IP = get_percentile_min_ip(year)
league_for_pct = df[df["IP"] >= PCT_IP].copy()
if league_for_pct.empty:
    st.error(f"No pitchers with ≥ {PCT_IP} IP in {year}.")
    st.stop()

# Filter to selected team
target_team = normalize_team(team_abbr)
team_df = df[df["Team"].astype(str).apply(normalize_team) == target_team].copy()
team_df = team_df[team_df["IP"] >= min_ip]

if team_df.empty:
    st.warning(f"No pitchers on {team_abbr} with ≥ {min_ip} IP in {year}.")
    st.stop()

# ─────────────────────────────────────────────
#  Stat builder setup
# ─────────────────────────────────────────────

stat_exclusions = {"Season", "PlayerId", "MLBAMID"}
numeric_stats = [
    col for col in df.columns
    if pd.api.types.is_numeric_dtype(df[col]) and col not in stat_exclusions
]

preferred_stats = [s for s in STAT_ALLOWLIST if s in numeric_stats]
other_stats = [s for s in numeric_stats if s not in preferred_stats]
stat_options = preferred_stats + other_stats
allowed_add_stats = preferred_stats if preferred_stats else stat_options.copy()

if not stat_options:
    st.error("No numeric stats available.")
    st.stop()

default_preset_name = "Statcast"
stat_preset_key = "stat_preset_select"
preset_options = list(STAT_PRESETS.keys())
stat_state_key = "stat_config"
manual_stat_update_key = "stat_config_manual_update"
add_select_key = "add_stat_select"
remove_select_key = "remove_stat_select"
add_reset_key = "reset_add_select"
remove_reset_key = "reset_remove_select"
stat_version_key = "stat_config_version"


def bump_stat_config_version():
    st.session_state[stat_version_key] = st.session_state.get(stat_version_key, 0) + 1


def normalize_stat_rows(rows, fallback):
    cleaned = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        stat_name = row.get("Stat")
        if not stat_name or stat_name not in stat_options or stat_name in seen:
            continue
        show_val = row.get("Show", True)
        if pd.isna(show_val):
            show_bool = True
        elif isinstance(show_val, str):
            show_bool = show_val.strip().lower() in TRUTHY_STRINGS
        else:
            show_bool = bool(show_val)
        cleaned.append({"Stat": stat_name, "Show": show_bool})
        seen.add(stat_name)
    return cleaned if cleaned else [r.copy() for r in fallback]


def _preset_base_config():
    preset = st.session_state.get(stat_preset_key, default_preset_name)
    candidates = [s for s in STAT_PRESETS.get(preset, []) if s in stat_options] or [stat_options[0]]
    return [{"Stat": s, "Show": True} for s in candidates]


def add_stat_callback(stat_key, select_key, reset_key, sentinel):
    choice = st.session_state.get(select_key)
    if not choice or choice == sentinel:
        return
    base = _preset_base_config()
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
    base = _preset_base_config()
    config = normalize_stat_rows(st.session_state.get(stat_key, base), base)
    config = [r for r in config if r.get("Stat") != choice] or [r.copy() for r in base]
    st.session_state[stat_key] = config
    bump_stat_config_version()
    st.session_state[manual_stat_update_key] = True
    st.session_state[reset_key] = True


def stat_preset_callback(preset_key, stat_key, available_stats):
    preset_name = st.session_state.get(preset_key, default_preset_name)
    filtered = [s for s in STAT_PRESETS.get(preset_name, []) if s in available_stats]
    if not filtered and available_stats:
        filtered = [available_stats[0]]
    if not filtered:
        return
    st.session_state[stat_key] = [{"Stat": s, "Show": True} for s in filtered]
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


def toggle_stat_show(index, state_key, fallback):
    rows = normalize_stat_rows(st.session_state.get(stat_state_key, fallback), fallback)
    if 0 <= index < len(rows):
        rows[index]["Show"] = bool(st.session_state.get(state_key, True))
        st.session_state[stat_state_key] = rows
        bump_stat_config_version()
        st.session_state[manual_stat_update_key] = True


if stat_state_key not in st.session_state:
    st.session_state[stat_preset_key] = default_preset_name
    st.session_state[stat_state_key] = _preset_base_config()
    st.session_state[stat_version_key] = 0
elif stat_version_key not in st.session_state:
    st.session_state[stat_version_key] = 0

preset_base_config = _preset_base_config()
current_stat_config = normalize_stat_rows(
    st.session_state.get(stat_state_key, preset_base_config), preset_base_config
)

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
        "Use the drop downs to add or remove stats and the arrows to reorder.</div>",
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
        st.selectbox(
            "Add stat", add_options, label_visibility="hidden",
            key=add_select_key, on_change=add_stat_callback,
            args=(stat_state_key, add_select_key, add_reset_key, sentinel_add),
        )
    with remove_col:
        st.selectbox(
            "Remove stat", remove_options, label_visibility="hidden",
            key=remove_select_key, on_change=remove_stat_callback,
            args=(stat_state_key, remove_select_key, remove_reset_key, sentinel_remove),
        )

    current_stat_config = normalize_stat_rows(
        st.session_state.get(stat_state_key, preset_base_config), preset_base_config
    )

    st.markdown("#### Order & visibility")
    header_cols = st.columns([0.25, 0.25, 0.25, 0.25])
    header_cols[0].markdown("**Up**")
    header_cols[1].markdown("**Down**")
    header_cols[2].markdown("**Stat**")
    header_cols[3].markdown("**Show**")

    for idx, row in enumerate(current_stat_config):
        up_col, down_col, stat_col, show_col = st.columns([0.25, 0.25, 0.25, 0.25])
        with up_col:
            st.button("▲", key=f"stat_up_{idx}", disabled=idx == 0,
                on_click=move_stat_row, args=(-1, idx, preset_base_config))
        with down_col:
            st.button("▼", key=f"stat_down_{idx}", disabled=idx == len(current_stat_config) - 1,
                on_click=move_stat_row, args=(1, idx, preset_base_config))
        with stat_col:
            sn = row.get("Stat", "")
            st.write(sn)
        with show_col:
            ck = f"stat_show_{idx}"
            st.checkbox("", value=bool(row.get("Show", True)), key=ck,
                label_visibility="collapsed", on_change=toggle_stat_show,
                args=(idx, ck, preset_base_config))

    st.session_state[stat_state_key] = normalize_stat_rows(
        st.session_state.get(stat_state_key, current_stat_config), preset_base_config
    )


# ─────────────────────────────────────────────
#  Build leader rows
# ─────────────────────────────────────────────

stats_order = [r["Stat"] for r in st.session_state[stat_state_key] if r.get("Show", True)]
if not stats_order:
    st.info("Add at least one stat and mark it as shown to build the chart.")
    st.stop()

leaders = []
for stat in stats_order:
    if stat not in df.columns:
        continue
    team_vals = team_df[[stat, "Name"]].dropna(subset=[stat])
    if team_vals.empty:
        continue
    league_vals = league_for_pct[stat].dropna()
    if league_vals.empty:
        continue

    team_leader_row = team_vals.sort_values(stat, ascending=stat in lower_better).iloc[0]
    leader_val = float(team_leader_row[stat])

    pct = (league_vals <= leader_val).mean() * 100.0
    if stat in lower_better:
        pct = 100 - pct
    pct = float(np.clip(pct, 0, 100))

    leaders.append({
        "Stat": label_map.get(stat, stat),
        "Leader": team_leader_row["Name"],
        "Value": leader_val,
        "Pct": pct,
    })

lead_df = pd.DataFrame(leaders)
if lead_df.empty:
    st.warning("No stats available to display.")
    st.stop()

lead_df["Display"] = lead_df.apply(lambda r: format_stat(r["Stat"], r["Value"]), axis=1)

# ─────────────────────────────────────────────
#  Render chart
# ─────────────────────────────────────────────

with right_col:
    cmap = LinearSegmentedColormap.from_list(
        "savant",
        [(0, "#335AA1"), (0.5, "#E8E8E8"), (1, "#D92229")],
    )

    fig_height = 1.25 + len(lead_df) * 0.4
    fig, ax = plt.subplots(figsize=(7.5, fig_height))

    top_pad = 0.75 / fig_height
    ax.set_position([0.08, 0.12, 0.8, 1.0 - top_pad - 0.12])

    title_y = 1 - 0.3 / fig_height
    subtitle_y = 1 - 0.6 / fig_height

    fig.text(0.5, title_y, f"{year} {team_full_name}",
             ha="center", va="center", fontsize=22, fontweight="bold")
    fig.text(0.5, subtitle_y, f"(min {min_ip} IP)",
             ha="center", va="center", fontsize=13, color="#555")
    fig.text(0.2, 0.08, "By: Sox_Savant", ha="center", va="center", fontsize=10, color="#555")
    fig.text(0.7, 0.08, "Data: FanGraphs, Bref", ha="center", va="center", fontsize=10, color="#555")

    y = np.arange(len(lead_df))
    TRACK_H     = 0.82
    BAR_H       = 0.82
    LEFT_OFFSET = 3
    BAR_LENGTH  = 45
    VALUE_X     = LEFT_OFFSET + BAR_LENGTH + 12
    BUBBLE_SIZE = 650

    ax.barh(y, BAR_LENGTH, left=LEFT_OFFSET, height=TRACK_H, color="#F1F1F1", edgecolor="none")

    for i, row in lead_df.iterrows():
        pct = row["Pct"]
        color = cmap(pct / 100)
        bar_width = pct / 100 * BAR_LENGTH
        ax.barh(i, bar_width, left=LEFT_OFFSET, height=BAR_H, color=color, edgecolor="none")

        name = row["Leader"]
        bubble_x = LEFT_OFFSET + bar_width
        needs_shift = pct < len(str(name)) * 3.7
        if needs_shift:
            name_x = bubble_x + (VALUE_X - bubble_x) * 0.2 - 2
            name_ha = "left"
        else:
            name_x = LEFT_OFFSET + bar_width / 2 + 0
            name_ha = "center"

        ax.text(name_x, i, name, ha=name_ha, va="center", fontsize=13, fontweight="bold", color="#111")
        ax.scatter(bubble_x, i, s=BUBBLE_SIZE, color=color, edgecolors="white", linewidth=2.4, zorder=3)
        ax.text(bubble_x, i + 0.04, f"{int(round(pct))}", ha="center", va="center",
                fontsize=11, fontweight="bold", color="white")
        ax.text(VALUE_X - 9.5, i, row["Display"], ha="left", va="center", fontsize=12, color="#111")
        ax.text(0, i, row["Stat"], ha="right", va="center", fontsize=13)

    for pos in (0.1, 0.5, 0.9):
        ax.vlines(LEFT_OFFSET + BAR_LENGTH * pos, -0.5, len(lead_df) - 0.5,
                  colors="white", linewidth=1.2, alpha=0.25, zorder=2.6)

    ax.set_xlim(-10, VALUE_X)
    ax.set_ylim(-0.5, len(lead_df) - 0.5)
    ax.invert_yaxis()
    ax.axis("off")

    st.pyplot(fig, use_container_width=True, clear_figure=False)

    pdf_buffer = BytesIO()
    fig.savefig(pdf_buffer, format="pdf", bbox_inches="tight", pad_inches=0.25)
    pdf_buffer.seek(0)
    st.download_button(
        "Download as PDF",
        data=pdf_buffer,
        file_name=f"{team_abbr}_{year}_pitcher_stat_leaders.pdf",
        mime="application/pdf",
    )
    plt.close(fig)
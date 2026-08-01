import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
from datetime import date
import h_utils
import p_utils
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from io import BytesIO
import unicodedata


st.set_page_config(page_title="Custom Savant Page", layout="wide", page_icon="⚾")

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
    st.title("Custom Savant Page")
with meta_col:
    st.markdown(
        """
        <div class="mobile-meta" style="text-align: right; font-size: 1rem; padding-top: 0.6rem;">
            Built by <a href="https://twitter.com/Sox_Savant" target="_blank">@Sox_Savant</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

type_mode = st.radio("Type", ["Hitting", "Pitching"], horizontal=True, key="cc_mode", label_visibility="collapsed")
is_hitting = (type_mode == "Hitting")
U = h_utils if is_hitting else p_utils
prefix = "hcc" if is_hitting else "pcc"

current_year = date.today().year
last_updated = U.get_last_updated(current_year)
st.caption(f"{current_year} data last updated: {last_updated}")

STAT_ALLOWLIST = U.STAT_ALLOWLIST
STAT_PRESETS_SAVANT = U.STAT_PRESETS_SAVANT
TRUTHY_STRINGS = U.TRUTHY_STRINGS
label_map = U.label_map
lower_better = U.lower_better
start_year = U.start_year
format_stat = U.format_stat
load_final_year = U.load_final_year
STAT_DISPLAY_NAMES = U.STAT_DISPLAY_NAMES
get_team_display = U.get_team_display

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

left_col, right_col = st.columns([1, 1.3])

with left_col:
    controls_container = st.container()
    stat_builder_container = st.container()

with controls_container:
    year = st.selectbox("Select Year", list(range(current_year, start_year-1, -1)))
    player_mode = st.selectbox("Player Input", ["Name", "FanGraphs ID"], key="player_mode")
    if player_mode == "Name":
        if is_hitting:
            player_input = st.text_input("Player Name", value=st.session_state.get(f"{prefix}_player_select", "Yordan Alvarez"), key=f"{prefix}_player_select")
        else:
            player_input = st.text_input("Player Name", value=st.session_state.get(f"{prefix}_player_select", "Jacob Misiorowski"), key=f"{prefix}_player_select")
    else:
        player_input = st.text_input("Player FanGraphs ID", value=st.session_state.get("player_fg_id", ""), key="player_fg_id")


df = load_final_year(year)

if df is None or df.empty:
    st.error(f"No data found for {year}.")
    st.stop()

df["PA"] = pd.to_numeric(df.get("PA"), errors="coerce")

if "Team" in df.columns:
    df["Team"] = df["Team"].astype(str).str.strip()

from utils import get_percentile_min_pa, get_percentile_min_ip
PCT_PA = get_percentile_min_pa(year)
PCT_IP = get_percentile_min_ip(year)

if is_hitting:
    league_for_pct = df[df["PA"] >= PCT_PA].copy()
    if league_for_pct.empty:
        st.error(f"No hitters with ≥ {PCT_PA} PA in {year}.")
        st.stop()
else:
    league_for_pct = df[df["IP"] >= PCT_IP].copy()
    if league_for_pct.empty:
        st.error(f"No pitchers with ≥ {PCT_IP} IP in {year}.")
        st.stop()


player_row = None

if player_mode == "Name":
    name_input = str(player_input).strip()
    if not name_input:
        st.error("Enter a player name.")
        st.stop()
    matches = df[df["Name"].str.lower().str.strip() == name_input.lower()]
    if matches.empty:
        matches = df[df["Name"].str.lower().str.contains(name_input.lower(), na=False)]
    if matches.empty:
        st.error(f"Player '{name_input}' not found for {year}.")
        st.stop()
    player_row = matches.sort_values("PA", ascending=False).iloc[0].copy()
else:
    try:
        fg_id = int(str(player_input).strip())
    except Exception:
        st.error("Enter a valid FanGraphs ID.")
        st.stop()
    if "PlayerId" not in df.columns:
        st.error("PlayerId column not found in dataset.")
        st.stop()
    matches = df[pd.to_numeric(df["PlayerId"], errors="coerce") == fg_id]
    if matches.empty:
        st.error(f"No player found with FanGraphs ID {fg_id} in {year}.")
        st.stop()
    if is_hitting:
        player_row = matches.sort_values("PA", ascending=False).iloc[0].copy()
    else:
        player_row = matches.sort_values("IP", ascending=False).iloc[0].copy()

player_name = str(player_row.get("Name", "")).strip()
team_val = str(player_row.get("Team", "N/A")).strip()
player_team_display = get_team_display(team_val)

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

default_preset_name = "Statcast" if is_hitting else "Default"

if f"{prefix}_preset_options" not in st.session_state:
    st.session_state[f"{prefix}_preset_options"] = list(STAT_PRESETS_SAVANT.keys())



def bump_stat_config_version():
    st.session_state[f"{prefix}_stat_version_key"] = st.session_state.get(f"{prefix}_stat_version_key", 0) + 1


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
    st.session_state[f"{prefix}_manual_stat_update_key"] = True
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
    st.session_state[f"{prefix}_manual_stat_update_key"] = True
    st.session_state[reset_key] = True


def stat_preset_callback(preset_key, stat_key, available_stats):
    preset_name = st.session_state.get(preset_key, default_preset_name)
    filtered = [s for s in STAT_PRESETS_SAVANT.get(preset_name, []) if s in available_stats]
    if not filtered and available_stats:
        filtered = [available_stats[0]]
    if not filtered:
        return
    st.session_state[stat_key] = [{"Stat": s, "Show": True} for s in filtered]
    bump_stat_config_version()
    st.session_state[f"{prefix}_manual_stat_update_key"] = True
    st.session_state[f"{prefix}_add_reset_key"] = True
    st.session_state[f"{prefix}_remove_reset_key"] = True


def move_stat_row(delta, index, fallback):
    rows = normalize_stat_rows(st.session_state.get(f"{prefix}_stat_state_key", fallback), fallback)
    target = index + delta
    if 0 <= target < len(rows):
        rows[index], rows[target] = rows[target], rows[index]
        st.session_state[f"{prefix}_stat_state_key"] = rows
        bump_stat_config_version()
        st.session_state[f"{prefix}_manual_stat_update_key"] = True


def _preset_base_config():
    preset = st.session_state.get(f"{prefix}_stat_preset_key", default_preset_name)
    candidates = [s for s in STAT_PRESETS_SAVANT[preset] if s in stat_options] or [stat_options[0]]
    return [{"Stat": s, "Show": True} for s in candidates]


if f"{prefix}_stat_state_key" not in st.session_state:
    st.session_state[f"{prefix}_stat_preset_key"] = default_preset_name
    st.session_state[f"{prefix}_stat_state_key"] = _preset_base_config()
    st.session_state[f"{prefix}_stat_version_key"] = 0
elif f"{prefix}_stat_version_key" not in st.session_state:
    st.session_state[f"{prefix}_stat_version_key"] = 0

preset_base_config = _preset_base_config()
current_stat_config = normalize_stat_rows(
    st.session_state.get(f"{prefix}_stat_state_key", preset_base_config), preset_base_config
)


def compute_percentiles() -> dict[str, float]:
    result = {}
    for stat in stat_options:
        if stat not in df.columns:
            continue
        player_val = player_row.get(stat, np.nan)
        if pd.isna(player_val):
            continue
        league_vals = league_for_pct[stat].dropna()
        if league_vals.empty:
            continue
        pct = (league_vals <= float(player_val)).mean() * 100.0
        if stat in lower_better:
            pct = 100 - pct
        result[stat] = float(np.clip(pct, 0, 100))
    return result


def sort_by_percentile(ascending: bool):
    pcts = compute_percentiles()
    config = normalize_stat_rows(st.session_state.get(f"{prefix}_stat_state_key", preset_base_config), preset_base_config)
    with_pct = [(pcts[r["Stat"]], r) for r in config if r["Stat"] in pcts]
    without_pct = [r for r in config if r["Stat"] not in pcts]
    with_pct.sort(key=lambda x: x[0], reverse=not ascending)
    st.session_state[f"{prefix}_stat_state_key"] = [r for _, r in with_pct] + without_pct
    bump_stat_config_version()
    st.session_state[f"{prefix}_manual_stat_update_key"] = True
    st.rerun()


with stat_builder_container:
    preset_opts = st.session_state[f"{prefix}_preset_options"]
    prior_preset = st.session_state.get(f"{prefix}_stat_preset_key", default_preset_name)
    preset_index = preset_opts.index(prior_preset) if prior_preset in preset_opts else 0
    st.selectbox(
        "Stat Preset", preset_opts, index=preset_index,
        key=f"{prefix}_stat_preset_key", on_change=stat_preset_callback,
        args=(f"{prefix}_stat_preset_key", f"{prefix}_stat_state_key", stat_options),
    )

    sort_col1, sort_col2 = st.columns(2)
    with sort_col1:
        if st.button("Sort by percentile ↓"):
            sort_by_percentile(ascending=False)
    with sort_col2:
        if st.button("Sort by percentile ↑"):
            sort_by_percentile(ascending=True)

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

    if st.session_state.get(f"{prefix}_add_select_key") not in add_options:
        st.session_state[f"{prefix}_add_select_key"] = sentinel_add
    if st.session_state.pop(f"{prefix}_add_reset_key", False):
        st.session_state[f"{prefix}_add_select_key"] = sentinel_add
    if st.session_state.get(f"{prefix}_remove_select_key") not in remove_options:
        st.session_state[f"{prefix}_remove_select_key"] = sentinel_remove
    if st.session_state.pop(f"{prefix}_remove_reset_key", False):
        st.session_state[f"{prefix}_remove_select_key"] = sentinel_remove

    add_col, remove_col = st.columns(2)
    with add_col:
        st.selectbox(
            "Add stat", add_options, label_visibility="hidden",
            key=f"{prefix}_add_select_key", on_change=add_stat_callback,
            args=(f"{prefix}_stat_state_key", f"{prefix}_add_select_key", f"{prefix}_add_reset_key", sentinel_add),
            format_func=lambda s: STAT_DISPLAY_NAMES.get(s, s) if s != sentinel_add else s,
        )
    with remove_col:
        st.selectbox(
            "Remove stat", remove_options, label_visibility="hidden",
            key=f"{prefix}_remove_select_key", on_change=remove_stat_callback,
            args=(f"{prefix}_stat_state_key", f"{prefix}_remove_select_key", f"{prefix}_remove_reset_key", sentinel_remove),
            format_func=lambda s: STAT_DISPLAY_NAMES.get(s, s) if s != sentinel_remove else s,
        )

    current_stat_config = normalize_stat_rows(
        st.session_state.get(f"{prefix}_stat_state_key", preset_base_config), preset_base_config
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
            st.session_state[f"{prefix}_stat_state_key"] = new_config
            bump_stat_config_version()
            current_stat_config = normalize_stat_rows(st.session_state[f"{prefix}_stat_state_key"], preset_base_config)

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

    st.session_state[f"{prefix}_stat_state_key"] = normalize_stat_rows(
        st.session_state.get(f"{prefix}_stat_state_key", current_stat_config), preset_base_config
    )


stats_order = [r["Stat"] for r in st.session_state[f"{prefix}_stat_state_key"] if r.get("Show", True)]
if not stats_order:
    st.info("Add at least one stat and mark it as shown to build the chart.")
    st.stop()

leaders = []
for stat in stats_order:
    if stat not in df.columns:
        continue
    player_val = player_row.get(stat, np.nan)
    if pd.isna(player_val):
        continue
    league_vals = league_for_pct[stat].dropna()
    if league_vals.empty:
        continue
    pct = (league_vals <= float(player_val)).mean() * 100.0
    if stat in lower_better:
        pct = 100 - pct
    pct = float(np.clip(pct, 1, 100))
    leaders.append({
        "Stat": label_map.get(stat, stat),
        "Value": float(player_val),
        "Pct": pct,
    })

lead_df = pd.DataFrame(leaders)
if lead_df.empty:
    st.warning("No stats available to display.")
    st.stop()

lead_df["Display"] = lead_df.apply(lambda r: format_stat(r["Stat"], r["Value"]), axis=1)

with right_col:
    cmap = LinearSegmentedColormap.from_list(
        "savant",
        [(0, "#335AA1"), (0.5, "#E8E8E8"), (1, "#D92229")],
    )

    n = len(lead_df)
    ROW_H = 0.5
    TITLE_H = 0.55
    FOOTER_H = 0.45
    fig_height = TITLE_H + n * ROW_H + FOOTER_H

    fig, ax = plt.subplots(figsize=(7.5, fig_height))

    ax_bottom = FOOTER_H / fig_height
    ax_top    = 1 - TITLE_H / fig_height
    ax.set_position([0.08, ax_bottom, 0.8, ax_top - ax_bottom])

    title = f"{year} {player_name}"
    if player_team_display:
        title += f" | {player_team_display}"

    title_y = 1 - (TITLE_H / 2) / fig_height
    fig.text(0.5,  title_y,            title,                ha="center", va="center", fontsize=22, fontweight="bold")

    footer_y = (FOOTER_H / 2) / fig_height
    fig.text(0.13,  footer_y,           "By: Sox_Savant",    ha="center", va="center", fontsize=10, color="#555")
    fig.text(0.6, footer_y,           "Data: FanGraphs • Baseball Reference • Baseball Savant", ha="center", va="center", fontsize=10, color="#555")

    y = np.arange(n)
    TRACK_H    = 0.82
    BAR_H      = 0.82
    LEFT_OFFSET = 3
    BAR_LENGTH  = 60
    VALUE_X     = LEFT_OFFSET + BAR_LENGTH + 15
    BUBBLE_SIZE = 1000

    ax.barh(y, BAR_LENGTH, left=LEFT_OFFSET, height=TRACK_H, color="#F1F1F1", edgecolor="none")

    for i, row in lead_df.iterrows():
        pct     = row["Pct"]
        color   = cmap(pct / 100)
        bar_width = pct / 100 * BAR_LENGTH
        bubble_x  = LEFT_OFFSET + bar_width
        BUBBLE_R  = 4
        visual_bar_width = np.clip(max(bar_width, BUBBLE_R), 0, BAR_LENGTH - (BUBBLE_R)/2)
        ax.barh(i, visual_bar_width, left=LEFT_OFFSET, height=BAR_H, color=color, edgecolor="none")
        bubble_x  = np.clip(bubble_x, LEFT_OFFSET + BUBBLE_R, LEFT_OFFSET + BAR_LENGTH)
        ax.scatter(bubble_x, i, s=BUBBLE_SIZE, color=color, edgecolors="white", linewidth=2.4, zorder=3)
        ax.text(bubble_x, i + 0.04, f"{int(round(pct))}", ha="center", va="center",
                fontsize=12, fontweight="bold", color="white")
        ax.text(VALUE_X - 10.5, i, row["Display"], ha="left",  va="center", fontsize=12, color="#111")
        ax.text(0,           i, row["Stat"],    ha="right", va="center", fontsize=13)

    for pos in (0.1, 0.5, 0.9):
        ax.vlines(LEFT_OFFSET + BAR_LENGTH * pos, -0.5, n - 0.5,
                  colors="white", linewidth=1.2, alpha=0.25, zorder=2.6)

    ax.set_xlim(-15, VALUE_X + 5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.invert_yaxis()
    ax.axis("off")

    st.pyplot(fig, width='stretch', clear_figure=False)

    pdf_buffer = BytesIO()
    fig.savefig(pdf_buffer, format="pdf", bbox_inches="tight", pad_inches=0.25, dpi=300)
    pdf_buffer.seek(0)
    st.download_button(
        "Download as PDF",
        data=pdf_buffer,
        file_name=f"{player_name.replace(' ', '_')}_{year}_savant.pdf",
        mime="application/pdf",
    )
    plt.close(fig)

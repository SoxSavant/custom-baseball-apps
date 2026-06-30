import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Career Trajectory", layout="wide", page_icon="⚾")

st.markdown("""
<style>
.block-container { padding-top: 2rem !important; padding-bottom: 1rem !important; }
[data-testid="stToolbar"]      { visibility: hidden; }
[data-testid="stDecoration"]   { display: none; }
[data-testid="stStatusWidget"] { display: none; }
.viewerBadge_link__qRi_k       { display: none; }
@media only screen and (max-width: 600px) {
    [data-testid="stAppViewContainer"] h1 { font-size: 1.8rem !important; }
    .mobile-meta { font-size: 0.8rem !important; padding-top: 0.3rem !important; }
}
</style>
""", unsafe_allow_html=True)

# ── constants ──────────────────────────────────────────────────────────────────
current_year = date.today().year

PCT_STATS_100 = {
    "K%", "BB%", "Chase%", "Whiff%", "Swing%", "Z-Swing%",
    "O-Contact%", "Z-Contact%", "Zone%", "Barrel%", "HardHit%",
    "Sweet-Spot%", "Squared-Up%", "Z-Swing% - Chase%",
}

ZERO_STATS = {
    "DRS", "OAA", "FRV", "BRV", "WAR", "fWAR", "bWAR",
    "oWAR", "dWAR", "Off", "Def", "BsR", "UZR", "UZR/150",
}

LINE_COLORS = ["#7EB8F7", "#F28B50", "#72D195", "#C97DD4"]

HITTING_DEFAULTS = ["wRC+", "xwOBA"]
PITCHING_DEFAULTS = ["ERA", "FIP"]

# ── header ─────────────────────────────────────────────────────────────────────
title_col, meta_col = st.columns([3, 1])
with title_col:
    st.title("Career Trajectory")
with meta_col:
    st.markdown(
        '<div class="mobile-meta" style="text-align:right; font-size:1rem; padding-top:0.6rem;">'
        'Built by <a href="https://twitter.com/Sox_Savant" target="_blank">@Sox_Savant</a>'
        "</div>",
        unsafe_allow_html=True,
    )

# ── session state defaults ─────────────────────────────────────────────────────
for key, default in [
    ("ct_domain",     "Hitting"),
    ("ct_start_year", 2018),
    ("ct_end_year",   current_year),
    ("ct_stats",      HITTING_DEFAULTS),
    ("ct_player_1",   "Shohei Ohtani"),
    ("ct_player_2",   ""),
    ("ct_player_3",   ""),
    ("ct_player_4",   ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── domain toggle + conditional imports ───────────────────────────────────────
col_left, col_right = st.columns([1, 1.8])

with col_left:
    domain = st.radio(
        "", ["Hitting", "Pitching"],
        key="ct_domain", horizontal=True,
    )

    if "ct_last_domain" not in st.session_state:
        st.session_state["ct_last_domain"] = domain
    if domain != st.session_state["ct_last_domain"]:
        st.session_state["ct_stats"] = HITTING_DEFAULTS if domain == "Hitting" else PITCHING_DEFAULTS
        st.session_state["ct_last_domain"] = domain

if domain == "Hitting":
    from h_utils import (
        STAT_ALLOWLIST, format_stat, start_year,
        label_map, load_final_year,
        aggregate_player_group,
        STAT_ROUND, get_last_updated,
        resolve_player_id,
        STAT_DISPLAY_NAMES,
    )
else:
    from p_utils import (
        STAT_ALLOWLIST, format_stat, start_year,
        label_map, load_final_year,
        aggregate_player_group,
        STAT_ROUND, get_last_updated,
        resolve_player_id,
        STAT_DISPLAY_NAMES,
    )

ALL_YEARS = list(range(current_year, start_year - 1, -1))

last_updated = get_last_updated(current_year)
st.caption(f"{current_year} data last updated: {last_updated}")

# ── data helpers ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_year_cached(year: int, dom: str) -> pd.DataFrame | None:
    return load_final_year(year)


def get_player_career(player_id: int, yr_start: int, yr_end: int) -> pd.DataFrame:
    seasons = []
    for yr in range(yr_start, yr_end + 1):
        df = load_year_cached(yr, domain)
        if df is None or df.empty:
            continue
        match = df[df["PlayerId"] == player_id].copy()
        if match.empty:
            continue
        if len(match) > 1:
            agg = aggregate_player_group(match)
            agg["Season"] = yr
            seasons.append(agg)
        else:
            match["Season"] = yr
            seasons.append(match)
    if not seasons:
        return pd.DataFrame()
    return pd.concat(seasons, ignore_index=True).sort_values("Season")

# ── shared chart styling ───────────────────────────────────────────────────────
LIGHT_LAYOUT = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font=dict(color="#000000"),
    hoverlabel=dict(bgcolor="#f0f0f0", font_color="#000000"),
)

AXIS_STYLE = dict(
    gridcolor="#e6e6e6",
    zerolinecolor="#e6e6e6",
    linecolor="#000000",
    showgrid=True,
    zeroline=False,
)


def yaxis_fmt(stat: str) -> dict:
    decimals = STAT_ROUND.get(stat, 1)
    if stat in PCT_STATS_100:
        return dict(tickformat=f".{decimals}f", ticksuffix="%")
    return dict(tickformat=f".{decimals}f", ticksuffix="")


def apply_xaxis_seasons(fig, all_seasons, n_rows):
    fig.update_xaxes(
        tickmode="array",
        tickvals=all_seasons,
        ticktext=[str(int(s)) for s in all_seasons],
        tickangle=-45,
        tickfont=dict(size=10, color="#000000"),
        **AXIS_STYLE,
    )
    bottom_key = "xaxis" if n_rows == 1 else f"xaxis{n_rows}"
    fig.update_layout(**{bottom_key: dict(title=dict(text="Season", font=dict(size=13, color="#000000")))})


# ── chart: up to 4 players × up to 4 stats ────────────────────────────────────
def build_chart(careers: list[tuple[str, pd.DataFrame]], stats: list[str]) -> go.Figure:
    n = len(stats)
    v_spacing = 0.0 if n == 1 else (0.15 if n == 2 else 0.07)

    fig = make_subplots(
        rows=n, cols=1,
        shared_xaxes=True,
        vertical_spacing=v_spacing,
        subplot_titles=[label_map.get(s, s) for s in stats],
    )

    all_seasons = set()
    for _, career in careers:
        all_seasons.update(career["Season"].dropna().tolist())
    all_seasons = sorted(all_seasons)

    for i, stat in enumerate(stats, start=1):
        stat_label = label_map.get(stat, stat)

        if stat in ZERO_STATS:
            fig.add_hline(y=0, line_dash="dot", line_color="rgba(0,0,0,0.25)", line_width=1, row=i, col=1)

        for j, (player_name, career) in enumerate(careers):
            color = LINE_COLORS[j]

            if stat not in career.columns:
                continue

            y_raw = pd.to_numeric(career[stat], errors="coerce")
            hover_text = [format_stat(stat, v) if pd.notna(v) else "N/A" for v in y_raw]

            fig.add_trace(
                go.Scatter(
                    x=career["Season"].tolist(),
                    y=y_raw,
                    mode="lines+markers",
                    name=player_name,
                    # only show each player in the legend once (first stat panel)
                    showlegend=(i == 1),
                    legendgroup=player_name,
                    line=dict(color=color, width=2.5),
                    marker=dict(size=7, color=color),
                    customdata=list(zip(career["Season"].tolist(), hover_text)),
                    hovertemplate=(
                        f"<b>{player_name}</b><br>"
                        f"{stat_label}: %{{customdata[1]}}<br>"
                        "Season: %{customdata[0]}<extra></extra>"
                    ),
                    connectgaps=False,
                ),
                row=i, col=1,
            )

        axis_key = "yaxis" if i == 1 else f"yaxis{i}"
        fig.update_layout(**{
            axis_key: dict(
                title=dict(text=stat_label, font=dict(size=13, color="#000000")),
                tickfont=dict(size=10, color="#000000"),
                **AXIS_STYLE,
                **yaxis_fmt(stat),
            )
        })

    apply_xaxis_seasons(fig, all_seasons, n)

    # legend only needed when more than one player
    show_legend = len(careers) > 1
    fig.update_layout(
        height=max(300, 280 * n),
        showlegend=show_legend,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.07,
            xanchor="left", x=0,
            font=dict(size=12, color="#000000"),
        ),
        margin=dict(l=65, r=30, t=50 if not show_legend else 50, b=55),
        **LIGHT_LAYOUT,
    )
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)
    for ann in fig.layout.annotations:
        ann.font.size = 13
        ann.font.color = "#000000"

    return fig


# ── left panel controls ────────────────────────────────────────────────────────
with col_left:
    yr_col1, yr_col2 = st.columns(2)
    with yr_col1:
        sel_start = st.selectbox(
            "Start Year", options=ALL_YEARS,
            key="ct_start_year",
        )
    with yr_col2:
        sel_end = st.selectbox(
            "End Year", options=ALL_YEARS,
            index=ALL_YEARS.index(st.session_state["ct_end_year"])
                  if st.session_state["ct_end_year"] in ALL_YEARS else 0,
            key="ct_end_year",
        )

    yr_start = min(sel_start, sel_end)
    yr_end   = max(sel_start, sel_end)

    st.markdown("**Players**")
    p1 = st.text_input("Player 1", key="ct_player_1")
    p2 = st.text_input("Player 2", key="ct_player_2")
    p3 = st.text_input("Player 3", key="ct_player_3")
    p4 = st.text_input("Player 4", key="ct_player_4")
    player_names_input = [p1, p2, p3, p4]

    st.markdown("**Stats**")
    selected_stats = st.multiselect(
        "Stats",
        options=STAT_ALLOWLIST,
        default=[s for s in st.session_state["ct_stats"] if s in STAT_ALLOWLIST],
        max_selections=4,
        label_visibility="collapsed",
        key="ct_stats_select",
        format_func=lambda s: label_map.get(s, s),
    )
    st.session_state["ct_stats"] = selected_stats

# ── main content ───────────────────────────────────────────────────────────────
with col_right:
    filled_names = [n for n in player_names_input if n.strip()]

    if not filled_names:
        st.info("Enter at least one player name to get started.")
    elif not selected_stats:
        st.warning("Select at least one stat.")
    else:
        careers = []
        for name in player_names_input:
            if not name.strip():
                continue
            with st.spinner(f"Loading {name}…"):
                pid = resolve_player_id(name.strip(), yr_start, yr_end)
            if not pid:
                st.warning(f"Could not find **{name}** — skipping.")
                continue
            career = get_player_career(pid, yr_start, yr_end)
            if career.empty:
                st.warning(f"No data for **{name}** in {yr_start}–{yr_end} — skipping.")
                continue
            display_name = str(career["Name"].iloc[-1]).strip() if "Name" in career.columns else name
            careers.append((display_name, career))

        if not careers:
            st.error("No data found for any of the entered players.")
        else:
            all_seasons = set()
            for _, c in careers:
                all_seasons.update(c["Season"].dropna().tolist())
            yr_min, yr_max = int(min(all_seasons)), int(max(all_seasons))
            player_labels = ", ".join(name for name, _ in careers)
            st.caption(f"{player_labels} · {yr_min}–{yr_max}")

            fig = build_chart(careers, selected_stats)
            st.plotly_chart(fig, width='stretch')

            with st.expander("Season data"):
                for player_name, career in careers:
                    st.markdown(f"**{player_name}**")
                    display_cols = ["Season"] + [s for s in selected_stats if s in career.columns]
                    tbl = career[display_cols].copy().set_index("Season")
                    col_config = {}
                    for s in selected_stats:
                        if s not in career.columns:
                            continue
                        lbl = STAT_DISPLAY_NAMES.get(s, label_map.get(s, s))
                        decimals = STAT_ROUND.get(s, 1)
                        col_config[s] = st.column_config.NumberColumn(lbl, format=f"%.{decimals}f")
                        tbl = tbl.rename(columns={s: lbl})
                    st.dataframe(tbl, width="stretch", column_config=col_config)
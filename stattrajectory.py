import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Stat Trajectory", layout="wide", page_icon="⚾")

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

LINE_COLORS = ["#7EB8F7", "#F28B50", "#72D195", "#C97DD4", "#BD0F0F", "#DDD31E"]

HITTING_DEFAULTS = ["fWAR",]
PITCHING_DEFAULTS = ["ERA"]

# ── header ─────────────────────────────────────────────────────────────────────
title_col, meta_col = st.columns([3, 1])
with title_col:
    st.title("Stat Trajectory")
with meta_col:
    st.markdown(
        '<div class="mobile-meta" style="text-align:right; font-size:1rem; padding-top:0.6rem;">'
        'Built by <a href="https://twitter.com/Sox_Savant" target="_blank">@Sox_Savant</a>'
        "</div>",
        unsafe_allow_html=True,
    )

from h_utils import get_last_updated
current_year = date.today().year
h_last_updated = get_last_updated(current_year)
st.caption(f"{current_year} data last updated: {h_last_updated}")

# ── session state defaults ─────────────────────────────────────────────────────
for key, default in [
    ("ct_domain",     "Hitting"),
    ("ct_start_year", 2018),
    ("ct_end_year",   current_year),
    ("ct_stats",      HITTING_DEFAULTS),
    ("ct_mode",       "Cumulative"),
    ("ct_player_1",   "Shohei Ohtani"),
    ("ct_player_2",   ""),
    ("ct_player_3",   ""),
    ("ct_player_4",   ""),
    ("ct_player_5",   ""),
    ("ct_player_6",   ""),
    ("ct_num_players", 1),
    ("ct_combine_ohtani_war", True),
    ("ct_year_mode",    "Calendar Years"),
    ("ct_career_start", 1),
    ("ct_career_end",   10),
    ("ct_pending_clear_slot", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Apply any pending "clear this player slot" request BEFORE the text_input
# widgets below are instantiated. Streamlit forbids writing to a widget's
# session_state key once that widget has already been created in the same
# run, so the clear has to happen here, on the run right after the button
# click triggered a rerun.
if st.session_state["ct_pending_clear_slot"] is not None:
    slot = st.session_state.pop("ct_pending_clear_slot")
    st.session_state[f"ct_player_{slot}"] = ""

# ── domain toggle + conditional imports ───────────────────────────────────────
col_left, col_right = st.columns([1, 2])

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


# ── data helpers ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_year_cached(year: int, dom: str) -> pd.DataFrame | None:
    return load_final_year(year)


@st.cache_data(show_spinner=False)
def find_debut_year(player_id: int, dom: str) -> int | None:
    """First year (searching from the earliest available data) the player appears."""
    for yr in range(start_year, current_year + 1):
        df = load_year_cached(yr, dom)
        if df is None or df.empty:
            continue
        if player_id in df["PlayerId"].values:
            return yr
    return None


def get_player_career(player_id: int, yr_start: int, yr_end: int, mode: str) -> pd.DataFrame:
    seasons = []
    for yr in range(yr_start, yr_end + 1):
        df = load_year_cached(yr, domain)
        if df is None or df.empty:
            continue
        match = df[df["PlayerId"] == player_id].copy()
        if match.empty:
            continue
        match["Season"] = yr
        seasons.append(match)

    if not seasons:
        return pd.DataFrame()

    if mode == "Split Season":
        result = []
        for match in seasons:
            yr = match["Season"].iloc[0]
            if len(match) > 1:
                agg = aggregate_player_group(match)
                agg["Season"] = yr
                result.append(agg)
            else:
                result.append(match)
        return pd.concat(result, ignore_index=True).sort_values("Season")

    # Multi-Year Span: each point = rolling aggregate from yr_start through that year
    result = []
    accumulated = []
    for match in seasons:
        yr = match["Season"].iloc[0]
        accumulated.append(match)
        combined = pd.concat(accumulated, ignore_index=True)
        agg = aggregate_player_group(combined)
        agg["Season"] = yr
        result.append(agg)
    return pd.concat(result, ignore_index=True).sort_values("Season")

def _is_ohtani(name: str) -> bool:
    return "ohtani" in name.strip().lower()


@st.cache_data(show_spinner=False)
def get_ohtani_combined_war(yr_start: int, yr_end: int) -> dict:
    import h_utils as hu
    import p_utils as pu

    h_pid = hu.resolve_player_id("Shohei Ohtani", yr_start, yr_end)
    p_pid = pu.resolve_player_id("Shohei Ohtani", yr_start, yr_end)

    per_season = {}
    for yr in range(yr_start, yr_end + 1):
        h_fwar = h_bwar = p_fwar = p_bwar = 0.0

        if h_pid:
            h_df = hu.load_final_year(yr)
            if h_df is not None and not h_df.empty:
                rows = h_df[h_df["PlayerId"] == h_pid]
                if not rows.empty:
                    if "fWAR" in rows.columns:
                        h_fwar = float(pd.to_numeric(rows["fWAR"], errors="coerce").fillna(0).sum())
                    if "bWAR" in rows.columns:
                        h_bwar = float(pd.to_numeric(rows["bWAR"], errors="coerce").fillna(0).sum())

        if p_pid:
            p_df = pu.load_final_year(yr)
            if p_df is not None and not p_df.empty:
                rows = p_df[p_df["PlayerId"] == p_pid]
                if not rows.empty:
                    if "fWAR" in rows.columns:
                        p_fwar = float(pd.to_numeric(rows["fWAR"], errors="coerce").fillna(0).sum())
                    if "bWAR" in rows.columns:
                        p_bwar = float(pd.to_numeric(rows["bWAR"], errors="coerce").fillna(0).sum())

        total_fwar = h_fwar + p_fwar
        total_bwar = h_bwar + p_bwar
        per_season[yr] = {
            "fWAR":         total_fwar,
            "bWAR":         total_bwar,
            "fWAR-bWAR AVG": (total_fwar + total_bwar) / 2,
        }

    return per_season


def apply_combined_ohtani_war(career: pd.DataFrame, yr_start: int, yr_end: int, mode: str) -> pd.DataFrame:
    career = career.copy()
    per_season = get_ohtani_combined_war(yr_start, yr_end)

    if mode == "Cumulative":
        cum_fwar = cum_bwar = 0.0
        source = {}
        for yr in sorted(per_season):
            cum_fwar += per_season[yr]["fWAR"]
            cum_bwar += per_season[yr]["bWAR"]
            source[yr] = {
                "fWAR":          cum_fwar,
                "bWAR":          cum_bwar,
                "fWAR-bWAR AVG": (cum_fwar + cum_bwar) / 2,
            }
    else:
        source = per_season

    for col in ("fWAR", "bWAR", "fWAR-bWAR AVG"):
        if col in career.columns:
            career[col] = career["Season"].map(
                lambda s, c=col: source.get(int(s), {}).get(c, np.nan)
            )

    return career

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


def apply_xaxis(fig, all_x_vals, n_rows, x_title):
    fig.update_xaxes(
        tickmode="array",
        tickvals=all_x_vals,
        ticktext=[str(int(s)) for s in all_x_vals],
        tickangle=0,
        tickfont=dict(size=11, color="#000000"),
        **AXIS_STYLE,
    )
    bottom_key = "xaxis" if n_rows == 1 else f"xaxis{n_rows}"
    fig.update_layout(**{bottom_key: dict(title=dict(text=x_title, font=dict(size=14, color="#000000")))})


# ── chart: up to 4 players × up to 4 stats ────────────────────────────────────
def build_chart(
    careers: list[tuple[str, pd.DataFrame]],
    stats: list[str],
    x_col: str = "Season",
    x_title: str = "Season",
    mode: str = "Cumulative",
) -> go.Figure:
    n = len(stats)
    v_spacing = 0.12

    subplot_titles = []
    for stat in stats:
        stat_label = label_map.get(stat, stat)
        if mode == "Cumulative":
            subplot_titles.append(f"{stat_label}")
        else:
            subplot_titles.append(f"{stat_label} by Year")

    fig = make_subplots(
        rows=n, cols=1,
        shared_xaxes=True,
        vertical_spacing=v_spacing,
        subplot_titles=subplot_titles,
    )
    fig.update_annotations(font=dict(size=15, color="#000000"))

    all_x_vals = set()
    for _, career in careers:
        all_x_vals.update(career[x_col].dropna().tolist())
    all_x_vals = sorted(all_x_vals)

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
                    x=career[x_col].tolist(),
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
                title=dict(text=stat_label, font=dict(size=16, color="#000000")),
                tickfont=dict(size=12, color="#000000"),
                **AXIS_STYLE,
                **yaxis_fmt(stat),
            )
        })

    apply_xaxis(fig, all_x_vals, n, x_title)

    # legend only needed when more than one player
    show_legend = True
    fig.update_layout(
        height=700,
        showlegend=show_legend,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.05,
            xanchor="left", x=0,
            font=dict(size=13, color="#000000"),
        ),
        margin=dict(l=65, r=30, t=50, b=55),
        **LIGHT_LAYOUT,
    )
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)

    return fig


# ── left panel controls ────────────────────────────────────────────────────────
with col_left:
    year_mode = st.radio(
        "Compare by", ["Calendar Years", "Career Years"],
        key="ct_year_mode", horizontal=True,
    )

    if year_mode == "Calendar Years":
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
        career_start = career_end = None
    else:
        cy_col1, cy_col2 = st.columns(2)
        with cy_col1:
            career_start = st.number_input(
                "From Career Year", min_value=1, max_value=25,
                value=st.session_state.get("ct_career_start", 1),
                key="ct_career_start", step=1,
            )
        with cy_col2:
            career_end = st.number_input(
                "To Career Year", min_value=1, max_value=25,
                value=st.session_state.get("ct_career_end", 10),
                key="ct_career_end", step=1,
            )
        career_start, career_end = min(career_start, career_end), max(career_start, career_end)
        # Placeholder bounds — each player's actual window is resolved
        # individually below based on their own debut year.
        yr_start, yr_end = start_year, current_year

    mode = st.radio(
        "Mode", ["Split Season", "Cumulative"],
        key="ct_mode", horizontal=True,
    )

    st.markdown("**Players**")
    num_players = st.session_state["ct_num_players"]

    player_names_input = []
    for i in range(1, 7):
        p_key = f"ct_player_{i}"
        if i <= num_players:
            player_names_input.append(st.text_input(f"Player {i}", key=p_key))
        else:
            # not rendered this run, but keep whatever value it last held
            player_names_input.append(st.session_state.get(p_key, ""))

    add_col, remove_col = st.columns(2)
    with add_col:
        if num_players < 6:
            if st.button("Add Player", key="ct_add_player", width="stretch"):
                st.session_state["ct_num_players"] = num_players + 1
                st.rerun()
    with remove_col:
        if num_players > 1:
            if st.button("Remove Player", key="ct_remove_player", width="stretch"):
                st.session_state["ct_pending_clear_slot"] = num_players
                st.session_state["ct_num_players"] = num_players - 1
                st.rerun()

    ohtani_in_inputs = any(_is_ohtani(n) for n in player_names_input if n.strip())
    if ohtani_in_inputs:
        st.checkbox("Combine Ohtani's WAR", key="ct_combine_ohtani_war")

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

            if year_mode == "Calendar Years":
                with st.spinner(f"Loading {name}…"):
                    pid = resolve_player_id(name.strip(), yr_start, yr_end)
                if not pid:
                    st.warning(f"Could not find **{name}** — skipping.")
                    continue
                p_yr_start, p_yr_end = yr_start, yr_end
                debut_year = None
            else:
                with st.spinner(f"Loading {name}…"):
                    pid = resolve_player_id(name.strip(), start_year, current_year)
                if not pid:
                    st.warning(f"Could not find **{name}** — skipping.")
                    continue
                debut_year = find_debut_year(pid, domain)
                if debut_year is None:
                    st.warning(f"Could not determine a debut year for **{name}** — skipping.")
                    continue
                p_yr_start = debut_year + career_start - 1
                if p_yr_start > current_year:
                    st.warning(f"**{name}** doesn't have data for Career Year {career_start} yet — skipping.")
                    continue
                p_yr_end = min(debut_year + career_end - 1, current_year)

            career = get_player_career(pid, p_yr_start, p_yr_end, st.session_state["ct_mode"])
            if career.empty:
                st.warning(f"No data for **{name}** in the selected range — skipping.")
                continue
            display_name = str(career["Name"].iloc[-1]).strip() if "Name" in career.columns else name
            if st.session_state.get("ct_combine_ohtani_war") and _is_ohtani(name):
                career = apply_combined_ohtani_war(career, p_yr_start, p_yr_end, st.session_state["ct_mode"])

            if year_mode == "Career Years":
                career["CareerYear"] = career["Season"] - debut_year + 1

            careers.append((display_name, career))

        if not careers:
            st.error("No data found for any of the entered players.")
        else:
            all_seasons = set()
            for _, c in careers:
                all_seasons.update(c["Season"].dropna().tolist())
            yr_min, yr_max = int(min(all_seasons)), int(max(all_seasons))
            player_labels = ", ".join(name for name, _ in careers)

            x_col = "CareerYear" if year_mode == "Career Years" else "Season"
            x_title = "Career Year" if year_mode == "Career Years" else "Season"

            fig = build_chart(
                careers, selected_stats,
                x_col=x_col, x_title=x_title,
                mode=st.session_state["ct_mode"],
            )
            st.plotly_chart(fig, width='stretch')

            with st.expander("Season data"):
                for player_name, career in careers:
                    st.markdown(f"**{player_name}**")
                    index_col = "CareerYear" if year_mode == "Career Years" else "Season"
                    extra_cols = ["Season"] if year_mode == "Career Years" else []
                    display_cols = [index_col] + extra_cols + [s for s in selected_stats if s in career.columns]
                    tbl = career[display_cols].copy().set_index(index_col)
                    tbl.index.name = "Career Years" if year_mode == "Career Years" else "Season"
                    col_config = {}
                    for s in selected_stats:
                        if s not in career.columns:
                            continue
                        lbl = STAT_DISPLAY_NAMES.get(s, label_map.get(s, s))
                        decimals = STAT_ROUND.get(s, 1)
                        col_config[s] = st.column_config.NumberColumn(lbl, format=f"%.{decimals}f")
                        tbl = tbl.rename(columns={s: lbl})
                    st.dataframe(tbl, width="stretch", column_config=col_config)
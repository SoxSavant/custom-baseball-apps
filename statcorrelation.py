import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date
from io import BytesIO

from utils import get_dynamic_min_pa, get_dynamic_min_ip, TEAM_OPTIONS, LEAGUES

st.set_page_config(page_title="Stat Correlation", layout="wide", page_icon="⚾")

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
    st.title("Stat Correlation")
with meta_col:
    st.markdown(
        """
        <div class="mobile-meta" style="text-align: right; font-size: 1rem; padding-top: 0.6rem;">
            Built by <a href="https://twitter.com/Sox_Savant" target="_blank">@Sox_Savant</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

from h_utils import get_last_updated
current_year = date.today().year
h_last_updated = get_last_updated(current_year)
st.caption(f"{current_year} data last updated: {h_last_updated}")


MODE_SINGLE = "Single Season"
MODE_SPLIT  = "Split Seasons"
MODE_MULTI  = "Multi-Year Span"

for key, default in [
    ("cr_domain", "Hitting"),
    ("cr_year", current_year),
    ("cr_start_year", current_year - 1),
    ("cr_end_year", current_year),
    ("cr_mode", MODE_SINGLE),
    ("cr_min_type", "PA"),
    ("cr_min_pa", get_dynamic_min_pa(current_year)/2),
    ("cr_min_ip", 0),
    ("cr_show_names", True),
    ("cr_search", ""),
    ("cr_position", "all"),
    ("cr_team", "all"),
    ("cr_league", "All"),
    ("cr_highlight_team", "all"),  
]:
    if key not in st.session_state:
        st.session_state[key] = default

col1, col2 = st.columns([.5, 2])

with col1:
    domain = st.radio("", ["Hitting", "Pitching", "Combined"], key="cr_domain", horizontal=True)

    if domain == "Hitting":
        from h_utils import (
            STAT_ALLOWLIST, format_stat, start_year, label_map,
            load_final_year,  lower_better, aggregate_player_group,
            POSITION_OPTIONS, normalize_team, filter_by_position,
        )
        min_type_default = "PA"

    elif domain == "Pitching":
        from p_utils import (
            STAT_ALLOWLIST, format_stat, start_year, label_map,
            load_final_year, lower_better, aggregate_player_group,
            normalize_team,
        )
        min_type_default = "IP"

    else:  # Combined
        from h_utils import load_final_year as load_hitting_year, normalize_team, start_year

        from p_utils import load_final_year as load_pitching_year

        min_type_default = "PA/IP"

        STAT_ALLOWLIST = ["fWAR", "bWAR"]
        label_map = {"fWAR": "fWAR", "bWAR": "bWAR"}
        lower_better = set()

        def format_stat(stat, val):
            if pd.isna(val):
                return "—"
            return f"{val:.1f}"

        _COMBINED_SLIM_COLS = ["PlayerId", "Name", "Team", "MLBAMID", "fWAR", "bWAR", "PA", "IP"]

        def _combined_slim(df, suffix):
            out = pd.DataFrame(columns=_COMBINED_SLIM_COLS)
            if df is not None and not df.empty:
                present = [c for c in _COMBINED_SLIM_COLS if c in df.columns]
                out = df[present].copy()
                for c in _COMBINED_SLIM_COLS:
                    if c not in out.columns:
                        out[c] = np.nan
                out["PlayerId"] = pd.to_numeric(out["PlayerId"], errors="coerce").astype("Int64").astype(str)
            return out.rename(columns={
                "Name": f"Name_{suffix}",
                "Team": f"Team_{suffix}",
                "MLBAMID": f"MLBAMID_{suffix}",
                "fWAR": f"fWAR_{suffix}",
                "bWAR": f"bWAR_{suffix}",
                "PA": f"PA_{suffix}",
                "IP": f"IP_{suffix}",
            })

        def load_final_year(year):
            hit_slim = _combined_slim(load_hitting_year(year), "h")
            pit_slim = _combined_slim(load_pitching_year(year), "p")

            merged = hit_slim.merge(pit_slim, on="PlayerId", how="outer")

            merged["Name"] = merged["Name_h"].combine_first(merged["Name_p"])
            merged["Team"] = merged["Team_h"].combine_first(merged["Team_p"])
            merged["MLBAMID"] = merged["MLBAMID_h"].combine_first(merged["MLBAMID_p"])

            merged["fWAR"] = pd.to_numeric(merged["fWAR_h"], errors="coerce").fillna(0) + \
                              pd.to_numeric(merged["fWAR_p"], errors="coerce").fillna(0)
            merged["bWAR"] = pd.to_numeric(merged["bWAR_h"], errors="coerce").fillna(0) + \
                              pd.to_numeric(merged["bWAR_p"], errors="coerce").fillna(0)
            # PA only ever comes from the hitting side, IP only from the pitching side;
            # summing is safe (and covers two-way players) since the other side is 0/NaN.
            merged["PA"] = pd.to_numeric(merged["PA_h"], errors="coerce").fillna(0) + \
                            pd.to_numeric(merged["PA_p"], errors="coerce").fillna(0)
            merged["IP"] = pd.to_numeric(merged["IP_h"], errors="coerce").fillna(0) + \
                            pd.to_numeric(merged["IP_p"], errors="coerce").fillna(0)

            merged = merged.drop(columns=[
                "Name_h", "Name_p", "Team_h", "Team_p",
                "MLBAMID_h", "MLBAMID_p", "fWAR_h", "fWAR_p", "bWAR_h", "bWAR_p",
                "PA_h", "PA_p", "IP_h", "IP_p",
            ])
            merged = merged[merged["PlayerId"].notna() & (merged["PlayerId"] != "<NA>")]
            merged = merged[merged["Name"].notna()]
            return merged

        def aggregate_player_group(df):
            df = df.copy()
            for col in ["fWAR", "bWAR", "PA", "IP"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

            last = df.sort_values("Season").groupby("PlayerId", as_index=False).last()[
                ["PlayerId", "Name", "MLBAMID"]
            ]

            team_info = (
                df.groupby("PlayerId")["Team"]
                .apply(lambda teams: (
                    "2+ Teams"
                    if len({normalize_team(t) for t in teams if pd.notna(t) and str(t).strip() not in ("", "- - -")}) > 1
                    else (normalize_team(str(teams.dropna().iloc[0])) if teams.dropna().shape[0] else "N/A")
                ))
                .reset_index()
            )

            summed = df.groupby("PlayerId", as_index=False)[["fWAR", "bWAR", "PA", "IP"]].sum()

            result = summed.merge(last, on="PlayerId", how="left").merge(team_info, on="PlayerId", how="left")
            return result

    if "cr_last_domain" not in st.session_state:
        st.session_state.cr_last_domain = domain
    if domain != st.session_state.cr_last_domain:
        st.session_state["cr_min_type"] = min_type_default
        st.session_state["cr_min_pa"] = get_dynamic_min_pa(current_year)/2
        st.session_state["cr_min_ip"] = get_dynamic_min_ip(current_year)/2
        st.session_state.cr_last_domain = domain

    mode = st.radio("Mode", options=[MODE_SINGLE, MODE_SPLIT, MODE_MULTI], key="cr_mode")

    if mode == MODE_SINGLE:
        st.selectbox("Year", options=list(range(current_year, start_year - 1, -1)), key="cr_year")
        sel_start = st.session_state["cr_year"]
        sel_end = st.session_state["cr_year"]
    else:
        st.selectbox("Start Year", options=list(range(current_year, start_year - 1, -1)), key="cr_start_year")
        st.selectbox("End Year",   options=list(range(current_year, start_year - 1, -1)), key="cr_end_year")
        sel_start = st.session_state["cr_start_year"]
        sel_end   = max(st.session_state["cr_end_year"], sel_start)

    x_stat = st.selectbox(
        "X Stat", STAT_ALLOWLIST, key="cr_x_stat",
        format_func=lambda x: label_map.get(x, x),
        index=0,
    )
    y_stat = st.selectbox(
        "Y Stat", STAT_ALLOWLIST, key="cr_y_stat",
        format_func=lambda x: label_map.get(x, x),
        index=min(1, len(STAT_ALLOWLIST) - 1),
    )

    if domain == "Hitting":
        st.number_input("Min PA", min_value=0, max_value=20000, key="cr_min_pa")
    elif domain == "Pitching":
        st.number_input("Min IP", min_value=0, max_value=5000, key="cr_min_ip")
    else:
        min_c1, min_c2 = st.columns(2)
        with min_c1:
            st.number_input("Min PA", min_value=0, max_value=20000, key="cr_min_pa")
        with min_c2:
            st.number_input("Min IP", min_value=0, max_value=5000, key="cr_min_ip")


    team_disabled = (mode == MODE_MULTI)
    league_disabled = (sel_start < 2013)


    st.checkbox("Show player names", key="cr_show_names")
    st.text_input("Search players (separate by commas)", key="cr_search")
    st.selectbox(
        "Highlight Team",
        options=list(TEAM_OPTIONS.keys()),
        format_func=lambda x: TEAM_OPTIONS[x],
        key="cr_highlight_team",
        disabled=team_disabled,
        help="Team highlight unavailable for multi-year span" if team_disabled else None,
    )
    if domain == "Hitting":
        st.selectbox(
            "Position",
            options=list(POSITION_OPTIONS.keys()),
            format_func=lambda x: POSITION_OPTIONS[x],
            key="cr_position",
        )
    st.selectbox(
        "Team",
        options=list(TEAM_OPTIONS.keys()),
        format_func=lambda x: TEAM_OPTIONS[x],
        key="cr_team",
        disabled=team_disabled,
        help="Team filter unavailable for multi-year span" if team_disabled else None,
    )
    st.selectbox(
        "League",
        options=LEAGUES.keys(),
        key="cr_league",
        disabled=team_disabled or league_disabled,
        help="League filter unavailable for years before 2013 due to possible innacuracies" if league_disabled else None,
    )


def load_data(s_year, e_year, mode, position="all"):
    if mode == MODE_SINGLE:
        return load_final_year(s_year)

    frames = []
    for yr in range(s_year, e_year + 1):
        d = load_final_year(yr)
        if d is not None and not d.empty:
            if mode == MODE_SPLIT and domain == "Hitting":
                d = filter_by_position(d, position)
            d["Season"] = yr
            frames.append(d)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    if mode == MODE_SPLIT:
        return combined

    if domain == "Hitting":
        combined = filter_by_position(combined, position)
        if combined.empty:
            return pd.DataFrame()

    return aggregate_player_group(combined)


position_val = st.session_state.get("cr_position", "all") if domain == "Hitting" else "all"

df = load_data(sel_start, sel_end, mode, position_val)
if df is None or df.empty:
    st.error(f"No data found for {sel_start}–{sel_end}.")
    st.stop()

if domain == "Hitting":
    qualifier_col = "PA"
    min_val = int(st.session_state.get("cr_min_pa", 0))

    if mode == MODE_SINGLE:
        df = filter_by_position(df, position_val)
elif domain == "Pitching":
    qualifier_col = "IP"
    min_val = int(st.session_state.get("cr_min_ip", 0))
else:
    qualifier_col = None
    min_pa_val = int(st.session_state.get("cr_min_pa", 0))
    min_ip_val = int(st.session_state.get("cr_min_ip", 0))

team_disabled = (mode == MODE_MULTI)
league_disabled = (sel_start < 2013)
team_val = "all" if team_disabled else st.session_state.get("cr_team", "all")
league_val = "All" if team_disabled or league_disabled else st.session_state.get("cr_league", "All")

if team_val != "all" and "Team" in df.columns:
    target = normalize_team(team_val)
    df = df[df["Team"].astype(str).apply(lambda t: normalize_team(t) == target)]

if league_val != "All" and "Team" in df.columns:
    league_teams = LEAGUES[league_val]
    df = df[df["Team"].astype(str).apply(lambda t: normalize_team(t) in league_teams)]

if df.empty:
    st.error("No players match the selected filters.")
    st.stop()

if domain == "Combined":
    if (min_pa_val > 0 or min_ip_val > 0) and {"PA", "IP"}.issubset(df.columns):
        pa_num = pd.to_numeric(df["PA"], errors="coerce").fillna(0)
        ip_num = pd.to_numeric(df["IP"], errors="coerce").fillna(0)
        pa_ok = pa_num >= min_pa_val if min_pa_val > 0 else pd.Series(False, index=df.index)
        ip_ok = ip_num >= min_ip_val if min_ip_val > 0 else pd.Series(False, index=df.index)
        df = df[pa_ok | ip_ok]
elif min_val > 0 and qualifier_col in df.columns:
    df = df[pd.to_numeric(df[qualifier_col], errors="coerce").fillna(0) >= min_val]

if x_stat not in df.columns or y_stat not in df.columns:
    st.error("One or both selected stats are not available in this dataset.")
    st.stop()

df[x_stat] = pd.to_numeric(df[x_stat], errors="coerce")
df[y_stat] = pd.to_numeric(df[y_stat], errors="coerce")
df = df.dropna(subset=[x_stat, y_stat])

with col2:
    if len(df) < 2:
        st.error("Not enough qualified players to compute a correlation.")
        st.stop()

    x_label = label_map.get(x_stat, x_stat)
    y_label = label_map.get(y_stat, y_stat)

    df["_disp_x"] = df[x_stat].apply(lambda v: format_stat(x_stat, v))
    df["_disp_y"] = df[y_stat].apply(lambda v: format_stat(y_stat, v))

    x_flip = x_stat in lower_better
    y_flip = y_stat in lower_better

    df["_plot_x"] = df[x_stat]
    df["_plot_y"] = df[y_stat]

    x_axis_label = x_label
    y_axis_label = y_label

    stats_x_for_math = -df[x_stat] if x_flip else df[x_stat]
    stats_y_for_math = -df[y_stat] if y_flip else df[y_stat]

    r = np.corrcoef(stats_x_for_math, stats_y_for_math)[0, 1]
    r_squared = r ** 2

    math_slope, math_intercept = np.polyfit(stats_x_for_math, stats_y_for_math, 1)

    hover_data = {
        "_disp_x": True,
        "_disp_y": True,
        "_plot_x": False,
        "_plot_y": False,
    }
    labels = {"_plot_x": x_axis_label, "_plot_y": y_axis_label, "_disp_x": x_label, "_disp_y": y_label}

    if mode == MODE_SPLIT and "Season" in df.columns:
        hover_data["Season"] = True

    show_names = st.session_state.get("cr_show_names", False)

    search_raw = st.session_state.get("cr_search", "").strip().lower()
    search_terms = [t.strip() for t in search_raw.split(",") if t.strip()]

    highlight_team_val = st.session_state.get("cr_highlight_team", "all")

    is_match = pd.Series(False, index=df.index)

    if search_terms and "Name" in df.columns:
        name_mask = pd.Series(False, index=df.index)
        for term in search_terms:
            name_mask |= df["Name"].astype(str).str.lower().str.contains(term, na=False)
        is_match |= name_mask

    if highlight_team_val != "all" and "Team" in df.columns:
        target_team = normalize_team(highlight_team_val)
        is_match |= df["Team"].astype(str).apply(lambda t: normalize_team(t) == target_team)

    df_main = df[~is_match].copy()

    if show_names and "Name" in df_main.columns:
        if mode == MODE_SPLIT and "Season" in df_main.columns:
            df_main["_label_text"] = df_main["Name"].astype(str) + " (" + df_main["Season"].astype(str) + ")"
        else:
            df_main["_label_text"] = df_main["Name"].astype(str)
        hover_data["_label_text"] = False
    scatter_mode = "markers+text" if show_names else "markers"

    fig = px.scatter(
        df_main,
        x="_plot_x",
        y="_plot_y",
        hover_name="Name" if "Name" in df_main.columns else None,
        hover_data=hover_data,
        labels=labels,
        text="_label_text" if show_names and "_label_text" in df_main.columns else None,
    )
    fig.update_traces(
        marker=dict(size=8, opacity=0.65, color="#2c3e50"),
        mode=scatter_mode,
        textposition="top center",
        textfont=dict(size=10, color="#1a1a1a"),
        selector=dict(type="scatter"),
    )

    if (search_terms or highlight_team_val != "all") and "Name" in df.columns:
        matches = df[is_match]
        if not matches.empty:
            if mode == MODE_SPLIT and "Season" in matches.columns:
                match_label = matches["Name"].astype(str) + " (" + matches["Season"].astype(str) + ")"
            else:
                match_label = matches["Name"].astype(str)
            match_text = "<b>" + match_label + "</b>"

            fig.add_trace(go.Scatter(
                x=matches["_plot_x"], y=matches["_plot_y"],
                mode="markers+text",
                text=match_text,
                textposition="top center",
                textfont=dict(size=13, color="#1e75aa"),
                marker=dict(size=16, color="#007AFF", line=dict(width=2, color="#6a1b9a")),
                name="Highlighted",
                customdata=matches[["_disp_x", "_disp_y"]].values if {"_disp_x", "_disp_y"}.issubset(matches.columns) else None,
                hovertemplate=(
                    "%{text}<br>"
                    + x_label + ": %{customdata[0]}<br>"
                    + y_label + ": %{customdata[1]}<extra></extra>"
                ) if {"_disp_x", "_disp_y"}.issubset(matches.columns) else None,
            ))

    plot_slope, plot_intercept = np.polyfit(df["_plot_x"], df["_plot_y"], 1)

    x_range = np.array([df["_plot_x"].min(), df["_plot_x"].max()])
    y_range = plot_slope * x_range + plot_intercept

    fig.add_trace(go.Scatter(
        x=x_range, y=y_range, mode="lines",
        line=dict(color="#c0392b", width=2),
        name="Trend", hoverinfo="skip",
    ))

    span_label = f"{sel_start}" if mode == MODE_SINGLE else f"{sel_start}–{sel_end}"
    if domain == "Combined":
        qualifier_label = f"Minimum {min_pa_val} PA or {min_ip_val} IP"
    else:
        qualifier_label = f"Minimum {min_val} {qualifier_col}"

    extra_bits = []
    if domain == "Hitting" and position_val != "all":
        extra_bits.append(POSITION_OPTIONS[position_val])
    if not team_disabled and team_val != "all":
        extra_bits.append(TEAM_OPTIONS[team_val])
    if not (team_disabled or league_disabled) and league_val != "All":
        extra_bits.append(league_val)
    if extra_bits:
        qualifier_label += " · " + " · ".join(extra_bits)

    mode_label = {MODE_SINGLE: "", MODE_SPLIT: "Single Season", MODE_MULTI: ""}[mode]

    title_main = f"{x_label} vs {y_label} ({span_label})"
    if mode_label:
        title_main += f" - {mode_label}"

    fig.update_layout(
        title=dict(
            text=f"<b>{title_main}</b><br><span style='font-size:14px; color:#0;'>{qualifier_label}</span>",
            font=dict(color="#1a1a1a", size=22), x=0.5, xanchor="center",
        ),
        height=650,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#1a1a1a"),
        margin=dict(l=80, r=40, t=60, b=40),
        legend=dict(font=dict(color="#1a1a1a")),
        showlegend=False,
        xaxis=dict(
            title=dict(font=dict(color="#1a1a1a", size=18)),
            tickfont=dict(color="#1a1a1a", size=15),
            showgrid=True,
            nticks=10,
            gridcolor="#e6e6e6",
            zerolinecolor="#e6e6e6",
            linecolor="#1a1a1a",
            autorange="reversed" if x_flip else True,
        ),
        yaxis=dict(
            title=dict(font=dict(color="#1a1a1a", size=18)),
            tickfont=dict(color="#1a1a1a", size=15),
            showgrid=True,
            nticks=10,
            gridcolor="#e6e6e6",
            zerolinecolor="#e6e6e6",
            linecolor="#1a1a1a",
            autorange="reversed" if y_flip else True,
        ),
    )

    st.markdown("""
<style>
[data-testid="stPlotlyChart"] {
    overflow-x: auto;
    overflow-y: auto;
}
[data-testid="stPlotlyChart"] > div {
    min-width: 700px;
    min-height: 700px;
}
</style>
""", unsafe_allow_html=True)
    st.plotly_chart(fig, width="stretch")

    m1, m2, m3 = st.columns(3)
    m1.metric("Correlation (r)", f"{r:.3f}")
    m2.metric("R²", f"{r_squared:.3f}")
    m3.metric("Sample Size", f"{len(df)}")

    st.caption(f"Trend line: y = {math_slope:.4f}x + {math_intercept:.4f}")

    fig.update_layout(margin=dict(l=80, r=40, t=60, b=100))
    pdf_buffer = BytesIO()
    fig.write_image(pdf_buffer, format="pdf", width=1200, height=700)
    pdf_buffer.seek(0)
    fig.update_layout(margin=dict(l=80, r=40, t=60, b=40))
    st.download_button(
        "Download as PDF",
        data=pdf_buffer,
        file_name=f"{span_label} {x_label} vs {y_label}.pdf",
        mime="application/pdf",
    )

    st.markdown(
        "<div style='text-align:center; color:#888; font-size:1rem; margin-top:1rem; margin-bottom:3rem;'>"
        "Data: Baseball Reference · FanGraphs · Baseball Savant"
        "</div>",
        unsafe_allow_html=True,
    )
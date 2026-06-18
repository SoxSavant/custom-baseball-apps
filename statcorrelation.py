import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date
from io import BytesIO

from utils import get_dynamic_min_pa, get_dynamic_min_ip

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
    st.title("Stat Correlation Finder")
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

for key, default in [
    ("cr_domain", "Hitting"),
    ("cr_year", current_year),
    ("cr_min_type", "PA"),
    ("cr_min_pa", get_dynamic_min_pa(current_year)/2),
    ("cr_min_ip", 0),
]:
    if key not in st.session_state:
        st.session_state[key] = default

col1, col2 = st.columns([.5, 2])

with col1:
    domain = st.radio("Domain", ["Hitting", "Pitching"], key="cr_domain", horizontal=True)

    if domain == "Hitting":
        from h_utils import (
            STAT_ALLOWLIST, format_stat, start_year, label_map,
            load_final_year, STAT_ROUND, lower_better,
        )
        min_type_default = "PA"
    else:
        from p_utils import (
            STAT_ALLOWLIST, format_stat, start_year, label_map,
            load_final_year, STAT_ROUND, lower_better,
        )
        min_type_default = "IP"

    if "cr_last_domain" not in st.session_state:
        st.session_state.cr_last_domain = domain
    if domain != st.session_state.cr_last_domain:
        st.session_state["cr_min_type"] = min_type_default
        st.session_state["cr_min_pa"] = get_dynamic_min_pa(current_year)/2
        st.session_state["cr_min_ip"] = get_dynamic_min_ip(current_year)/2
        st.session_state.cr_last_domain = domain

    st.selectbox("Year", options=list(range(current_year, start_year - 1, -1)), key="cr_year")
    year = st.session_state["cr_year"]

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
    else:
        st.number_input("Min IP", min_value=0, max_value=5000, key="cr_min_ip")

df = load_final_year(year)
if df is None or df.empty:
    st.error(f"No data found for {year}.")
    st.stop()

if domain == "Hitting":
    qualifier_col = "PA"
    min_val = int(st.session_state.get("cr_min_pa", 0))
else:
    qualifier_col = "IP"
    min_val = int(st.session_state.get("cr_min_ip", 0))

if min_val > 0 and qualifier_col in df.columns:
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

    # 1. Use the original POSITIVE stats for plotting so the axis ticks look correct
    df["_plot_x"] = df[x_stat]
    df["_plot_y"] = df[y_stat]

    x_axis_label = x_label
    y_axis_label = y_label

    # 2. For Math/Correlation: Account for 'lower is better' so r and slope don't flip backwards
    stats_x_for_math = -df[x_stat] if x_flip else df[x_stat]
    stats_y_for_math = -df[y_stat] if y_flip else df[y_stat]
    
    r = np.corrcoef(stats_x_for_math, stats_y_for_math)[0, 1]
    r_squared = r ** 2
    
    # Calculate math slope using the flipped logic
    math_slope, math_intercept = np.polyfit(stats_x_for_math, stats_y_for_math, 1)

    # 3. Create the scatter plot using the POSITIVE numbers
    fig = px.scatter(
        df,
        x="_plot_x",
        y="_plot_y",
        hover_name="Name" if "Name" in df.columns else None,
        # Hover data simplifies since we are using native positive values
        hover_data={
            "_plot_x": True,
            "_plot_y": True,
        },
        labels={"_plot_x": x_axis_label, "_plot_y": y_axis_label},
    )
    fig.update_traces(marker=dict(size=8, opacity=0.65, color="#2c3e50"))

    # 4. Draw the Trendline using native plot coordinates
    # We figure out if the trendline should look visually positive or negative
    # If one (and only one) axis is flipped, the visual slope direction inverts
    visual_slope_direction = -1 if (x_flip ^ y_flip) else 1
    plot_slope, plot_intercept = np.polyfit(df["_plot_x"], df["_plot_y"], 1)
    
    x_range = np.array([df["_plot_x"].min(), df["_plot_x"].max()])
    
    # Use standard polyfit for the visual line orientation
    y_range = plot_slope * x_range + plot_intercept
    
    fig.add_trace(go.Scatter(
        x=x_range, y=y_range, mode="lines",
        line=dict(color="#c0392b", width=2),
        name="Trend", hoverinfo="skip",
    ))
    
    fig.update_layout(
        title=dict(text=f"{year} {x_label} vs {y_label}", font=dict(color="#1a1a1a", size=22), x=0.5, xanchor="center"),
        height=650,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#1a1a1a"),
        margin=dict(l=80, r=40, t=60, b=40),
        legend=dict(font=dict(color="#1a1a1a")),
        xaxis=dict(
            title=dict(font=dict(color="#1a1a1a", size = 18)),
            tickfont=dict(color="#1a1a1a", size = 15),
            gridcolor="#e6e6e6",
            zerolinecolor="#e6e6e6",
            linecolor="#1a1a1a",
            autorange="reversed" if x_flip else True
        ),
        yaxis=dict(
            title=dict(font=dict(color="#1a1a1a", size = 18)),
            tickfont=dict(color="#1a1a1a", size = 15),
            gridcolor="#e6e6e6",
            zerolinecolor="#e6e6e6",
            linecolor="#1a1a1a",
            autorange="reversed" if y_flip else True
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

    pdf_buffer = BytesIO()
    fig.update_layout(margin=dict(l=80, r=40, t=60, b=100))
    pdf_buffer = BytesIO()
    fig.write_image(pdf_buffer, format="pdf", width=1200, height=700)
    pdf_buffer.seek(0)
    fig.update_layout(margin=dict(l=80, r=40, t=60, b=40))
    st.download_button(
        "Download as PDF",
        data=pdf_buffer,
        file_name=f"{year} {x_label} vs {y_label}.pdf",
        mime="application/pdf",
    )


    st.markdown(
        "<div style='text-align:center; color:#888; font-size:1rem; margin-top:1rem; margin-bottom:3rem;'>"
        "Data: Baseball Reference · FanGraphs · Baseball Savant"
        "</div>",
        unsafe_allow_html=True,
    )

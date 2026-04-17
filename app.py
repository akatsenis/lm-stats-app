import re
from io import StringIO, BytesIO
from textwrap import dedent
from itertools import product

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy import stats
from scipy.stats import t, norm, gaussian_kde, chi2, nct

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.diagnostic import normal_ad
from sklearn.decomposition import PCA

from openpyxl import Workbook
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
)

# -------------------------------------------------
# Page configuration and style
# -------------------------------------------------
st.set_page_config(page_title="lm Stats Suite", page_icon="🔬", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 0.9rem; padding-bottom: 2rem;}
    .app-header {
        border:1px solid #e2e8f0; border-radius:14px; padding:16px 20px;
        background: linear-gradient(90deg, #f8fafc 0%, #ffffff 100%);
        margin-bottom: 1rem; box-shadow: 0 1px 4px rgba(15,23,42,0.05);
    }
    .app-title {font-size: 1.75rem; font-weight: 700; margin-bottom: 0.2rem; color:#0f172a;}
    .app-sub {font-size: 0.96rem; color:#475569;}
    .report-table table {width:100%; border-collapse:collapse; background:white; font-size:0.95rem;}
    .report-table caption {text-align:left; font-weight:700; font-size:1rem; color:#111827; margin-bottom:0.55rem;}
    .report-table thead th {
        border-top:2px solid #111827; border-bottom:1px solid #111827;
        padding:8px 12px; text-align:center; background:#f8fafc; color:#111827;
    }
    .report-table tbody td {padding:8px 12px; text-align:center; border:none;}
    .report-table tbody tr:last-child td {border-bottom:2px solid #111827;}
    .report-caption {font-size:0.85rem; color:#475569; margin-top:-0.5rem; margin-bottom:0.75rem;}
    div[data-testid='stMetric'] {
        border:1px solid #e2e8f0; border-radius:12px; padding:10px 12px; background:#fff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# -------------------------------------------------
# Global visual controls
# -------------------------------------------------
st.sidebar.title("🔬 lm Stats")
st.sidebar.markdown("Select a tool below:")
app_selection = st.sidebar.radio(
    "Navigation",
    [
        "01 - Descriptive Statistics",
        "02 - Regression Intervals",
        "03 - Shelf Life Estimator",
        "04 - Dissolution Comparison (f2)",
        "05 - Two-Sample Tests",
        "06 - Two-Way ANOVA",
        "07 - Tolerance & Confidence Intervals",
        "08 - PCA Analysis",
        "09 - DoE / Response Surfaces",
    ],
)

PLOT_STYLE_KEYS = [
    "All graphs",
    "Descriptive summary",
    "Regression intervals",
    "Shelf life",
    "Dissolution comparison",
    "Two-sample comparison",
    "Two-way ANOVA interaction",
    "PCA score plot",
    "PCA loading plot",
    "DoE contour",
    "Residual plot",
    "Q-Q plot",
]
LINE_STYLE_MAP = {"Solid": "-", "Dash": "--", "Dot": ":", "Dash-dot": "-."}
DEFAULT_STYLE_CFG = {
    "fig_w": 8.5,
    "fig_h": 5.5,
    "show_legend": True,
    "legend_loc": "best",
    "primary_color": "#1f77b4",
    "secondary_color": "#ff7f0e",
    "tertiary_color": "#2ca02c",
    "band_color": "#93c5fd",
    "grid_alpha": 0.25,
    "line_style": "-",
    "aux_line_style": "--",
    "line_width": 2.0,
    "aux_line_width": 1.4,
    "marker_size": 46,
    "tick_dir": "out",
    "tick_len": 4,
    "border_width": 1.0,
    "show_top": True,
    "show_right": True,
    "x_min": None,
    "x_max": None,
    "y_min": None,
    "y_max": None,
    "arrow_size": 0.03,
}
if "plot_style_cfg" not in st.session_state:
    st.session_state["plot_style_cfg"] = {k: {} for k in PLOT_STYLE_KEYS}
    st.session_state["plot_style_cfg"]["All graphs"] = DEFAULT_STYLE_CFG.copy()

with st.sidebar.expander("Display & export settings", expanded=False):
    DEFAULT_DECIMALS = st.slider("Default decimals", 1, 8, 3)

    target_graph = st.selectbox("Graph to customize", PLOT_STYLE_KEYS, index=0)
    base_cfg = DEFAULT_STYLE_CFG.copy()
    base_cfg.update(st.session_state["plot_style_cfg"].get("All graphs", {}))
    current_cfg = base_cfg.copy()
    if target_graph != "All graphs":
        current_cfg.update(st.session_state["plot_style_cfg"].get(target_graph, {}))

    c1, c2 = st.columns(2)
    with c1:
        fig_w = st.slider("Figure width", 5.0, 16.0, float(current_cfg.get("fig_w", 8.5)), 0.5, key=f"{target_graph}_fig_w")
        show_legend = st.checkbox("Show legend", value=bool(current_cfg.get("show_legend", True)), key=f"{target_graph}_show_legend")
        primary_color = st.color_picker("Primary color", value=current_cfg.get("primary_color", "#1f77b4"), key=f"{target_graph}_primary")
        secondary_color = st.color_picker("Secondary color", value=current_cfg.get("secondary_color", "#ff7f0e"), key=f"{target_graph}_secondary")
        tertiary_color = st.color_picker("Tertiary color", value=current_cfg.get("tertiary_color", "#2ca02c"), key=f"{target_graph}_tertiary")
        line_width = st.slider("Main line width", 0.5, 4.0, float(current_cfg.get("line_width", 2.0)), 0.1, key=f"{target_graph}_lw")
        aux_line_width = st.slider("Aux line width", 0.5, 3.0, float(current_cfg.get("aux_line_width", 1.4)), 0.1, key=f"{target_graph}_alw")
        marker_size = st.slider("Marker size", 10, 150, int(current_cfg.get("marker_size", 46)), 1, key=f"{target_graph}_ms")
        arrow_size = st.slider("Arrow size", 0.005, 0.20, float(current_cfg.get("arrow_size", 0.03)), 0.005, key=f"{target_graph}_arrow")
    with c2:
        fig_h = st.slider("Figure height", 3.0, 12.0, float(current_cfg.get("fig_h", 5.5)), 0.5, key=f"{target_graph}_fig_h")
        legend_loc = st.selectbox("Legend location", ["best", "upper right", "upper left", "lower right", "lower left", "center left", "center right", "lower center", "upper center"], index=["best", "upper right", "upper left", "lower right", "lower left", "center left", "center right", "lower center", "upper center"].index(current_cfg.get("legend_loc", "best")), key=f"{target_graph}_legend_loc")
        band_color = st.color_picker("Band / area color", value=current_cfg.get("band_color", "#93c5fd"), key=f"{target_graph}_band")
        grid_alpha = st.slider("Grid transparency", 0.0, 1.0, float(current_cfg.get("grid_alpha", 0.25)), 0.05, key=f"{target_graph}_ga")
        line_style_name = st.selectbox("Main line style", list(LINE_STYLE_MAP.keys()), index=list(LINE_STYLE_MAP.values()).index(current_cfg.get("line_style", "-")) if current_cfg.get("line_style", "-") in LINE_STYLE_MAP.values() else 0, key=f"{target_graph}_ls")
        aux_line_style_name = st.selectbox("Aux line style", list(LINE_STYLE_MAP.keys()), index=list(LINE_STYLE_MAP.values()).index(current_cfg.get("aux_line_style", "--")) if current_cfg.get("aux_line_style", "--") in LINE_STYLE_MAP.values() else 1, key=f"{target_graph}_als")
        tick_dir = st.selectbox("Tick direction", ["out", "in", "inout"], index=["out", "in", "inout"].index(current_cfg.get("tick_dir", "out")), key=f"{target_graph}_tick_dir")
        tick_len = st.slider("Tick length", 0, 12, int(current_cfg.get("tick_len", 4)), 1, key=f"{target_graph}_tick_len")
        border_width = st.slider("Border width", 0.5, 3.0, float(current_cfg.get("border_width", 1.0)), 0.1, key=f"{target_graph}_bw")
        show_top = st.checkbox("Show top border", value=bool(current_cfg.get("show_top", True)), key=f"{target_graph}_top")
        show_right = st.checkbox("Show right border", value=bool(current_cfg.get("show_right", True)), key=f"{target_graph}_right")

    st.markdown("**Axis limits (leave blank for automatic)**")
    x1, x2, y1, y2 = st.columns(4)
    x_min_cfg = x1.text_input("X min", value="" if current_cfg.get("x_min", None) in [None, ""] else str(current_cfg.get("x_min")), key=f"{target_graph}_xmin")
    x_max_cfg = x2.text_input("X max", value="" if current_cfg.get("x_max", None) in [None, ""] else str(current_cfg.get("x_max")), key=f"{target_graph}_xmax")
    y_min_cfg = y1.text_input("Y min", value="" if current_cfg.get("y_min", None) in [None, ""] else str(current_cfg.get("y_min")), key=f"{target_graph}_ymin")
    y_max_cfg = y2.text_input("Y max", value="" if current_cfg.get("y_max", None) in [None, ""] else str(current_cfg.get("y_max")), key=f"{target_graph}_ymax")

    st.session_state["plot_style_cfg"][target_graph] = {
        "fig_w": fig_w,
        "fig_h": fig_h,
        "show_legend": show_legend,
        "legend_loc": legend_loc,
        "primary_color": primary_color,
        "secondary_color": secondary_color,
        "tertiary_color": tertiary_color,
        "band_color": band_color,
        "grid_alpha": grid_alpha,
        "line_style": LINE_STYLE_MAP[line_style_name],
        "aux_line_style": LINE_STYLE_MAP[aux_line_style_name],
        "line_width": line_width,
        "aux_line_width": aux_line_width,
        "marker_size": marker_size,
        "tick_dir": tick_dir,
        "tick_len": tick_len,
        "border_width": border_width,
        "show_top": show_top,
        "show_right": show_right,
        "x_min": x_min_cfg.strip() if isinstance(x_min_cfg, str) else x_min_cfg,
        "x_max": x_max_cfg.strip() if isinstance(x_max_cfg, str) else x_max_cfg,
        "y_min": y_min_cfg.strip() if isinstance(y_min_cfg, str) else y_min_cfg,
        "y_max": y_max_cfg.strip() if isinstance(y_max_cfg, str) else y_max_cfg,
        "arrow_size": arrow_size,
    }

    if st.button("Reset selected graph style", key="reset_graph_style"):
        st.session_state["plot_style_cfg"][target_graph] = {} if target_graph != "All graphs" else DEFAULT_STYLE_CFG.copy()
        st.rerun()

def _parse_style_float(val):
    if val is None:
        return None
    if isinstance(val, (int, float, np.number)):
        return float(val)
    s = str(val).strip()
    if s == "":
        return None
    try:
        return float(s)
    except Exception:
        return None

def get_plot_cfg(plot_key="All graphs"):
    cfg = DEFAULT_STYLE_CFG.copy()
    cfg.update(st.session_state.get("plot_style_cfg", {}).get("All graphs", {}))
    if plot_key and plot_key != "All graphs":
        specific = st.session_state.get("plot_style_cfg", {}).get(plot_key, {})
        cfg.update({k: v for k, v in specific.items() if v not in [None, ""]})
    cfg["x_min"] = _parse_style_float(cfg.get("x_min"))
    cfg["x_max"] = _parse_style_float(cfg.get("x_max"))
    cfg["y_min"] = _parse_style_float(cfg.get("y_min"))
    cfg["y_max"] = _parse_style_float(cfg.get("y_max"))
    return cfg

_ALL_CFG = get_plot_cfg("All graphs")
FIG_W = _ALL_CFG["fig_w"]
FIG_H = _ALL_CFG["fig_h"]
SHOW_LEGEND = _ALL_CFG["show_legend"]
LEGEND_LOC = _ALL_CFG["legend_loc"]
PRIMARY_COLOR = _ALL_CFG["primary_color"]
SECONDARY_COLOR = _ALL_CFG["secondary_color"]
BAND_COLOR = _ALL_CFG["band_color"]
GRID_ALPHA = _ALL_CFG["grid_alpha"]

st.sidebar.divider()
st.sidebar.info("Paste data from Excel. Tables, charts, Excel exports, and PDF-style reports are built into the app. Exported figures keep the current display settings.")


# -------------------------------------------------
# UI helpers
# -------------------------------------------------
def app_header(title, subtitle=""):
    st.markdown(
        f"""
        <div class='app-header'>
            <div class='app-title'>{title}</div>
            <div class='app-sub'>{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_box(text):
    st.markdown(f"<div class='report-caption'>{text}</div>", unsafe_allow_html=True)


# -------------------------------------------------
# Data helpers
# -------------------------------------------------
def to_numeric(series):
    return pd.to_numeric(series.astype(str).str.strip().str.replace("%", "", regex=False), errors="coerce")


def parse_pasted_table(text, header=True):
    text = str(text).strip()
    if not text:
        return None
    parsers = [
        lambda s: pd.read_csv(StringIO(s), sep="\t", engine="python", header=0 if header else None),
        lambda s: pd.read_csv(StringIO(s), sep=",", engine="python", header=0 if header else None),
        lambda s: pd.read_csv(StringIO(s), sep=";", engine="python", header=0 if header else None),
        lambda s: pd.read_csv(StringIO(s), sep=r"\s+", engine="python", header=0 if header else None),
    ]
    for parser in parsers:
        try:
            df = parser(text)
            if df is not None and df.shape[1] >= 1:
                if header:
                    df.columns = [str(c).strip() for c in df.columns]
                return df.dropna(how="all").reset_index(drop=True)
        except Exception:
            continue
    return None


def parse_xy(text):
    """Return dataframe with x,y and axis labels; accepts with or without headers."""
    raw = parse_pasted_table(text, header=False)
    if raw is None or raw.shape[1] < 2:
        raise ValueError("Paste at least two columns from Excel.")
    raw = raw.iloc[:, :2].copy()
    x_label, y_label = "X", "Y"
    first_row = raw.iloc[0].astype(str)
    first_row_numeric = pd.to_numeric(first_row.str.replace("%", "", regex=False), errors="coerce")
    if first_row_numeric.isna().any():
        x_label = str(raw.iloc[0, 0]).strip() or "X"
        y_label = str(raw.iloc[0, 1]).strip() or "Y"
        raw = raw.iloc[1:].reset_index(drop=True)
    raw.columns = ["x", "y"]
    raw["x"] = to_numeric(raw["x"])
    raw["y"] = to_numeric(raw["y"])
    raw = raw.dropna().sort_values("x").reset_index(drop=True)
    if len(raw) < 3 or raw["x"].nunique() < 2:
        raise ValueError("At least 3 valid rows and 2 unique X values are required.")
    return raw, x_label, y_label


def parse_x_values(text):
    text = str(text).strip()
    if not text:
        return np.array([])
    vals = []
    for part in re.split(r"[\s,;\t]+", text):
        if part:
            vals.append(float(part))
    return np.array(vals, dtype=float)


def parse_optional_float(txt):
    txt = str(txt).strip()
    if txt == "":
        return None
    return float(txt)


def parse_one_col(text):
    df = parse_pasted_table(text, header=False)
    if df is None:
        return np.array([])
    return to_numeric(df.iloc[:, 0]).dropna().to_numpy()


def get_numeric_columns(df, min_nonempty=2, required_numeric_ratio=0.95):
    out = []
    for col in df.columns:
        raw = df[col]
        raw_str = raw.astype(str).str.strip()
        nonempty_mask = raw.notna() & raw_str.ne("") & raw_str.str.lower().ne("nan")
        if nonempty_mask.sum() < min_nonempty:
            continue
        converted = pd.to_numeric(raw_str.str.replace("%", "", regex=False), errors="coerce")
        numeric_ratio_among_nonempty = converted[nonempty_mask].notna().mean()
        if numeric_ratio_among_nonempty >= required_numeric_ratio:
            out.append(col)
    return out


def fmt_p(p):
    if pd.isna(p):
        return "-"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# -------------------------------------------------
# Table display and download helpers
# -------------------------------------------------
def report_table(df, caption="", decimals=None):
    decimals = DEFAULT_DECIMALS if decimals is None else decimals
    styled = (
        df.style
        .hide(axis="index")
        .set_caption(caption)
        .set_table_styles([
            {"selector": "caption", "props": [("text-align", "left"), ("font-size", "1rem"), ("font-weight", "700"), ("margin-bottom", "0.55rem")]},
            {"selector": "thead th", "props": [("border-top", "2px solid #111827"), ("border-bottom", "1px solid #111827"), ("padding", "8px 12px"), ("text-align", "center"), ("background-color", "#f8fafc")]},
            {"selector": "tbody td", "props": [("padding", "8px 12px"), ("text-align", "center")]},
            {"selector": "tbody tr:last-child td", "props": [("border-bottom", "2px solid #111827")]},
        ])
        .format(precision=decimals, na_rep="-")
    )
    st.markdown(f"<div class='report-table'>{styled.to_html()}</div>", unsafe_allow_html=True)


def make_excel_bytes(sheet_map):
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        for sheet_name, df in sheet_map.items():
            safe = re.sub(r"[^A-Za-z0-9 _-]", "", sheet_name)[:31] or "Sheet1"
            out = df.copy()
            out.to_excel(writer, sheet_name=safe, index=False)
            ws = writer.sheets[safe]
            for col_cells in ws.columns:
                max_len = 0
                col_letter = col_cells[0].column_letter
                for cell in col_cells:
                    try:
                        max_len = max(max_len, len(str(cell.value)))
                    except Exception:
                        pass
                ws.column_dimensions[col_letter].width = min(max_len + 2, 28)
    bio.seek(0)
    return bio.getvalue()


def fig_to_png_bytes(fig):
    bio = BytesIO()
    fig.savefig(bio, format="png", dpi=220, bbox_inches="tight", facecolor="white")
    bio.seek(0)
    return bio.getvalue()


# -------------------------------------------------
# PDF report generation
# -------------------------------------------------
def _pdf_table(df, styles, title, decimals=3, max_rows=40):
    story = []
    story.append(Paragraph(title, styles["Heading3"]))
    if len(df) > max_rows:
        story.append(Paragraph(f"Table truncated to first {max_rows} rows for compact reporting.", styles["BodyText"]))
        df = df.head(max_rows)
    fmt_df = df.copy()
    for c in fmt_df.columns:
        if pd.api.types.is_numeric_dtype(fmt_df[c]):
            fmt_df[c] = fmt_df[c].map(lambda x: "-" if pd.isna(x) else f"{x:.{decimals}f}")
        else:
            fmt_df[c] = fmt_df[c].fillna("-").astype(str)
    data = [list(fmt_df.columns)] + fmt_df.values.tolist()
    ncols = max(1, len(fmt_df.columns))
    page_w = 27.5 * cm
    col_width = page_w / ncols
    tbl = Table(data, repeatRows=1, colWidths=[col_width] * ncols)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F8FAFC")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("LEADING", (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LINEABOVE", (0, 0), (-1, 0), 1.2, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 1.2, colors.black),
    ]))
    story.extend([tbl, Spacer(1, 0.35 * cm)])
    return story


def make_pdf_report(report_title, module_name, statistical_analysis, offer_text, python_tools, tables, figures, conclusion=None, decimals=3):
    bio = BytesIO()
    doc = SimpleDocTemplate(
        bio,
        pagesize=landscape(A4),
        leftMargin=1.2 * cm,
        rightMargin=1.2 * cm,
        topMargin=1.2 * cm,
        bottomMargin=1.1 * cm,
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="SmallBody", parent=styles["BodyText"], fontSize=9, leading=12, alignment=TA_LEFT))
    story = []

    story.append(Paragraph(report_title, styles["Title"]))
    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph(f"Module: <b>{module_name}</b>", styles["Heading2"]))
    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph("Statistical Analysis", styles["Heading2"]))
    story.append(Paragraph(statistical_analysis, styles["SmallBody"]))
    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph("What this analysis offers", styles["Heading2"]))
    story.append(Paragraph(offer_text, styles["SmallBody"]))
    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph("Python tools used", styles["Heading2"]))
    story.append(Paragraph(python_tools, styles["SmallBody"]))
    if conclusion:
        story.append(Spacer(1, 0.15 * cm))
        story.append(Paragraph("Conclusion", styles["Heading2"]))
        story.append(Paragraph(conclusion, styles["SmallBody"]))
    story.append(Spacer(1, 0.2 * cm))

    if tables:
        story.append(Paragraph("Tables", styles["Heading2"]))
        for caption, df in tables:
            story.extend(_pdf_table(df, styles, caption, decimals=decimals))

    if figures:
        story.append(PageBreak())
        story.append(Paragraph("Figures", styles["Heading2"]))
        for caption, fig_bytes in figures:
            story.append(Paragraph(caption, styles["Heading3"]))
            img = Image(BytesIO(fig_bytes))
            img._restrictSize(24.5 * cm, 13.5 * cm)
            story.append(img)
            story.append(Spacer(1, 0.3 * cm))

    doc.build(story)
    bio.seek(0)
    return bio.getvalue()


def export_results(prefix, report_title, module_name, statistical_analysis, offer_text, python_tools, table_map, figure_map=None, conclusion=None, decimals=None):
    decimals = DEFAULT_DECIMALS if decimals is None else decimals
    figure_map = figure_map or {}
    c1, c2 = st.columns(2)
    excel_bytes = make_excel_bytes(table_map)
    pdf_bytes = make_pdf_report(
        report_title=report_title,
        module_name=module_name,
        statistical_analysis=statistical_analysis,
        offer_text=offer_text,
        python_tools=python_tools,
        tables=list(table_map.items()),
        figures=list(figure_map.items()),
        conclusion=conclusion,
        decimals=decimals,
    )
    with c1:
        st.download_button("Download Excel workbook", excel_bytes, file_name=f"{prefix}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with c2:
        st.download_button("Download PDF-style report", pdf_bytes, file_name=f"{prefix}.pdf", mime="application/pdf")


# -------------------------------------------------
# Plot helpers
# -------------------------------------------------

def apply_ax_style(ax, title, xlabel, ylabel, legend=None, plot_key="All graphs"):
    cfg = get_plot_cfg(plot_key)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=cfg["grid_alpha"])
    ax.tick_params(direction=cfg["tick_dir"], length=cfg["tick_len"], width=cfg["border_width"])
    ax.spines["left"].set_linewidth(cfg["border_width"])
    ax.spines["bottom"].set_linewidth(cfg["border_width"])
    ax.spines["top"].set_linewidth(cfg["border_width"])
    ax.spines["right"].set_linewidth(cfg["border_width"])
    ax.spines["top"].set_visible(cfg["show_top"])
    ax.spines["right"].set_visible(cfg["show_right"])
    if cfg["x_min"] is not None or cfg["x_max"] is not None:
        ax.set_xlim(left=cfg["x_min"], right=cfg["x_max"])
    if cfg["y_min"] is not None or cfg["y_max"] is not None:
        ax.set_ylim(bottom=cfg["y_min"], top=cfg["y_max"])
    if legend is None:
        legend = cfg["show_legend"]
    if cfg["show_legend"] and legend:
        ax.legend(frameon=False, loc=cfg["legend_loc"])


def residual_plot(fitted, residuals, xlabel="Fitted", ylabel="Residuals", title="Residuals vs fitted"):
    cfg = get_plot_cfg("Residual plot")
    fig, ax = plt.subplots(figsize=(cfg["fig_w"], cfg["fig_h"]))
    ax.scatter(fitted, residuals, color=cfg["primary_color"], s=cfg["marker_size"])
    ax.axhline(0, color="#111827", ls=cfg["aux_line_style"], lw=cfg["aux_line_width"])
    apply_ax_style(ax, title, xlabel, ylabel, plot_key="Residual plot")
    return fig


def qq_plot(residuals, title="Normal probability plot of residuals"):
    cfg = get_plot_cfg("Q-Q plot")
    fig, ax = plt.subplots(figsize=(cfg["fig_w"], cfg["fig_h"]))
    stats.probplot(residuals, dist="norm", plot=ax)
    if len(ax.lines) >= 2:
        ax.lines[0].set_marker("o")
        ax.lines[0].set_linestyle("None")
        ax.lines[0].set_color(cfg["primary_color"])
        ax.lines[0].set_markersize(max(3, cfg["marker_size"] / 12))
        ax.lines[1].set_color(cfg["secondary_color"])
        ax.lines[1].set_linestyle(cfg["aux_line_style"])
        ax.lines[1].set_linewidth(cfg["aux_line_width"])
    apply_ax_style(ax, title, "Theoretical quantiles", "Ordered residuals", plot_key="Q-Q plot")
    return fig


def tol_interval_normal(x, coverage=0.99, confidence=0.95):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return np.nan, np.nan, np.nan
    mean = np.mean(x)
    sd = np.std(x, ddof=1)
    nu = n - 1
    z_p = norm.ppf((1 + coverage) / 2)
    chi = stats.chi2.ppf(confidence, nu)
    if chi <= 0:
        return mean, np.nan, np.nan
    k = z_p * np.sqrt(nu * (1 + 1 / n) / chi)
    return mean, mean - k * sd, mean + k * sd


def tolerance_interval_normal(data, p=0.95, conf=0.95, two_sided=True):
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if len(data) < 2:
        return np.nan, np.nan, np.nan
    # current app uses only the two-sided version in downstream modules
    if two_sided:
        return tol_interval_normal(data, coverage=p, confidence=conf)
    n = len(data)
    mean = np.mean(data)
    sd = np.std(data, ddof=1)
    zp = norm.ppf(p)
    k = nct.ppf(conf, n - 1, np.sqrt(n) * zp) / np.sqrt(n)
    return mean, mean - k * sd, mean + k * sd


def draw_conf_ellipse(scores, ax, edgecolor=PRIMARY_COLOR, facecolor=None, plot_key="PCA score plot"):
    if scores.shape[0] < 3:
        return
    cov = np.cov(scores.T)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    q = stats.chi2.ppf(0.95, 2)
    width, height = 2 * np.sqrt(vals * q)
    cfg = get_plot_cfg(plot_key)
    lw = cfg["aux_line_width"]
    ls = cfg["aux_line_style"]
    fc = facecolor if facecolor is not None else edgecolor
    ell = Ellipse(
        xy=np.mean(scores, axis=0),
        width=width,
        height=height,
        angle=theta,
        edgecolor=edgecolor,
        facecolor=fc,
        alpha=0.15,
        lw=lw,
        ls=ls,
    )
    ax.add_patch(ell)

    if scores.shape[0] < 3:
        return
    cov = np.cov(scores[:, 0], scores[:, 1])
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    width, height = 2 * np.sqrt(5.991 * eigvals)
    ell = Ellipse(scores.mean(axis=0), width, height, angle=angle, fill=False, lw=2, edgecolor=edgecolor)
    ax.add_patch(ell)


def doe_formula(safe_factors, model_type="interaction"):
    terms = list(safe_factors)
    if model_type in ["interaction", "quadratic"]:
        for i in range(len(safe_factors)):
            for j in range(i + 1, len(safe_factors)):
                terms.append(f"{safe_factors[i]}:{safe_factors[j]}")
    if model_type == "quadratic":
        for f in safe_factors:
            terms.append(f"I({f}**2)")
    return "Response ~ " + " + ".join(terms)


# -------------------------------------------------
# App 01 Descriptive Statistics
# -------------------------------------------------
if app_selection == "01 - Descriptive Statistics":
    app_header("📊 App 01 - Descriptive Statistics", "Paste one or more numeric columns with headers. For one column, get a graphical summary. For multiple columns, choose a reference and a test column to compare.")

    data_input = st.text_area("Data (paste with headers)", height=220)
    decimals = st.slider("Decimals", 1, 8, DEFAULT_DECIMALS, key="desc_dec")
    alpha = st.slider("Significance level α", 0.001, 0.100, 0.050, 0.001, key="desc_alpha")
    mean_ci_conf = st.slider("Mean CI confidence (%)", 80, 99, 95, 1, key="desc_mean_ci")
    tol_cov = st.slider("Tolerance interval coverage (%)", 80, 99, 99, 1, key="desc_tol_cov")
    tol_conf = st.slider("Tolerance interval confidence (%)", 80, 99, 95, 1, key="desc_tol_conf")

    def _one_sample_summary(arr, label, ci_conf=0.95, tol_p=0.99, tol_confidence=0.95):
        arr = np.asarray(arr, dtype=float)
        n = len(arr)
        mean = np.mean(arr)
        sd = np.std(arr, ddof=1) if n > 1 else np.nan
        se = sd / np.sqrt(n) if n > 1 else np.nan
        tcrit = t.ppf(1 - (1 - ci_conf) / 2, n - 1) if n > 1 else np.nan
        ci_half = tcrit * se if n > 1 else np.nan
        _, tol_lower, tol_upper = tolerance_interval_normal(arr, p=tol_p, conf=tol_confidence, two_sided=True)
        ad_stat, ad_p = normal_ad(arr) if n >= 8 else (np.nan, np.nan)
        try:
            sh_stat, sh_p = stats.shapiro(arr) if 3 <= n <= 5000 else (np.nan, np.nan)
        except Exception:
            sh_stat, sh_p = (np.nan, np.nan)
        q1, med, q3 = np.percentile(arr, [25, 50, 75])
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        whisker_lower = np.min(arr[arr >= lower_fence]) if np.any(arr >= lower_fence) else np.min(arr)
        whisker_upper = np.max(arr[arr <= upper_fence]) if np.any(arr <= upper_fence) else np.max(arr)
        return {
            "label": label,
            "n": n,
            "sum": np.sum(arr),
            "mean": mean,
            "sd": sd,
            "var": np.var(arr, ddof=1) if n > 1 else np.nan,
            "min": np.min(arr),
            "q1": q1,
            "median": med,
            "q3": q3,
            "max": np.max(arr),
            "whisker_lower": whisker_lower,
            "whisker_upper": whisker_upper,
            "ci_half": ci_half,
            "ci_lower": mean - ci_half if pd.notna(ci_half) else np.nan,
            "ci_upper": mean + ci_half if pd.notna(ci_half) else np.nan,
            "tol_lower": tol_lower,
            "tol_upper": tol_upper,
            "ad_stat": ad_stat,
            "ad_p": ad_p,
            "shapiro_stat": sh_stat,
            "shapiro_p": sh_p,
        }

    def _f_test_equal_var(a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        v1 = np.var(a, ddof=1)
        v2 = np.var(b, ddof=1)
        if np.isnan(v1) or np.isnan(v2) or v1 == 0 or v2 == 0:
            return np.nan, np.nan
        if v1 >= v2:
            fstat = v1 / v2
            dfn, dfd = len(a) - 1, len(b) - 1
        else:
            fstat = v2 / v1
            dfn, dfd = len(b) - 1, len(a) - 1
        p = 2 * min(stats.f.cdf(fstat, dfn, dfd), 1 - stats.f.cdf(fstat, dfn, dfd))
        return fstat, min(p, 1.0)

    def _anova_two_groups(a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        n1, n2 = len(a), len(b)
        allv = np.concatenate([a, b])
        grand = np.mean(allv)
        m1, m2 = np.mean(a), np.mean(b)
        ss_between = n1 * (m1 - grand) ** 2 + n2 * (m2 - grand) ** 2
        ss_within = np.sum((a - m1) ** 2) + np.sum((b - m2) ** 2)
        ss_total = np.sum((allv - grand) ** 2)
        df_between = 1
        df_within = n1 + n2 - 2
        df_total = n1 + n2 - 1
        ms_between = ss_between / df_between
        ms_within = ss_within / df_within if df_within > 0 else np.nan
        f_stat = ms_between / ms_within if ms_within and ms_within > 0 else np.nan
        p = 1 - stats.f.cdf(f_stat, df_between, df_within) if pd.notna(f_stat) else np.nan
        return pd.DataFrame({
            "Source of Variation": ["Between Groups", "Within Groups", "Total"],
            "SS": [ss_between, ss_within, ss_total],
            "df": [df_between, df_within, df_total],
            "MS": [ms_between, ms_within, np.nan],
            "F": [f_stat, np.nan, np.nan],
            "P-Value": [p, np.nan, np.nan],
        }), ms_within, ss_between, ss_total

    def _acceptance_band(ref, test, alpha_level=0.05):
        ref = np.asarray(ref, dtype=float)
        test = np.asarray(test, dtype=float)
        n1, n2 = len(ref), len(test)
        m1 = np.mean(ref)
        v1 = np.var(ref, ddof=1)
        v2 = np.var(test, ddof=1)
        sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
        se_diff = np.sqrt((1 / n1 + 1 / n2) * sp2)
        tcrit = t.ppf(1 - alpha_level / 2, n1 + n2 - 2)
        return m1 - tcrit * se_diff, m1 + tcrit * se_diff

    
    def _graphical_summary_figure(stats_list, title, shaded_range=None, shaded_label=None):
        cfg = get_plot_cfg("Descriptive summary")
        if len(stats_list) > 1:
            colors = [cfg["primary_color"], cfg["secondary_color"], cfg["tertiary_color"]]
        else:
            colors = [cfg["primary_color"]]
        labels = [s["label"] for s in stats_list]

        mins, maxs = [], []
        for s in stats_list:
            for key in ["min", "whisker_lower", "q1", "mean", "tol_lower", "ci_lower"]:
                if pd.notna(s.get(key, np.nan)):
                    mins.append(s[key])
            for key in ["max", "whisker_upper", "q3", "mean", "tol_upper", "ci_upper"]:
                if pd.notna(s.get(key, np.nan)):
                    maxs.append(s[key])

        sr = None
        if shaded_range is not None:
            try:
                sr = np.asarray(shaded_range, dtype=float).ravel()
                if sr.size == 2 and np.all(np.isfinite(sr)):
                    mins.append(float(np.min(sr)))
                    maxs.append(float(np.max(sr)))
                else:
                    sr = None
            except Exception:
                sr = None

        x_min = min(mins) if mins else 0.0
        x_max = max(maxs) if maxs else 1.0
        pad = 0.08 * (x_max - x_min if x_max > x_min else 1)
        x_lo, x_hi = x_min - pad, x_max + pad

        fig, (ax, axr) = plt.subplots(
            1, 2,
            figsize=(max(cfg["fig_w"] * 1.95, 13), max(cfg["fig_h"] * 1.55, 7.2)),
            gridspec_kw={"width_ratios": [1.6, 1]}
        )

        density_y0 = 6.35
        row_centers = [5.25, 4.35, 3.45, 2.55, 1.65, 0.75]
        row_names = [
            "Whisker Min/Max",
            "Min/Max",
            "Mean ± 3SD",
            "IQR (Q1, Q3)",
            f"{tol_cov}%/{tol_conf}% Tol. Interval",
            f"{mean_ci_conf}% CI for Mean",
        ]

        if sr is not None:
            ax.axvspan(sr[0], sr[1], color=cfg["band_color"], alpha=0.18)
            ax.axvline(sr[0], color=cfg["secondary_color"], ls=cfg["aux_line_style"], lw=cfg["aux_line_width"])
            ax.axvline(sr[1], color=cfg["secondary_color"], ls=cfg["aux_line_style"], lw=cfg["aux_line_width"])
            if shaded_label:
                ax.text(
                    float(np.mean(sr)),
                    6.55,
                    shaded_label,
                    color=cfg["secondary_color"],
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2),
                )

        xgrid = np.linspace(x_lo, x_hi, 600)
        for i, s in enumerate(stats_list):
            arr = s["raw"]
            col = colors[i]
            if len(np.unique(arr)) > 1 and len(arr) >= 3:
                try:
                    dens = gaussian_kde(arr)(xgrid)
                    dens = dens / dens.max() * 0.85
                except Exception:
                    dens = np.zeros_like(xgrid)
            else:
                dens = np.zeros_like(xgrid)
            ax.plot(xgrid, density_y0 + dens, color=col, lw=cfg["line_width"], ls=cfg["line_style"])
            ax.hlines(density_y0, x_lo, x_hi, color="#111827", lw=0.8)

        offsets = [0.10, -0.10] if len(stats_list) > 1 else [0.0]
        for ridx, yc in enumerate(row_centers):
            ax.hlines(yc - 0.37, x_lo, x_hi, color="#d1d5db", lw=0.8)
            for i, s in enumerate(stats_list):
                yy = yc + offsets[i]
                col = colors[i]
                ms = max(4, cfg["marker_size"] / 12)
                if ridx == 0:
                    ax.hlines(yy, s["whisker_lower"], s["whisker_upper"], color=col, lw=cfg["line_width"], ls=cfg["line_style"])
                    ax.plot(s["median"], yy, 'o', color=col, ms=ms)
                elif ridx == 1:
                    ax.hlines(yy, s["min"], s["max"], color=col, lw=cfg["line_width"], ls=cfg["line_style"])
                    ax.plot(s["median"], yy, 'o', color=col, ms=ms)
                elif ridx == 2:
                    lo = s["mean"] - 3 * s["sd"] if pd.notna(s["sd"]) else np.nan
                    hi = s["mean"] + 3 * s["sd"] if pd.notna(s["sd"]) else np.nan
                    if pd.notna(lo) and pd.notna(hi):
                        ax.hlines(yy, lo, hi, color=col, lw=cfg["line_width"], ls=cfg["line_style"])
                    ax.plot(s["mean"], yy, 'o', color=col, ms=max(4.5, cfg["marker_size"] / 10))
                elif ridx == 3:
                    ax.hlines(yy, s["q1"], s["q3"], color=col, lw=cfg["line_width"] + 0.2, ls=cfg["line_style"])
                    ax.plot(s["median"], yy, 'o', color=col, ms=ms)
                elif ridx == 4:
                    ax.hlines(yy, s["tol_lower"], s["tol_upper"], color=col, lw=cfg["line_width"], ls=cfg["line_style"])
                    ax.plot(s["mean"], yy, 'o', color=col, ms=max(4.5, cfg["marker_size"] / 10))
                elif ridx == 5:
                    ax.hlines(yy, s["ci_lower"], s["ci_upper"], color=col, lw=cfg["line_width"], ls=cfg["line_style"])
                    ax.plot(s["mean"], yy, 'o', color=col, ms=max(4.5, cfg["marker_size"] / 10))

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(0.35, 6.95)
        ax.set_yticks([density_y0] + row_centers)
        ax.set_yticklabels(["Normal distribution"] + row_names)
        apply_ax_style(ax, title, "", "", legend=False, plot_key="Descriptive summary")
        ax.grid(axis="x", alpha=cfg["grid_alpha"])

        if cfg["show_legend"] and len(labels) > 1:
            handles = [plt.Line2D([0], [0], color=colors[i], marker='o', lw=cfg["line_width"], ls=cfg["line_style"], label=labels[i]) for i in range(len(labels))]
            ax.legend(handles=handles, frameon=False, loc=cfg["legend_loc"])

        axr.axis("off")
        axr.set_title("Graphical Summary with Descriptive Statistics", fontsize=13, weight="bold", pad=10)

        rows = []
        if len(stats_list) == 1:
            s = stats_list[0]
            rows = [
                ["Normality (AD), p-value", f"{s['ad_p']:.3f}" if pd.notna(s['ad_p']) else "-"],
                ["Normality (Shapiro), p-value", f"{s['shapiro_p']:.3f}" if pd.notna(s['shapiro_p']) else "-"],
                ["Mean", f"{s['mean']:.3f}"],
                ["SD", f"{s['sd']:.3f}"],
                ["N", f"{s['n']:.0f}"],
                ["Variance", f"{s['var']:.3f}"],
                ["Minimum", f"{s['min']:.3f}"],
                ["1st Quartile", f"{s['q1']:.3f}"],
                ["Median", f"{s['median']:.3f}"],
                ["3rd Quartile", f"{s['q3']:.3f}"],
                ["Maximum", f"{s['max']:.3f}"],
                [f"{tol_cov}%/{tol_conf}% Tol. Int. Lower", f"{s['tol_lower']:.3f}"],
                [f"{tol_cov}%/{tol_conf}% Tol. Int. Upper", f"{s['tol_upper']:.3f}"],
                [f"{mean_ci_conf}% LCI for Mean", f"{s['ci_lower']:.3f}"],
                [f"{mean_ci_conf}% UCI for Mean", f"{s['ci_upper']:.3f}"],
            ]
            x0, x1 = 0.02, 0.84
            y = 0.94
            axr.text(0.65, 0.98, s["label"], ha="center", va="top", fontsize=12, weight="bold")
            for label, val in rows:
                axr.text(x0, y, label, ha="left", va="center", fontsize=10.5, weight="bold")
                axr.text(x1, y, val, ha="right", va="center", fontsize=10.5)
                y -= 0.06
        else:
            s1, s2 = stats_list[:2]
            x0, x1, x2 = 0.02, 0.72, 0.95
            y = 0.94
            axr.text(x1, 0.98, s1["label"], ha="center", va="top", fontsize=12, weight="bold")
            axr.text(x2, 0.98, s2["label"], ha="center", va="top", fontsize=12, weight="bold")
            rows = [
                ["Normality (AD), p-value", f"{s1['ad_p']:.3f}" if pd.notna(s1['ad_p']) else "-", f"{s2['ad_p']:.3f}" if pd.notna(s2['ad_p']) else "-"],
                ["Mean", f"{s1['mean']:.3f}", f"{s2['mean']:.3f}"],
                ["SD", f"{s1['sd']:.3f}", f"{s2['sd']:.3f}"],
                ["N", f"{s1['n']:.3f}", f"{s2['n']:.3f}"],
                ["Variance", f"{s1['var']:.3f}", f"{s2['var']:.3f}"],
                ["Minimum", f"{s1['min']:.3f}", f"{s2['min']:.3f}"],
                ["1st Quartile", f"{s1['q1']:.3f}", f"{s2['q1']:.3f}"],
                ["Median", f"{s1['median']:.3f}", f"{s2['median']:.3f}"],
                ["3rd Quartile", f"{s1['q3']:.3f}", f"{s2['q3']:.3f}"],
                ["Maximum", f"{s1['max']:.3f}", f"{s2['max']:.3f}"],
                [f"{tol_cov}%/{tol_conf}% Tol. Int. Lower", f"{s1['tol_lower']:.3f}", f"{s2['tol_lower']:.3f}"],
                [f"{tol_cov}%/{tol_conf}% Tol. Int. Upper", f"{s1['tol_upper']:.3f}", f"{s2['tol_upper']:.3f}"],
                [f"{mean_ci_conf}% LCI for Mean", f"{s1['ci_lower']:.3f}", f"{s2['ci_lower']:.3f}"],
                [f"{mean_ci_conf}% UCI for Mean", f"{s1['ci_upper']:.3f}", f"{s2['ci_upper']:.3f}"],
            ]
            for label, v1, v2 in rows:
                axr.text(x0, y, label, ha="left", va="center", fontsize=10.5, weight="bold")
                axr.text(x1, y, v1, ha="center", va="center", fontsize=10.5)
                axr.text(x2, y, v2, ha="center", va="center", fontsize=10.5)
                y -= 0.06

        fig.tight_layout()
        return fig

    if data_input:
        df = parse_pasted_table(data_input, header=True)
        if df is None or df.empty:
            st.error("Could not parse the pasted data.")
        else:
            st.success(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")
            with st.expander("Preview data"):
                st.dataframe(df, use_container_width=True)

            numeric_cols = get_numeric_columns(df)
            if len(numeric_cols) == 0:
                st.error("No numeric columns were detected.")
            else:
                is_single = len(numeric_cols) == 1
                if is_single:
                    ref_col = numeric_cols[0]
                    test_col = None
                    st.info(f"Single numeric column detected: {ref_col}")
                else:
                    c1, c2 = st.columns(2)
                    with c1:
                        ref_col = st.selectbox("Reference column", numeric_cols, index=0)
                    with c2:
                        test_candidates = [c for c in numeric_cols if c != ref_col]
                        test_col = st.selectbox("Test column", test_candidates, index=0)

                if st.button("Run descriptive statistics", type="primary"):
                    ref = to_numeric(df[ref_col]).dropna().to_numpy()
                    if len(ref) < 3:
                        st.error("Reference column must contain at least 3 numeric values.")
                    else:
                        ref_stats = _one_sample_summary(ref, ref_col, ci_conf=mean_ci_conf / 100, tol_p=tol_cov / 100, tol_confidence=tol_conf / 100)
                        ref_stats["raw"] = ref

                        tables = {}
                        figs = {}

                        summary_tbl = pd.DataFrame({
                            "Groups": [ref_col],
                            "Count": [ref_stats["n"]],
                            "Sum": [ref_stats["sum"]],
                            "Average": [ref_stats["mean"]],
                            "StDev": [ref_stats["sd"]],
                            f"{mean_ci_conf}% CI ±": [ref_stats["ci_half"]],
                        })

                        normality_tbl = pd.DataFrame([
                            {"Test": "Anderson-Darling", "Group": ref_col, "Statistic": ref_stats["ad_stat"], "P-Value": ref_stats["ad_p"], "Comment": f"{'Normally distributed' if pd.notna(ref_stats['ad_p']) and ref_stats['ad_p'] >= alpha else 'Possible non-normality'} (p {'>=' if pd.notna(ref_stats['ad_p']) and ref_stats['ad_p'] >= alpha else '<'} {alpha:.3f})" if pd.notna(ref_stats['ad_p']) else "AD test not available"},
                            {"Test": "Shapiro-Wilk", "Group": ref_col, "Statistic": ref_stats["shapiro_stat"], "P-Value": ref_stats["shapiro_p"], "Comment": f"{'Normally distributed' if pd.notna(ref_stats['shapiro_p']) and ref_stats['shapiro_p'] >= alpha else 'Possible non-normality'} (p {'>=' if pd.notna(ref_stats['shapiro_p']) and ref_stats['shapiro_p'] >= alpha else '<'} {alpha:.3f})" if pd.notna(ref_stats['shapiro_p']) else "Shapiro test not available"},
                        ])

                        st.markdown("### Tables")
                        report_table(summary_tbl, "Summary of Means", decimals)
                        report_table(normality_tbl, "Normality Tests", decimals)
                        tables["Summary of Means"] = summary_tbl
                        tables["Normality Tests"] = normality_tbl

                        fig = _graphical_summary_figure([ref_stats], f"Graphical Summary: {ref_col}")
                        st.markdown("### Graphical Summary")
                        st.pyplot(fig)
                        figs["Graphical Summary"] = fig_to_png_bytes(fig)
                        plt.close(fig)

                        export_results(
                            prefix="descriptive_statistics_single",
                            report_title="Statistical Analysis Report",
                            module_name="Descriptive Statistics",
                            statistical_analysis="This one-sample descriptive analysis summarizes a single quantitative variable using count, sum, mean, standard deviation, quartiles, minimum and maximum. It also checks normality using Anderson-Darling and Shapiro-Wilk tests, computes a confidence interval for the mean, and calculates a normal-theory tolerance interval.",
                            offer_text="It offers a compact graphical and tabular summary for a single population, helping you assess central tendency, spread, distribution shape, normality, confidence bounds for the mean, and an interval that is expected to cover a chosen proportion of the population.",
                            python_tools="Python tools used in this analysis include pandas and numpy for data handling and descriptive calculations, scipy.stats and statsmodels for normality tests and interval calculations, matplotlib for the graphical summary, openpyxl for Excel export, and reportlab for the PDF-style report.",
                            table_map=tables,
                            figure_map=figs,
                            conclusion=f"The variable {ref_col} was summarized with descriptive statistics and normality checks. Review the graphical summary, the mean confidence interval, and the tolerance interval to judge both the center and the expected spread of the population.",
                            decimals=decimals,
                        )

                        if not is_single and test_col is not None:
                            test = to_numeric(df[test_col]).dropna().to_numpy()
                            if len(test) < 3:
                                st.error("Test column must contain at least 3 numeric values.")
                            else:
                                test_stats = _one_sample_summary(test, test_col, ci_conf=mean_ci_conf / 100, tol_p=tol_cov / 100, tol_confidence=tol_conf / 100)
                                test_stats["raw"] = test

                                summary_tbl = pd.DataFrame({
                                    "Groups": [ref_col, test_col],
                                    "Count": [ref_stats["n"], test_stats["n"]],
                                    "Sum": [ref_stats["sum"], test_stats["sum"]],
                                    "Average": [ref_stats["mean"], test_stats["mean"]],
                                    "StDev": [ref_stats["sd"], test_stats["sd"]],
                                    f"{mean_ci_conf}% CI ±": [ref_stats["ci_half"], test_stats["ci_half"]],
                                })

                                normality_tbl = pd.DataFrame([
                                    {"Test": "Anderson-Darling", "Group": ref_col, "Statistic": ref_stats["ad_stat"], "P-Value": ref_stats["ad_p"], "Comment": "Normally distributed" if pd.notna(ref_stats["ad_p"]) and ref_stats["ad_p"] >= alpha else "Possible non-normality"},
                                    {"Test": "Anderson-Darling", "Group": test_col, "Statistic": test_stats["ad_stat"], "P-Value": test_stats["ad_p"], "Comment": "Normally distributed" if pd.notna(test_stats["ad_p"]) and test_stats["ad_p"] >= alpha else "Possible non-normality"},
                                    {"Test": "Shapiro-Wilk", "Group": ref_col, "Statistic": ref_stats["shapiro_stat"], "P-Value": ref_stats["shapiro_p"], "Comment": "Normally distributed" if pd.notna(ref_stats["shapiro_p"]) and ref_stats["shapiro_p"] >= alpha else "Possible non-normality"},
                                    {"Test": "Shapiro-Wilk", "Group": test_col, "Statistic": test_stats["shapiro_stat"], "P-Value": test_stats["shapiro_p"], "Comment": "Normally distributed" if pd.notna(test_stats["shapiro_p"]) and test_stats["shapiro_p"] >= alpha else "Possible non-normality"},
                                ])

                                f_stat, f_p = _f_test_equal_var(ref, test)
                                lev_stat, lev_p = stats.levene(ref, test, center="mean")
                                eqvar_tbl = pd.DataFrame([
                                    {"Test": "F Test", "Statistic": f_stat, "P-Value": f_p, "Comment": "Equal variances" if pd.notna(f_p) and f_p >= alpha else "Unequal variances"},
                                    {"Test": "Levene's Test (mean)", "Statistic": lev_stat, "P-Value": lev_p, "Comment": "Equal variances" if lev_p >= alpha else "Unequal variances"},
                                ])

                                t_eq = stats.ttest_ind(ref, test, equal_var=True)
                                t_welch = stats.ttest_ind(ref, test, equal_var=False)
                                mw = stats.mannwhitneyu(ref, test, alternative="two-sided")
                                comp_tbl = pd.DataFrame([
                                    {"Test": "Student t-test", "Statistic": t_eq.statistic, "P-Value": t_eq.pvalue, "Comment": "Difference in means" if t_eq.pvalue < alpha else "No evidence of difference in means"},
                                    {"Test": "Welch t-test", "Statistic": t_welch.statistic, "P-Value": t_welch.pvalue, "Comment": "Difference in means" if t_welch.pvalue < alpha else "No evidence of difference in means"},
                                    {"Test": "Mann-Whitney U", "Statistic": mw.statistic, "P-Value": mw.pvalue, "Comment": "Difference in distributions" if mw.pvalue < alpha else "No evidence of distributional difference"},
                                ])

                                anova_tbl, mse, ss_between, ss_total = _anova_two_groups(ref, test)
                                rsq = ss_between / ss_total if ss_total > 0 else np.nan
                                rsq_adj = 1 - (1 - rsq) * ((len(ref) + len(test) - 1) / (len(ref) + len(test) - 2 - 0)) if (len(ref) + len(test) - 2) > 0 and pd.notna(rsq) else np.nan
                                model_tbl = pd.DataFrame({"Pooled SD": [np.sqrt(mse)], "R²": [rsq], "R² (adj)": [rsq_adj]})

                                shaded = _acceptance_band(ref, test, alpha_level=alpha)
                                graph_tbl = pd.DataFrame({
                                    "Reference": [ref_col],
                                    "Reference Mean": [ref_stats["mean"]],
                                    "Acceptance Lower": [shaded[0]],
                                    "Acceptance Upper": [shaded[1]],
                                    "Test Mean": [test_stats["mean"]],
                                })

                                st.markdown("### Comparison Tables")
                                report_table(summary_tbl, "Summary of Means", decimals)
                                report_table(normality_tbl, "Normality Tests", decimals)
                                report_table(eqvar_tbl, "Equal Variances Test", decimals)
                                report_table(anova_tbl, "ANOVA", decimals)
                                report_table(model_tbl, "Model Summary (ANOVA)", decimals)
                                report_table(comp_tbl, "Mean / Distribution Comparison", decimals)

                                tables = {
                                    "Summary of Means": summary_tbl,
                                    "Normality Tests": normality_tbl,
                                    "Equal Variances Test": eqvar_tbl,
                                    "ANOVA": anova_tbl,
                                    "Model Summary (ANOVA)": model_tbl,
                                    "Mean / Distribution Comparison": comp_tbl,
                                    "Acceptance Range": graph_tbl,
                                }

                                shade_label = f"p > {alpha:.3f} zone around {ref_col} mean"
                                fig = _graphical_summary_figure([ref_stats, test_stats], f"Graphical Summary: {ref_col} vs {test_col}", shaded_range=shaded, shaded_label=shade_label)
                                st.markdown("### Graphical Summary")
                                info_box(f"The shaded area is centered on the reference mean and spans the range in which the test mean would remain within the two-sided t-test acceptance zone at α = {alpha:.3f}, using the pooled within-group variance.")
                                st.pyplot(fig)
                                figs = {"Graphical Summary": fig_to_png_bytes(fig)}
                                plt.close(fig)

                                equal_var_msg = "equal variances" if lev_p >= alpha else "unequal variances"
                                conclusion = (
                                    f"{ref_col} was treated as the reference and {test_col} as the test population. "
                                    f"The shaded region in the graph shows the approximate range around the reference mean that would keep the test mean non-significant at α = {alpha:.3f} under the pooled-variance t-test framework. "
                                    f"The variance assessment suggested {equal_var_msg}. Review the Student/Welch and Mann-Whitney results together with the graphical summary before concluding whether the two populations differ in mean or broader distribution."
                                )

                                export_results(
                                    prefix="descriptive_statistics_comparison",
                                    report_title="Statistical Analysis Report",
                                    module_name="Descriptive Statistics / Two-Group Comparison",
                                    statistical_analysis="This analysis summarizes each selected population using descriptive statistics, normality tests, confidence intervals for the mean, and normal-theory tolerance intervals. When both a reference and a test column are selected, it also evaluates equality of variances and compares the two populations using Student's t-test, Welch's t-test, Mann-Whitney U, and a two-group ANOVA summary. The shaded band in the graphical summary is centered on the reference mean and represents the approximate region in which the test mean would remain non-significant at the chosen alpha level using the pooled within-group error term.",
                                    offer_text="It offers a report-ready way to summarize one population or compare two populations for difference in means, variability, and overall distribution. It also shows whether the test mean stays within the practical acceptance region around the reference mean, which is useful when you want a quick visual link between mean separation and the p-value threshold.",
                                    python_tools="Python tools used in this analysis include pandas and numpy for cleaning and calculations, scipy.stats for t-tests, Levene, Shapiro-Wilk, F distributions, and tolerance-related statistics, statsmodels for Anderson-Darling normality testing, matplotlib for the graphical summary, openpyxl for Excel export, and reportlab for the PDF-style report.",
                                    table_map=tables,
                                    figure_map=figs,
                                    conclusion=conclusion,
                                    decimals=decimals,
                                )



# -------------------------------------------------
# App 02 Regression Intervals
# -------------------------------------------------
elif app_selection == "02 - Regression Intervals":
    app_header("📈 App 02 - Regression Intervals", "Linear regression with CI / PI / both, one-sided or two-sided bands, prediction points, and spec-limit crossing.")

    left, right = st.columns([1.45, 1])
    with left:
        xy_input = st.text_area("Paste X and Y data (two Excel columns, with or without headers)", height=220)
    with right:
        x_pred_text = st.text_area("Predict X (optional)", height=110, placeholder="Paste X values to predict")

    if xy_input:
        try:
            data_df, x_label_detected, y_label_detected = parse_xy(xy_input)

            st.markdown("### Options")
            c1, c2, c3 = st.columns([1, 1, 1.2])
            with c1:
                interval_mode = st.selectbox("Interval", ["ci", "pi", "both"], format_func=lambda x: {"ci":"CI", "pi":"PI", "both":"Both"}[x])
            with c2:
                side_mode = st.selectbox("Side", ["upper", "lower", "two-sided"], format_func=lambda x: {"upper":"Upper", "lower":"Lower", "two-sided":"Two-sided"}[x])
            with c3:
                confidence = st.slider("Confidence", 0.80, 0.99, 0.95, 0.01, format="%.2f")

            c4, c5, c6, c7 = st.columns([1.2, 1.1, 1.1, 0.9])
            with c4:
                plot_title = st.text_input("Title", value="")
            with c5:
                xlabel = st.text_input("X label", value=x_label_detected or "X")
            with c6:
                ylabel = st.text_input("Y label", value=y_label_detected or "Y")
            with c7:
                point_label = st.text_input("Point label", value="Data")

            c8, c9, c10, c11 = st.columns([0.9, 0.9, 0.9, 0.9])
            with c8:
                y_suffix = st.text_input("Y suffix", value="%")
            with c9:
                x_min_txt = st.text_input("X min", value="")
            with c10:
                default_xmax = str(max(40.0, float(max(data_df["x"].max(), parse_x_values(x_pred_text).max()) if len(parse_x_values(x_pred_text)) else float(data_df["x"].max()) * 1.15)))
                x_max_txt = st.text_input("X max", value=default_xmax)
            with c11:
                decimals = st.slider("Decimals", 1, 8, DEFAULT_DECIMALS, key="reg_dec_refined")

            st.markdown("### Specification / crossing")
            s1, s2, s3, s4 = st.columns([0.9, 1, 1, 1.2])
            with s1:
                spec_enabled = st.checkbox("Use spec limit", value=True)
            with s2:
                spec_value_txt = st.text_input("Spec value", value="3.0", disabled=not spec_enabled)
            with s3:
                spec_label = st.text_input("Spec label", value="US", disabled=not spec_enabled)
            with s4:
                crossing_on = st.selectbox(
                    "Crossing on",
                    ["auto", "fit", "ci_upper", "ci_lower", "pi_upper", "pi_lower"],
                    format_func=lambda x: {
                        "auto": "Auto", "fit": "Fit", "ci_upper": "CI upper", "ci_lower": "CI lower", "pi_upper": "PI upper", "pi_lower": "PI lower"
                    }[x],
                    disabled=not spec_enabled,
                )

            if st.button("Run regression analysis", type="primary"):
                pred_x = parse_x_values(x_pred_text)
                x_all_max = data_df["x"].max()
                if len(pred_x) > 0:
                    x_all_max = max(x_all_max, np.max(pred_x))

                def parse_optional_float(txt):
                    txt = str(txt).strip()
                    return None if txt == "" else float(txt)

                x_min = parse_optional_float(x_min_txt)
                x_max = parse_optional_float(x_max_txt)
                if x_min is None:
                    x_min = min(0.0, float(data_df["x"].min()))
                if x_max is None:
                    x_max = x_all_max * 1.15 if x_all_max != 0 else 1.0
                if x_max <= x_min:
                    raise ValueError("X max must be greater than X min.")

                grid_x = np.linspace(x_min, x_max, 500)
                model = reg_fit_linear_model(data_df["x"], data_df["y"])
                grid_df = reg_predict_with_intervals(model, grid_x, confidence=confidence, side=side_mode)

                fig_main, crossing_x = plot_regression_advanced(
                    data_df=data_df,
                    model=model,
                    grid_df=grid_df,
                    confidence=confidence,
                    interval=interval_mode,
                    side=side_mode,
                    title=plot_title,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    point_label=point_label,
                    y_suffix=y_suffix,
                    spec_enabled=spec_enabled,
                    spec_limit=parse_optional_float(spec_value_txt) if spec_enabled else None,
                    spec_label=spec_label,
                    crossing_on=crossing_on,
                )
                st.pyplot(fig_main)

                summary_tbl = pd.DataFrame({
                    "Intercept": [model["intercept"]],
                    "Slope": [model["slope"]],
                    "R²": [model["r2"]],
                    "Residual SD (s)": [model["s"]],
                    "Degrees of Freedom": [model["df"]],
                })
                if crossing_x is not None:
                    summary_tbl["Crossing Point"] = [crossing_x]
                report_table(summary_tbl, "Regression model summary", decimals)

                report_table(data_df.rename(columns={"x": "X Value", "y": "Actual Y"}), "Table 1: Parsed input data", decimals)

                new_pred_x = np.setdiff1d(pred_x, data_df["x"].to_numpy()) if len(pred_x) > 0 else np.array([])
                if len(new_pred_x) > 0:
                    new_pts_df = pd.DataFrame({"x": new_pred_x, "y": np.nan})
                    combined_pts_df = pd.concat([data_df[["x", "y"]], new_pts_df], ignore_index=True)
                else:
                    combined_pts_df = data_df[["x", "y"]].copy()
                combined_pts_df = combined_pts_df.sort_values("x").reset_index(drop=True)
                unique_x = combined_pts_df["x"].unique()
                intervals_df = reg_predict_with_intervals(model, unique_x, confidence=confidence, side=side_mode)
                final_table_df = pd.merge(combined_pts_df, intervals_df, on="x", how="left")
                final_table_df = final_table_df[[c for c in ["x", "y", "fit", "ci_lower", "ci_upper", "pi_lower", "pi_upper"] if c in final_table_df.columns]]
                final_table_df.columns = ["X Value", "Actual Y", "Fitted Y", "Lower CI", "Upper CI", "Lower PI", "Upper PI"]
                report_table(final_table_df, "Table 2: Fitted values and intervals", decimals)

                fig_res = residual_plot(model["fitted"], model["resid"], xlabel="Fitted values", ylabel="Residuals", title="Residuals vs fitted")
                st.pyplot(fig_res)
                fig_qq = qq_plot(model["resid"], title="Normal probability plot of regression residuals")
                st.pyplot(fig_qq)

                crossing_text = f" A crossing with the selected specification limit was identified at x = {crossing_x:.{decimals}f}." if crossing_x is not None else " No crossing with the selected specification limit was identified in the displayed X range."
                conclusion = (
                    f"A simple linear regression was fitted to {len(data_df)} observations. "
                    f"The fitted equation was y = {model['intercept']:.{decimals}f} + {model['slope']:.{decimals}f} × x, "
                    f"with R² = {model['r2']:.{decimals}f} and residual SD = {model['s']:.{decimals}f}. "
                    f"The analysis displayed {('confidence intervals' if interval_mode == 'ci' else 'prediction intervals' if interval_mode == 'pi' else 'both confidence and prediction intervals')} using a {side_mode} setting at {confidence:.0%} confidence." + crossing_text
                )
                export_results(
                    prefix="regression_intervals_refined",
                    report_title="Statistical Analysis Report",
                    module_name="Regression Intervals",
                    statistical_analysis=(
                        "A simple linear regression model was fitted to the pasted X and Y data using ordinary least squares. "
                        "The analysis estimates the intercept and slope of the linear relationship, summarizes goodness of fit using R² and residual standard deviation, "
                        "and then calculates confidence intervals for the fitted mean response and prediction intervals for future observations. "
                        "The module also allows one-sided or two-sided interval construction and can estimate a crossing point against a user-defined specification limit."
                    ),
                    offer_text=(
                        "This analysis offers a practical way to evaluate linear trends over X, quantify the expected response at user-selected X values, "
                        "compare observed responses with fitted values, and distinguish between uncertainty in the average response and variability expected for individual future measurements. "
                        "When a specification limit is supplied, it can also estimate where the fitted curve or selected interval band crosses that limit."
                    ),
                    python_tools=(
                        "Python tools used here include pandas for parsing pasted Excel-style data, numpy for matrix algebra and grid generation, "
                        "scipy.stats for t-based interval calculations, matplotlib for the fitted-curve, residual, and normal probability plots, "
                        "openpyxl for Excel export, and reportlab for the PDF-style report."
                    ),
                    table_map={
                        "Regression Model Summary": summary_tbl,
                        "Parsed Input Data": data_df.rename(columns={"x": "X Value", "y": "Actual Y"}),
                        "Fitted Values and Intervals": final_table_df,
                    },
                    figure_map={
                        "Regression plot": fig_to_png_bytes(fig_main),
                        "Residuals vs fitted": fig_to_png_bytes(fig_res),
                        "Normal probability plot": fig_to_png_bytes(fig_qq),
                    },
                    conclusion=conclusion,
                    decimals=decimals,
                )
        except Exception as e:
            st.error(str(e))


# App 03 Shelf Life Estimator
# -------------------------------------------------
elif app_selection == "03 - Shelf Life Estimator":
    app_header("⏳ App 03 - Shelf Life Estimator", "Paste stability data, choose lower or upper specification, and estimate shelf life from fit, CI, or PI crossing.")

    def sl_predict_local(model, x_values, confidence=0.95, one_sided=True):
        x_values = np.asarray(x_values, dtype=float).ravel()
        Xg = np.column_stack([np.ones(len(x_values)), x_values])
        beta = np.array([model["intercept"], model["slope"]])
        fit = Xg @ beta
        h = np.einsum("ij,jk,ik->i", Xg, model["XtX_inv"], Xg)
        se_mean = model["s"] * np.sqrt(h)
        se_pred = model["s"] * np.sqrt(1 + h)
        alpha = 1 - confidence
        tcrit = t.ppf(confidence, model["df"]) if one_sided else t.ppf(1 - alpha / 2, model["df"])
        return pd.DataFrame({
            "x": x_values,
            "fit": fit,
            "ci_lower": fit - tcrit * se_mean,
            "ci_upper": fit + tcrit * se_mean,
            "pi_lower": fit - tcrit * se_pred,
            "pi_upper": fit + tcrit * se_pred,
        })

    def sl_find_crossing_local(xv, yv, limit):
        xv = np.asarray(xv, dtype=float)
        yv = np.asarray(yv, dtype=float)
        d = yv - limit
        if len(d) == 0:
            return None
        if d[0] == 0:
            return float(xv[0])
        for i in range(len(d) - 1):
            if d[i] == 0:
                return float(xv[i])
            if d[i] * d[i + 1] < 0:
                x1, x2 = xv[i], xv[i + 1]
                y1, y2 = yv[i], yv[i + 1]
                if y2 == y1:
                    return float(x1)
                return float(x1 + (limit - y1) * (x2 - x1) / (y2 - y1))
        return None

    def sl_get_bound_column_local(spec_side, shelf_basis):
        if shelf_basis == "fit":
            return "fit"
        if shelf_basis == "ci":
            return "ci_lower" if spec_side == "lower" else "ci_upper"
        if shelf_basis == "pi":
            return "pi_lower" if spec_side == "lower" else "pi_upper"
        raise ValueError("Invalid shelf-life basis.")

    def sl_plot_local(data_df, grid_df, spec_side, spec_limit, shelf_basis, show_ci_band, show_pi_band,
                      title, xlabel, ylabel, point_label, y_suffix, spec_label):
        x = data_df["x"].to_numpy()
        y = data_df["y"].to_numpy()
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

        if show_pi_band:
            ax.fill_between(grid_df["x"], grid_df["pi_lower"], grid_df["pi_upper"], color=SECONDARY_COLOR, alpha=0.10, label="PI band")
            ax.plot(grid_df["x"], grid_df["pi_lower"], color=SECONDARY_COLOR, lw=1.0, ls=(0, (4, 4)))
            ax.plot(grid_df["x"], grid_df["pi_upper"], color=SECONDARY_COLOR, lw=1.0, ls=(0, (4, 4)))

        if show_ci_band:
            ax.fill_between(grid_df["x"], grid_df["ci_lower"], grid_df["ci_upper"], color=BAND_COLOR, alpha=0.15, label="CI band")
            ax.plot(grid_df["x"], grid_df["ci_lower"], color=BAND_COLOR, lw=1.0, ls="--")
            ax.plot(grid_df["x"], grid_df["ci_upper"], color=BAND_COLOR, lw=1.0, ls="--")

        ax.scatter(x, y, color=PRIMARY_COLOR, s=50, alpha=0.85, label=point_label, zorder=3)
        ax.plot(grid_df["x"], grid_df["fit"], color="#2c3e50", lw=2, label="Fitted line")

        bound_col = sl_get_bound_column_local(spec_side, shelf_basis)
        bound_color = {"fit": "#2c3e50", "ci": BAND_COLOR, "pi": SECONDARY_COLOR}[shelf_basis]
        bound_label = {
            "fit": "Shelf-life line (fit)",
            "ci": f"Shelf-life bound ({'lower' if spec_side == 'lower' else 'upper'} CI)",
            "pi": f"Shelf-life bound ({'lower' if spec_side == 'lower' else 'upper'} PI)",
        }[shelf_basis]
        if shelf_basis != "fit":
            ax.plot(grid_df["x"], grid_df[bound_col], color=bound_color, lw=2.5, label=bound_label)

        ax.axhline(spec_limit, color="#27ae60", ls="--", lw=1.5, label=f"Limit ({spec_label})")
        shelf_life = sl_find_crossing_local(grid_df["x"].to_numpy(), grid_df[bound_col].to_numpy(), spec_limit)
        if shelf_life is not None:
            ax.axvline(shelf_life, color="#27ae60", ls=":", lw=1.5)

        xmin = float(grid_df["x"].min())
        xmax = float(grid_df["x"].max())
        ymax_data = max(np.max(y), np.max(grid_df["fit"]), np.max(grid_df["ci_upper"]), np.max(grid_df["pi_upper"]))
        ymin_data = min(np.min(y), np.min(grid_df["fit"]), np.min(grid_df["ci_lower"]), np.min(grid_df["pi_lower"]))
        pad = 0.03 * ((ymax_data - ymin_data) if ymax_data > ymin_data else 1)

        ax.text(
            xmin + (xmax - xmin) * 0.02,
            spec_limit + pad,
            f"{spec_label} = {spec_limit:.2f}{y_suffix}",
            ha="left", va="bottom", fontsize=11, color="#27ae60", weight="bold",
            bbox=dict(facecolor="white", alpha=0.82, edgecolor="none", pad=3),
        )
        if shelf_life is not None:
            ax.text(
                shelf_life,
                ymin_data + pad,
                f" {shelf_life:.2f} ",
                ha="right", va="bottom", fontsize=11, color="#27ae60", weight="bold",
                bbox=dict(facecolor="white", alpha=0.82, edgecolor="none", pad=2),
            )

        if y_suffix:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, pos: f"{v:.1f}{y_suffix}"))

        if not str(title).strip():
            side_txt = "Lower Spec" if spec_side == "lower" else "Upper Spec"
            basis_txt = {"fit": "Fit", "ci": "Confidence Bound", "pi": "Prediction Bound"}[shelf_basis]
            title = f"Shelf Life Estimator ({side_txt}, {basis_txt})"

        apply_ax_style(ax, title, xlabel, ylabel, legend=True)
        return fig, shelf_life, bound_col

    c1, c2 = st.columns([1.35, 1])
    with c1:
        xy_input = st.text_area("Paste Time and Response data (with or without headers)", height=220)
    with c2:
        pred_x_text = st.text_area("Predict future X values (optional)", value="30\n36\n48", height=120)
        decimals = st.slider("Decimals", 1, 8, DEFAULT_DECIMALS, key="sl_dec")

    r1c1, r1c2, r1c3 = st.columns([1, 1, 1.15])
    with r1c1:
        spec_side = st.selectbox("Spec side", ["lower", "upper"], format_func=lambda x: "Lower spec" if x == "lower" else "Upper spec")
    with r1c2:
        shelf_basis = st.selectbox("Shelf-life on", ["ci", "pi", "fit"], format_func=lambda x: {"ci": "Confidence bound", "pi": "Prediction bound", "fit": "Fit line"}[x])
    with r1c3:
        confidence = st.slider("Confidence", 0.80, 0.99, 0.95, 0.01, format="%.2f")

    r2c1, r2c2, r2c3, r2c4 = st.columns([1, 1, 1, 1])
    with r2c1:
        spec_value_txt = st.text_input("Spec value", value="90")
    with r2c2:
        spec_label = st.text_input("Spec label", value="Spec")
    with r2c3:
        show_ci_band = st.checkbox("Show CI band", value=True)
    with r2c4:
        show_pi_band = st.checkbox("Show PI band", value=False)

    plot_title = st.text_input("Title", value="")

    r3c1, r3c2, r3c3, r3c4 = st.columns([1, 1, 1, 0.8])
    with r3c1:
        xlabel_override = st.text_input("X label", value="")
    with r3c2:
        ylabel_override = st.text_input("Y label", value="")
    with r3c3:
        point_label = st.text_input("Point label", value="Data")
    with r3c4:
        y_suffix = st.text_input("Y suffix", value="%")

    r4c1, r4c2 = st.columns([1, 1])
    with r4c1:
        x_min_txt = st.text_input("X min", value="")
    with r4c2:
        x_max_txt = st.text_input("X max", value="")

    if xy_input:
        try:
            data_df, x_label_from_header, y_label_from_header = parse_xy(xy_input)
            xlabel = xlabel_override.strip() or x_label_from_header or "Time"
            ylabel = ylabel_override.strip() or y_label_from_header or "Response"
            pred_x = parse_x_values(pred_x_text)
            spec_limit = parse_optional_float(spec_value_txt)
            if spec_limit is None:
                raise ValueError("Enter a valid specification value.")

            x_data_max = float(data_df["x"].max())
            x_future_max = float(np.max(pred_x)) if len(pred_x) > 0 else x_data_max
            x_min = parse_optional_float(x_min_txt)
            x_max = parse_optional_float(x_max_txt)
            if x_min is None:
                x_min = min(0.0, float(data_df["x"].min()))
            if x_max is None:
                x_max = max(x_data_max * 3, x_future_max * 1.15, x_data_max + 12)
            if x_max <= x_min:
                raise ValueError("X max must be greater than X min.")

            model = fit_linear(data_df["x"], data_df["y"])
            grid_x = np.linspace(x_min, x_max, 600)
            grid_df = sl_predict_local(model, grid_x, confidence=confidence, one_sided=True)

            fig_main, shelf_life, bound_col = sl_plot_local(
                data_df=data_df,
                grid_df=grid_df,
                spec_side=spec_side,
                spec_limit=spec_limit,
                shelf_basis=shelf_basis,
                show_ci_band=show_ci_band,
                show_pi_band=show_pi_band,
                title=plot_title,
                xlabel=xlabel,
                ylabel=ylabel,
                point_label=point_label,
                y_suffix=y_suffix,
                spec_label=spec_label,
            )
            st.pyplot(fig_main)

            summary_tbl = pd.DataFrame({
                "Intercept": [model["intercept"]],
                "Slope": [model["slope"]],
                "R²": [model["r2"]],
                "Residual SD (s)": [model["s"]],
                "Degrees of Freedom": [model["df"]],
                "Shelf-life basis": [bound_col],
                "Confidence": [f"{confidence:.0%} one-sided"],
                "Estimated Shelf Life": [np.nan if shelf_life is None else shelf_life],
            })
            report_table(summary_tbl, "Shelf-life estimation summary", decimals)
            report_table(data_df.rename(columns={"x": x_label_from_header, "y": y_label_from_header}), "Table 1: Parsed data", decimals)

            new_pred_x = np.setdiff1d(pred_x, data_df["x"].to_numpy()) if len(pred_x) > 0 else np.array([])
            if len(new_pred_x) > 0:
                new_pts_df = pd.DataFrame({"x": new_pred_x, "y": np.nan})
                combined_pts_df = pd.concat([data_df[["x", "y"]], new_pts_df], ignore_index=True)
            else:
                combined_pts_df = data_df[["x", "y"]].copy()
            combined_pts_df = combined_pts_df.sort_values("x").reset_index(drop=True)
            unique_x = combined_pts_df["x"].unique()
            intervals_df = sl_predict_local(model, unique_x, confidence=confidence, one_sided=True)
            final_table_df = pd.merge(combined_pts_df, intervals_df, on="x", how="left")
            final_table_df = final_table_df[[c for c in ["x", "y", "fit", "ci_lower", "ci_upper", "pi_lower", "pi_upper"] if c in final_table_df.columns]]
            final_table_df.columns = [xlabel, f"Actual {ylabel}", f"Fitted {ylabel}", "Lower CI", "Upper CI", "Lower PI", "Upper PI"]
            report_table(final_table_df, "Table 2: Fitted values and one-sided bounds", decimals)

            fig_res = residual_plot(model["fitted"], model["resid"], xlabel="Fitted values", ylabel="Residuals", title="Residuals vs fitted")
            st.pyplot(fig_res)
            fig_qq = qq_plot(model["resid"], title="Normal probability plot of stability residuals")
            st.pyplot(fig_qq)

            conclusion = (
                f"A linear regression was fitted to the stability data and one-sided bounds were calculated at {confidence:.0%} confidence. "
                f"Shelf life was estimated using the {bound_col} crossing against the {spec_label} limit of {spec_limit:.{decimals}f}{y_suffix}. "
                + (f"The estimated shelf life was {shelf_life:.{decimals}f}." if shelf_life is not None else "No crossing was found within the plotted range.")
            )
            export_results(
                prefix="shelf_life_refined",
                report_title="Statistical Analysis Report",
                module_name="Shelf Life Estimator",
                statistical_analysis=(
                    "A simple linear regression model was fitted to the response-versus-time stability data using ordinary least squares. "
                    "One-sided confidence and prediction bounds were then derived from the fitted model. Shelf life was estimated as the earliest time at which the selected basis "
                    "(fit line, confidence bound, or prediction bound) crossed the chosen lower or upper specification limit."
                ),
                offer_text=(
                    "This analysis offers a practical way to quantify the stability trend, visualize fitted performance and uncertainty, project future responses, and obtain a conservative shelf-life estimate based on either the fit, a confidence bound, or a prediction bound."
                ),
                python_tools=(
                    "Python tools used here include pandas for parsing pasted Excel-style stability data, numpy for matrix calculations and prediction grids, scipy.stats for one-sided t-based bounds, matplotlib for the shelf-life, residual, and normal probability plots, openpyxl for Excel export, and reportlab for the PDF-style report."
                ),
                table_map={
                    "Shelf-life Summary": summary_tbl,
                    "Parsed Data": data_df.rename(columns={"x": x_label_from_header, "y": y_label_from_header}),
                    "Fitted Values and One-Sided Bounds": final_table_df,
                },
                figure_map={
                    "Shelf-life plot": fig_to_png_bytes(fig_main),
                    "Residuals vs fitted": fig_to_png_bytes(fig_res),
                    "Normal probability plot": fig_to_png_bytes(fig_qq),
                },
                conclusion=conclusion,
                decimals=decimals,
            )
        except Exception as e:
            st.error(str(e))

elif app_selection == "04 - Dissolution Comparison (f2)":
    app_header("💊 App 04 - Dissolution Comparison (f2)", "Paste reference and test profiles and compare them with the similarity factor f₂.")
    c1, c2 = st.columns(2)
    with c1:
        ref_text = st.text_area("Reference profile table", height=220)
    with c2:
        test_text = st.text_area("Test profile table", height=220)
    decimals = st.slider("Decimals", 1, 8, 2, key="f2_dec")
    if ref_text and test_text:
        try:
            ref = parse_pasted_table(ref_text, header=True)
            test = parse_pasted_table(test_text, header=True)
            ref.iloc[:, 0] = to_numeric(ref.iloc[:, 0])
            test.iloc[:, 0] = to_numeric(test.iloc[:, 0])
            ref = ref.dropna().reset_index(drop=True)
            test = test.dropna().reset_index(drop=True)
            if len(ref) == 0 or len(test) == 0 or not np.allclose(ref.iloc[:, 0], test.iloc[:, 0]):
                raise ValueError("Reference and test must have matching time points in the first column.")
            time_col = ref.columns[0]
            time = to_numeric(ref.iloc[:, 0])
            ref_vals = ref.iloc[:, 1:].apply(to_numeric)
            test_vals = test.iloc[:, 1:].apply(to_numeric)
            summary = pd.DataFrame({
                time_col: time,
                "Reference Mean": ref_vals.mean(axis=1),
                "Reference SD": ref_vals.std(axis=1, ddof=1),
                "Reference CV (%)": ref_vals.std(axis=1, ddof=1) / ref_vals.mean(axis=1) * 100,
                "Test Mean": test_vals.mean(axis=1),
                "Test SD": test_vals.std(axis=1, ddof=1),
                "Test CV (%)": test_vals.std(axis=1, ddof=1) / test_vals.mean(axis=1) * 100,
            })
            summary["Absolute Difference"] = (summary["Reference Mean"] - summary["Test Mean"]).abs()
            summary["Squared Difference"] = (summary["Reference Mean"] - summary["Test Mean"]) ** 2
            f2 = 50 * np.log10((1 + summary["Squared Difference"].mean()) ** -0.5 * 100)
            verdict = "Similar" if f2 >= 50 else "Not Similar"
            assess = pd.DataFrame({"Similarity Factor f₂": [f2], "Conclusion": [verdict]})
            report_table(summary, "Dissolution profile summary", decimals)
            report_table(assess, "Overall f₂ assessment", decimals)

            fig_main, ax = plt.subplots(figsize=(FIG_W, FIG_H))
            ax.errorbar(time, summary["Reference Mean"], yerr=summary["Reference SD"], fmt='o-', color=PRIMARY_COLOR, lw=2, capsize=4, label="Reference")
            ax.errorbar(time, summary["Test Mean"], yerr=summary["Test SD"], fmt='s-', color=SECONDARY_COLOR, lw=2, capsize=4, label="Test")
            apply_ax_style(ax, f"Dissolution profiles (f₂ = {f2:.{decimals}f})", time_col, "Dissolved (%)", legend=True)
            st.pyplot(fig_main)

            export_results(
                prefix="dissolution_f2",
                report_title="Statistical Analysis Report",
                module_name="Dissolution Comparison (f₂)",
                statistical_analysis="Mean dissolution profiles were calculated for the reference and test products at each common time point. Variability was summarized with standard deviation and coefficient of variation. The similarity factor f₂ was then computed from the squared differences between the mean profiles.",
                offer_text="This analysis offers a concise way to compare two dissolution profiles, identify the size of mean differences across time, and support a similarity conclusion using the commonly used f₂ metric.",
                python_tools="Python tools used here include pandas and numpy for profile handling, matplotlib for the dissolution plot with error bars, openpyxl for Excel export, and reportlab for the PDF-style report.",
                table_map={"Profile Summary": summary, "f2 Assessment": assess},
                figure_map={"Dissolution profiles": fig_to_png_bytes(fig_main)},
                conclusion=f"The calculated similarity factor was {f2:.{decimals}f}. Based on the usual threshold of 50, the two profiles were classified as {verdict.lower()}.",
                decimals=decimals,
            )
        except Exception as e:
            st.error(str(e))


# -------------------------------------------------
# App 05 Two-Sample Tests
# -------------------------------------------------
elif app_selection == "05 - Two-Sample Tests":
    app_header("⚖️ App 05 - Two-Sample Tests", "Paste one table with headers, then choose any two sample columns to compare.")
    st.markdown("Paste a **wide** table from Excel. If you paste more than two numeric columns, you can choose which two columns to compare from the dropdowns. The selected headers are used automatically in tables and plots.")
    data_input = st.text_area("Data table (with headers)", height=240)
    mode = st.radio("Comparison type", ["Independent samples", "Paired samples"], horizontal=True)
    alpha = st.slider("Significance level α", 0.001, 0.100, 0.05, 0.001)
    decimals = st.slider("Decimals", 1, 8, DEFAULT_DECIMALS, key="two_dec")

    if data_input:
        try:
            df = parse_pasted_table(data_input, header=True)
            num_cols = get_numeric_columns(df)
            if len(num_cols) < 2:
                st.error("Please paste at least two numeric columns with headers.")
            else:
                c1, c2 = st.columns(2)
                with c1:
                    sample_a = st.selectbox("Sample A", num_cols, index=0)
                with c2:
                    sample_b = st.selectbox("Sample B", [c for c in num_cols if c != sample_a], index=0)

                x = to_numeric(df[sample_a]).dropna().to_numpy()
                y = to_numeric(df[sample_b]).dropna().to_numpy()
                if mode == "Paired samples":
                    paired_df = df[[sample_a, sample_b]].copy().apply(to_numeric).dropna()
                    x = paired_df[sample_a].to_numpy()
                    y = paired_df[sample_b].to_numpy()
                    if len(x) < 2:
                        raise ValueError("Paired analysis requires at least two complete pairs.")

                def ad(a):
                    stat, p = normal_ad(a)
                    return stat, p, p >= alpha

                a1, p1, n1 = ad(x)
                a2, p2, n2 = ad(y)
                desc = pd.DataFrame({
                    "Sample": [sample_a, sample_b],
                    "N": [len(x), len(y)],
                    "Mean": [x.mean(), y.mean()],
                    "Std. Deviation": [x.std(ddof=1), y.std(ddof=1)],
                    "Median": [np.median(x), np.median(y)],
                    "Minimum": [x.min(), y.min()],
                    "Maximum": [x.max(), y.max()],
                    "AD A* Statistic": [a1, a2],
                    "AD P-Value": [p1, p2],
                    "Normal at α": ["Yes" if n1 else "No", "Yes" if n2 else "No"],
                })
                report_table(desc, "Sample summary and normality checks", decimals)

                if mode == "Independent samples":
                    lev_stat, lev_p = stats.levene(x, y)
                    equal_var = lev_p >= alpha
                    t_stat, t_p = stats.ttest_ind(x, y, equal_var=equal_var)
                    mw_stat, mw_p = stats.mannwhitneyu(x, y, alternative="two-sided")
                    tests = pd.DataFrame({
                        "Test": ["Levene test", "Student/Welch t-test", "Mann–Whitney U"],
                        "Statistic": [lev_stat, t_stat, mw_stat],
                        "P-Value": [lev_p, t_p, mw_p],
                        "Conclusion": [
                            "Equal variances" if equal_var else "Unequal variances",
                            "Significant" if t_p < alpha else "Not significant",
                            "Significant" if mw_p < alpha else "Not significant",
                        ],
                    })
                    conclusion = f"The comparison between {sample_a} and {sample_b} was evaluated as independent samples. The t-test p-value was {fmt_p(t_p)} and the Mann–Whitney p-value was {fmt_p(mw_p)}."
                else:
                    d = x - y
                    ad_d, p_d, nd = ad(d)
                    t_stat, t_p = stats.ttest_rel(x, y)
                    try:
                        w_stat, w_p = stats.wilcoxon(x, y)
                    except Exception:
                        w_stat, w_p = np.nan, np.nan
                    tests = pd.DataFrame({
                        "Test": ["AD test of paired differences", "Paired t-test", "Wilcoxon signed-rank"],
                        "Statistic": [ad_d, t_stat, w_stat],
                        "P-Value": [p_d, t_p, w_p],
                        "Conclusion": [
                            "Normal differences" if nd else "Non-normal differences",
                            "Significant" if t_p < alpha else "Not significant",
                            "Significant" if (pd.notna(w_p) and w_p < alpha) else "Not significant",
                        ],
                    })
                    conclusion = f"The comparison between {sample_a} and {sample_b} was evaluated as paired samples. The paired t-test p-value was {fmt_p(t_p)} and the Wilcoxon p-value was {fmt_p(w_p)}."

                report_table(tests, f"Two-sample test results (α = {alpha})", decimals)

                fig_box, ax = plt.subplots(figsize=(FIG_W, FIG_H))
                ax.boxplot([x, y], labels=[sample_a, sample_b], patch_artist=True)
                apply_ax_style(ax, "Two-sample comparison", "Sample", "Value")
                st.pyplot(fig_box)

                fig_dens, ax2 = plt.subplots(figsize=(FIG_W, FIG_H))
                if len(np.unique(x)) > 1:
                    xs = np.linspace(np.min(x), np.max(x), 200)
                    ax2.plot(xs, gaussian_kde(x)(xs), color=PRIMARY_COLOR, lw=2, label=sample_a)
                if len(np.unique(y)) > 1:
                    ys = np.linspace(np.min(y), np.max(y), 200)
                    ax2.plot(ys, gaussian_kde(y)(ys), color=SECONDARY_COLOR, lw=2, label=sample_b)
                apply_ax_style(ax2, "Density comparison", "Value", "Density", legend=True)
                st.pyplot(fig_dens)

                export_results(
                    prefix="two_sample_tests",
                    report_title="Statistical Analysis Report",
                    module_name="Two-Sample Tests",
                    statistical_analysis="Two selected sample columns from the pasted Excel table were compared. The module first summarized each sample and checked approximate normality using the Anderson–Darling test. For independent samples, equality of variances was assessed with Levene’s test before applying the appropriate t-test; a Mann–Whitney test was also provided as a non-parametric comparison. For paired samples, analyses were based on row-wise differences using the paired t-test and Wilcoxon signed-rank test.",
                    offer_text="This analysis offers a structured way to compare two populations for differences in means or distributions, verify key assumptions, and switch easily between any two columns from a wider pasted table without repasting data.",
                    python_tools="Python tools used here include pandas and numpy for selecting the chosen sample columns from the pasted table, statsmodels.stats.diagnostic.normal_ad for Anderson–Darling normality testing, scipy.stats for Levene, t-tests, Mann–Whitney, and Wilcoxon procedures, matplotlib for comparison plots, openpyxl for Excel export, and reportlab for the PDF-style report.",
                    table_map={"Summary": desc, "Tests": tests},
                    figure_map={"Box plot": fig_to_png_bytes(fig_box), "Density plot": fig_to_png_bytes(fig_dens)},
                    conclusion=conclusion,
                    decimals=decimals,
                )
        except Exception as e:
            st.error(str(e))


# -------------------------------------------------
# App 06 Two-Way ANOVA
# -------------------------------------------------
elif app_selection == "06 - Two-Way ANOVA":
    app_header("📐 App 06 - Two-Way ANOVA", "Analyze two categorical factors and their interaction for a selected numeric response.")
    data_input = st.text_area("Paste data with headers", height=240)
    decimals = st.slider("Decimals", 1, 8, DEFAULT_DECIMALS, key="anova2_dec")
    if data_input:
        try:
            df = parse_pasted_table(data_input, header=True)
            all_cols = list(df.columns)
            c1, c2, c3 = st.columns(3)
            with c1:
                factor_a = st.selectbox("Factor A", all_cols, index=0)
            with c2:
                factor_b = st.selectbox("Factor B", [c for c in all_cols if c != factor_a], index=0)
            with c3:
                response = st.selectbox("Response", [c for c in all_cols if c not in [factor_a, factor_b]], index=0)
            d = df[[factor_a, factor_b, response]].copy()
            d[response] = to_numeric(d[response])
            d = d.dropna().rename(columns={factor_a: "FactorA", factor_b: "FactorB", response: "Response"})
            model = smf.ols("Response ~ C(FactorA) * C(FactorB)", data=d).fit()
            raw_anova = anova_lm(model, typ=2)

            st.markdown(f"**Model: {response} ~ {factor_a} + {factor_b} + {factor_a} * {factor_b}**")

            anova_rows = []
            mapping = [
                ("C(FactorA)", factor_a),
                ("C(FactorB)", factor_b),
                ("C(FactorA):C(FactorB)", "Interaction"),
            ]
            for idx, label in mapping:
                if idx in raw_anova.index:
                    row = raw_anova.loc[idx]
                    dfv = row.get("df", np.nan)
                    ss = row.get("sum_sq", np.nan)
                    anova_rows.append({
                        "Source": label,
                        "DF": dfv,
                        "Sum of Squares": ss,
                        "Mean Square": ss / dfv if pd.notna(dfv) and dfv != 0 else np.nan,
                        "F Value": row.get("F", np.nan),
                        "P Value": row.get("PR(>F)", np.nan),
                    })
            if "Residual" in raw_anova.index:
                row = raw_anova.loc["Residual"]
                dfv = row.get("df", np.nan)
                ss = row.get("sum_sq", np.nan)
                anova_rows.append({
                    "Source": "Error",
                    "DF": dfv,
                    "Sum of Squares": ss,
                    "Mean Square": ss / dfv if pd.notna(dfv) and dfv != 0 else np.nan,
                    "F Value": np.nan,
                    "P Value": np.nan,
                })
            anova_rows.append({
                "Source": "N",
                "DF": len(d),
                "Sum of Squares": np.nan,
                "Mean Square": np.nan,
                "F Value": np.nan,
                "P Value": np.nan,
            })
            anova = pd.DataFrame(anova_rows)
            report_table(anova, "Two-way ANOVA table", decimals)

            summary = d.groupby(["FactorA", "FactorB"])["Response"].agg(["count", "mean", "std", "min", "max"]).reset_index()
            summary.columns = [factor_a, factor_b, "N", "Mean", "Std. Deviation", "Minimum", "Maximum"]
            report_table(summary, "Cell summary statistics", decimals)

            fig_inter, ax = plt.subplots(figsize=(FIG_W, FIG_H))
            for lvl in d["FactorB"].astype(str).unique():
                sub = d[d["FactorB"].astype(str) == lvl]
                means = sub.groupby("FactorA")["Response"].mean().reset_index()
                ax.plot(means["FactorA"].astype(str), means["Response"], marker='o', lw=2, label=f"{factor_b} = {lvl}")
            apply_ax_style(ax, "Interaction plot", factor_a, response, legend=True)
            st.pyplot(fig_inter)

            fig_res = residual_plot(model.fittedvalues, model.resid, xlabel="Fitted values", ylabel="Residuals", title="Residuals vs fitted")
            st.pyplot(fig_res)
            fig_qq = qq_plot(model.resid, title="Normal probability plot of ANOVA residuals")
            st.pyplot(fig_qq)

            export_results(
                prefix="two_way_anova",
                report_title="Statistical Analysis Report",
                module_name="Two-Way ANOVA",
                statistical_analysis="A two-way analysis of variance was fitted to the selected response variable using two chosen categorical factors and their interaction. Sums of squares, mean squares, F statistics, and p-values were computed from the linear model, and residual diagnostics were generated to support assessment of model assumptions.",
                offer_text="This analysis offers a direct way to quantify the main effects of two factors, test whether their interaction is present, compare cell means, and visualize whether factor effects are consistent across levels of the other factor.",
                python_tools="Python tools used here include pandas and numpy for column selection and aggregation, statsmodels.formula.api and statsmodels.stats.anova for model fitting and ANOVA calculations, matplotlib for interaction and residual plots, openpyxl for Excel export, and reportlab for the PDF-style report.",
                table_map={"ANOVA": anova, "Cell Summary": summary},
                figure_map={
                    "Interaction plot": fig_to_png_bytes(fig_inter),
                    "Residuals vs fitted": fig_to_png_bytes(fig_res),
                    "Normal probability plot": fig_to_png_bytes(fig_qq),
                },
                conclusion="The ANOVA table reports whether the selected factors and their interaction contributed significantly to variation in the selected response.",
                decimals=decimals,
            )
        except Exception as e:
            st.error(str(e))


# -------------------------------------------------
# App 07 Tolerance & Confidence Intervals
# -------------------------------------------------
elif app_selection == "07 - Tolerance & Confidence Intervals":
    app_header("🎯 App 07 - Tolerance & Confidence Intervals", "Generate confidence intervals and normal-theory tolerance intervals for one or two samples.")
    data_input = st.text_area("Paste one table with headers", height=240)
    confidence = st.slider("Confidence level (%)", 80, 99, 95)
    coverage = st.slider("Population coverage for TI (%)", 80, 99, 95)
    paired = st.checkbox("Paired comparison for difference in means", value=False)
    decimals = st.slider("Decimals", 1, 8, DEFAULT_DECIMALS, key="ti_dec")

    if data_input:
        try:
            df = parse_pasted_table(data_input, header=True)
            num_cols = get_numeric_columns(df)
            if not num_cols:
                st.error("No numeric columns were found.")
            else:
                c1, c2 = st.columns(2)
                with c1:
                    sample_a = st.selectbox("Sample A", num_cols, index=0)
                with c2:
                    sample_b = st.selectbox("Sample B (optional)", ["(None)"] + [c for c in num_cols if c != sample_a], index=0)
                alpha = 1 - confidence / 100
                p = coverage / 100
                conf = confidence / 100
                x = to_numeric(df[sample_a]).dropna().to_numpy()
                summaries = []

                def mean_ci(a):
                    n = len(a)
                    m = np.mean(a)
                    s = np.std(a, ddof=1)
                    se = s / np.sqrt(n)
                    tcrit = t.ppf(1 - alpha / 2, n - 1)
                    return m, s, m - tcrit * se, m + tcrit * se

                mx, sx, lx, ux = mean_ci(x)
                _, tlx, tux = tolerance_interval_normal(x, p=p, conf=conf, two_sided=True)
                summaries.append({"Sample": sample_a, "N": len(x), "Mean": mx, "Std. Deviation": sx, f"{confidence}% CI Lower": lx, f"{confidence}% CI Upper": ux, f"{coverage}%/{confidence}% TI Lower": tlx, f"{coverage}%/{confidence}% TI Upper": tux})
                diff_tbl = None
                if sample_b != "(None)":
                    y_all = df[[sample_a, sample_b]].copy().apply(to_numeric) if paired else None
                    y = (y_all[sample_b].dropna().to_numpy() if not paired else y_all[sample_b].dropna().to_numpy()) if paired else to_numeric(df[sample_b]).dropna().to_numpy()
                    if paired:
                        pair_df = df[[sample_a, sample_b]].copy().apply(to_numeric).dropna()
                        x = pair_df[sample_a].to_numpy()
                        y = pair_df[sample_b].to_numpy()
                    my, sy, ly, uy = mean_ci(y)
                    _, tly, tuy = tolerance_interval_normal(y, p=p, conf=conf, two_sided=True)
                    summaries.append({"Sample": sample_b, "N": len(y), "Mean": my, "Std. Deviation": sy, f"{confidence}% CI Lower": ly, f"{confidence}% CI Upper": uy, f"{coverage}%/{confidence}% TI Lower": tly, f"{coverage}%/{confidence}% TI Upper": tuy})

                    if paired:
                        d = x - y
                        md, sd, ld, ud = mean_ci(d)
                        diff_tbl = pd.DataFrame({"Comparison": [f"{sample_a} - {sample_b}"], "Mean Difference": [md], f"{confidence}% CI Lower": [ld], f"{confidence}% CI Upper": [ud]})
                    else:
                        nx, ny = len(x), len(y)
                        dx = x.mean() - y.mean()
                        sx2, sy2 = x.var(ddof=1), y.var(ddof=1)
                        se = np.sqrt(sx2 / nx + sy2 / ny)
                        dfw = (sx2 / nx + sy2 / ny) ** 2 / (((sx2 / nx) ** 2) / (nx - 1) + ((sy2 / ny) ** 2) / (ny - 1))
                        tcrit = t.ppf(1 - alpha / 2, dfw)
                        diff_tbl = pd.DataFrame({"Comparison": [f"{sample_a} - {sample_b}"], "Mean Difference": [dx], f"{confidence}% CI Lower": [dx - tcrit * se], f"{confidence}% CI Upper": [dx + tcrit * se]})

                out = pd.DataFrame(summaries)
                report_table(out, "Confidence and tolerance intervals", decimals)
                table_map = {"Intervals": out}
                if diff_tbl is not None:
                    report_table(diff_tbl, "Confidence interval for mean difference", decimals)
                    table_map["Mean Difference CI"] = diff_tbl

                data_list = [to_numeric(df[sample_a]).dropna().to_numpy()] if sample_b == "(None)" else ([x, y] if paired else [to_numeric(df[sample_a]).dropna().to_numpy(), to_numeric(df[sample_b]).dropna().to_numpy()])
                labels = [sample_a] if sample_b == "(None)" else [sample_a, sample_b]
                fig_box, ax = plt.subplots(figsize=(FIG_W, FIG_H))
                ax.boxplot(data_list, labels=labels, patch_artist=True)
                apply_ax_style(ax, "Sample distributions", "Sample", "Value")
                st.pyplot(fig_box)

                export_results(
                    prefix="tolerance_confidence_intervals",
                    report_title="Statistical Analysis Report",
                    module_name="Tolerance & Confidence Intervals",
                    statistical_analysis="For each selected sample column, a confidence interval for the mean was calculated using the t distribution. In addition, a normal-theory tolerance interval was calculated to estimate a range expected to contain a chosen proportion of the population with the chosen confidence level. When two samples were selected, a confidence interval for the mean difference was also computed.",
                    offer_text="This analysis offers a concise way to summarize uncertainty around the sample mean, estimate population coverage ranges, and compare two populations for differences in means while using the pasted column headers directly in the results.",
                    python_tools="Python tools used here include pandas and numpy for extracting selected columns, scipy.stats for t and tolerance-interval related calculations, matplotlib for distribution plots, openpyxl for Excel export, and reportlab for the PDF-style report.",
                    table_map=table_map,
                    figure_map={"Box plot": fig_to_png_bytes(fig_box)},
                    conclusion="Confidence intervals quantify uncertainty around the sample mean, while tolerance intervals provide an estimated range expected to contain a specified share of the full population.",
                    decimals=decimals,
                )
        except Exception as e:
            st.error(str(e))



# -------------------------------------------------
# App 08 PCA Analysis
# -------------------------------------------------
elif app_selection == "08 - PCA Analysis":
    app_header("🌐 App 08 - PCA Analysis", "Reduce multivariate data to principal components and visualize scores and loadings.")
    data_input = st.text_area("Paste data with headers", height=240)
    decimals = st.slider("Decimals", 1, 8, DEFAULT_DECIMALS, key="pca_dec")
    if data_input:
        try:
            df = parse_pasted_table(data_input, header=True)
            num_cols = get_numeric_columns(df)
            all_cols = list(df.columns)
            c1, c2, c3 = st.columns([1.25, 1, 1])
            with c1:
                vars_sel = st.multiselect("Numeric variables", num_cols, default=num_cols)
            with c2:
                label_col = st.selectbox("Label column (optional)", ["(None)"] + all_cols)
            with c3:
                group_col = st.selectbox("Group column (optional)", ["(None)"] + [c for c in all_cols if c != label_col])

            if len(vars_sel) >= 2:
                X = df[vars_sel].apply(to_numeric).dropna()
                Z = (X - X.mean()) / X.std(ddof=1)
                pca = PCA(n_components=2)
                scores = pca.fit_transform(Z)
                loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
                exp = pca.explained_variance_ratio_ * 100
                eig = pd.DataFrame({
                    "Principal Component": ["PC1", "PC2"],
                    "Eigenvalue": pca.explained_variance_,
                    "Variance Explained (%)": exp,
                    "Cumulative Variance (%)": np.cumsum(exp),
                })
                load_df = pd.DataFrame({"Variable": vars_sel, "PC1": loadings[:, 0], "PC2": loadings[:, 1]})
                report_table(eig, "Eigenvalues and explained variance", decimals)
                report_table(load_df, "Loading matrix", decimals)

                scores_df = pd.DataFrame({"PC1": scores[:, 0], "PC2": scores[:, 1]}, index=X.index)
                if label_col != "(None)":
                    scores_df["Label"] = df.loc[X.index, label_col].astype(str).values
                if group_col != "(None)":
                    scores_df["Group"] = df.loc[X.index, group_col].astype(str).values

                score_cfg = get_plot_cfg("PCA score plot")
                fig_scores, ax = plt.subplots(figsize=(score_cfg["fig_w"], score_cfg["fig_h"]))
                color_cycle = [score_cfg["primary_color"], score_cfg["secondary_color"], score_cfg["tertiary_color"], "#9467bd", "#8c564b", "#e377c2"]

                if group_col != "(None)":
                    unique_groups = list(scores_df["Group"].unique())
                    for i, grp in enumerate(unique_groups):
                        col = color_cycle[i % len(color_cycle)]
                        m = scores_df["Group"] == grp
                        ax.scatter(
                            scores_df.loc[m, "PC1"],
                            scores_df.loc[m, "PC2"],
                            s=score_cfg["marker_size"],
                            color=col,
                            label=str(grp),
                        )
                        draw_conf_ellipse(
                            scores_df.loc[m, ["PC1", "PC2"]].to_numpy(),
                            ax,
                            edgecolor=col,
                            facecolor=col,
                            plot_key="PCA score plot",
                        )
                else:
                    col = score_cfg["primary_color"]
                    ax.scatter(scores_df["PC1"], scores_df["PC2"], s=score_cfg["marker_size"], color=col, label="Scores")
                    draw_conf_ellipse(scores_df[["PC1", "PC2"]].to_numpy(), ax, edgecolor=col, facecolor=col, plot_key="PCA score plot")

                if label_col != "(None)":
                    for _, row in scores_df.iterrows():
                        ax.text(row["PC1"], row["PC2"], str(row["Label"]), fontsize=8)

                ax.axhline(0, color="#64748b", lw=score_cfg["aux_line_width"], ls=score_cfg["aux_line_style"])
                ax.axvline(0, color="#64748b", lw=score_cfg["aux_line_width"], ls=score_cfg["aux_line_style"])
                apply_ax_style(ax, "PCA score plot", f"PC1 ({exp[0]:.1f}% var)", f"PC2 ({exp[1]:.1f}% var)", legend=(group_col != "(None)"), plot_key="PCA score plot")
                st.pyplot(fig_scores)

                load_cfg = get_plot_cfg("PCA loading plot")
                fig_load, ax2 = plt.subplots(figsize=(load_cfg["fig_w"], load_cfg["fig_h"]))
                ax2.axhline(0, color="#64748b", lw=load_cfg["aux_line_width"], ls=load_cfg["aux_line_style"])
                ax2.axvline(0, color="#64748b", lw=load_cfg["aux_line_width"], ls=load_cfg["aux_line_style"])
                for i, var in enumerate(vars_sel):
                    ax2.arrow(
                        0,
                        0,
                        loadings[i, 0],
                        loadings[i, 1],
                        head_width=load_cfg["arrow_size"],
                        length_includes_head=True,
                        color=load_cfg["primary_color"],
                        lw=load_cfg["line_width"],
                        ls=load_cfg["line_style"],
                    )
                    ax2.text(loadings[i, 0], loadings[i, 1], var)
                lim = max(1.1, np.max(np.abs(loadings)) * 1.2)
                ax2.set_xlim(-lim, lim)
                ax2.set_ylim(-lim, lim)
                apply_ax_style(ax2, "PCA loading plot", "PC1", "PC2", plot_key="PCA loading plot")
                st.pyplot(fig_load)

                export_results(
                    prefix="pca_analysis",
                    report_title="Statistical Analysis Report",
                    module_name="PCA Analysis",
                    statistical_analysis="Principal component analysis was performed on the selected numeric variables after standardization to zero mean and unit variance. Eigenvalues, explained variance, component scores, and variable loadings were calculated, and optional pasted header columns were used as point labels or groups on the score plot.",
                    offer_text="This analysis offers a way to reduce dimensionality, detect clustering or separation patterns, identify influential variables, and visualize multivariate relationships while retaining labels or grouping information from the pasted Excel headers.",
                    python_tools="Python tools used here include pandas and numpy for data handling and standardization, sklearn.decomposition.PCA for principal component analysis, matplotlib for score and loading plots, openpyxl for Excel export, and reportlab for the PDF-style report.",
                    table_map={"Explained Variance": eig, "Loadings": load_df, "Scores": scores_df.reset_index(drop=True)},
                    figure_map={"PCA score plot": fig_to_png_bytes(fig_scores), "PCA loading plot": fig_to_png_bytes(fig_load)},
                    conclusion="PCA transforms correlated variables into orthogonal components that summarize the major variation structure in the data and help reveal clustering, separation, and variable influence patterns.",
                    decimals=decimals,
                )
        except Exception as e:
            st.error(str(e))



# -------------------------------------------------
# App 09 DoE / Response Surfaces
# -------------------------------------------------
elif app_selection == "09 - DoE / Response Surfaces":
    app_header("🧪 App 09 - DoE / Response Surfaces", "Build a full-factorial design, assign blocks, and fit linear / interaction / quadratic response-surface models.")
    tab_build, tab_analyze = st.tabs(["Build design", "Analyze responses"])

    with tab_build:
        st.markdown("### Build a full-factorial DoE")
        st.markdown("Enter factor names and low / high levels. You can add blocks, center points, replicates, and randomize the run order within block.")
        c1, c2, c3, c4 = st.columns([0.8, 0.9, 0.9, 1])
        with c1:
            n_factors = st.number_input("Number of factors", min_value=2, max_value=8, value=2, step=1)
        with c2:
            n_blocks = st.number_input("Number of blocks", min_value=1, max_value=20, value=1, step=1)
        with c3:
            replicates = st.number_input("Replicates / treatment / block", min_value=1, max_value=20, value=1, step=1)
        with c4:
            center_points = st.number_input("Center points / block", min_value=0, max_value=20, value=0, step=1)

        factor_names_txt = st.text_input(
            "Factor names (comma-separated)",
            value=", ".join([chr(65 + i) for i in range(n_factors)])
        )
        factor_names = [f.strip() for f in factor_names_txt.split(",") if f.strip()]
        if len(factor_names) != n_factors:
            st.warning(f"Please provide exactly {n_factors} factor names.")
        else:
            st.markdown("**Low / high levels**")
            cols = st.columns(n_factors)
            low_vals, high_vals = {}, {}
            for i, f in enumerate(factor_names):
                with cols[i]:
                    low_vals[f] = st.text_input(f"{f} low", value="-1", key=f"low_{f}")
                    high_vals[f] = st.text_input(f"{f} high", value="1", key=f"high_{f}")

            randomize = st.checkbox("Randomize runs within each block", value=True)
            random_seed = st.number_input("Random seed", min_value=0, max_value=100000, value=42, step=1) if randomize else 42

            try:
                low_numeric = {f: float(low_vals[f]) for f in factor_names}
                high_numeric = {f: float(high_vals[f]) for f in factor_names}
                for f in factor_names:
                    if high_numeric[f] == low_numeric[f]:
                        raise ValueError(f"{f}: high level must differ from low level.")

                coded_combos = list(product([-1, 1], repeat=n_factors))
                rows = []
                rng = np.random.default_rng(int(random_seed))
                for block in range(1, int(n_blocks) + 1):
                    block_rows = []
                    for rep in range(1, int(replicates) + 1):
                        for combo in coded_combos:
                            row = {"Block": block, "Replicate": rep, "Center point": "No"}
                            for i, f in enumerate(factor_names):
                                coded = combo[i]
                                actual = low_numeric[f] if coded == -1 else high_numeric[f]
                                row[f"{f} (coded)"] = coded
                                row[f] = actual
                            block_rows.append(row)
                    for cp in range(1, int(center_points) + 1):
                        row = {"Block": block, "Replicate": np.nan, "Center point": f"CP{cp}"}
                        for f in factor_names:
                            row[f"{f} (coded)"] = 0
                            row[f] = (low_numeric[f] + high_numeric[f]) / 2
                        block_rows.append(row)
                    if randomize:
                        rng.shuffle(block_rows)
                    for run_idx, row in enumerate(block_rows, start=1):
                        row["Run"] = run_idx
                        rows.append(row)

                design_df = pd.DataFrame(rows)
                ordered_cols = ["Block", "Run", "Replicate", "Center point"]
                for f in factor_names:
                    ordered_cols += [f"{f} (coded)", f]
                design_df = design_df[ordered_cols]
                report_table(design_df, "Generated DoE design", 3)

                c_csv, c_xlsx = st.columns(2)
                with c_csv:
                    st.download_button(
                        "⬇️ Export design as CSV",
                        data=design_df.to_csv(index=False).encode("utf-8"),
                        file_name="doe_design.csv",
                        mime="text/csv",
                    )
                with c_xlsx:
                    st.download_button(
                        "⬇️ Export design as Excel",
                        data=make_excel_bytes({"DoE Design": design_df}),
                        file_name="doe_design.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
            except Exception as e:
                st.error(str(e))

    with tab_analyze:
        st.markdown("### Analyze responses and build response surfaces")
        data_input = st.text_area("Paste completed DoE data with headers", height=240)
        decimals = st.slider("Decimals", 1, 8, DEFAULT_DECIMALS, key="doe_dec")
        if data_input:
            try:
                df = parse_pasted_table(data_input, header=True)
                num_cols = get_numeric_columns(df)
                all_cols = list(df.columns)

                c1, c2, c3, c4 = st.columns([1.35, 1, 1, 1])
                with c1:
                    factors = st.multiselect("Numeric factors", num_cols, default=num_cols[: min(2, len(num_cols))])
                with c2:
                    response = st.selectbox("Response", [c for c in num_cols if c not in factors] or num_cols)
                with c3:
                    model_type = st.selectbox("Model type", ["linear", "interaction", "quadratic"])
                with c4:
                    block_col = st.selectbox("Block column (optional)", ["(None)"] + [c for c in all_cols if c not in factors + [response]])

                if len(factors) >= 2:
                    use_cols = factors + [response] + ([block_col] if block_col != "(None)" else [])
                    d = df[use_cols].copy()
                    for c in factors + [response]:
                        d[c] = to_numeric(d[c])
                    d = d.dropna()

                    safe_factor_names = [f"F{i+1}" for i in range(len(factors))]
                    rename_map = {orig: safe for orig, safe in zip(factors, safe_factor_names)}
                    inv_map = {v: k for k, v in rename_map.items()}

                    safe_df = d.rename(columns=rename_map).rename(columns={response: "Response"})
                    if block_col != "(None)":
                        safe_df["Block"] = d[block_col].astype(str).values

                    formula = doe_formula(safe_factor_names, model_type=model_type)
                    if block_col != "(None)":
                        formula += " + C(Block)"

                    model = smf.ols(formula, data=safe_df).fit()
                    anova = anova_lm(model, typ=2).reset_index().rename(
                        columns={
                            "index": "Source",
                            "sum_sq": "Sum of Squares",
                            "df": "df",
                            "F": "F-Statistic",
                            "PR(>F)": "P-Value",
                        }
                    )
                    anova["Mean Square"] = anova["Sum of Squares"] / anova["df"]
                    anova["SS (%)"] = anova["Sum of Squares"] / anova["Sum of Squares"].sum() * 100

                    def pretty_term(term):
                        term = str(term)
                        if term == "Residual":
                            return "Error"
                        if term == "Intercept":
                            return "Intercept"
                        if term.startswith("C(Block)"):
                            return "Block"
                        term = term.replace(":", " × ")
                        term = term.replace("I(", "").replace(" ** 2)", "²")
                        for safe, orig in inv_map.items():
                            term = term.replace(safe, orig)
                        return term

                    anova["Source"] = anova["Source"].map(pretty_term)
                    coef = pd.DataFrame({
                        "Term": [pretty_term(t) for t in model.params.index],
                        "Coefficient": model.params.values,
                        "P-Value": model.pvalues.values,
                    })

                    report_table(anova[["Source", "df", "Sum of Squares", "Mean Square", "F-Statistic", "P-Value", "SS (%)"]], f"DoE ANOVA ({model_type} model)", decimals)
                    report_table(coef, "Model coefficients", decimals)

                    xfac = st.selectbox("X-axis factor", factors, index=0)
                    yfac = st.selectbox("Y-axis factor", [f for f in factors if f != xfac], index=0)
                    other_factors = [f for f in factors if f not in [xfac, yfac]]
                    fixed_vals = {}
                    if other_factors:
                        st.markdown("**Fixed levels for remaining factors**")
                        cols = st.columns(len(other_factors))
                        for i, f in enumerate(other_factors):
                            fixed_vals[f] = cols[i].number_input(f, value=float(d[f].mean()))

                    x_vals = np.linspace(d[xfac].min(), d[xfac].max(), 60)
                    y_vals = np.linspace(d[yfac].min(), d[yfac].max(), 60)
                    xx, yy = np.meshgrid(x_vals, y_vals)
                    grid = pd.DataFrame({xfac: xx.ravel(), yfac: yy.ravel()})
                    for f in other_factors:
                        grid[f] = fixed_vals[f]
                    if block_col != "(None)":
                        block_default = str(d[block_col].iloc[0])
                        selected_block = st.selectbox("Block level for prediction grid", sorted(d[block_col].astype(str).unique()))
                        grid["Block"] = selected_block
                    safe_grid = grid.rename(columns=rename_map)
                    zz = model.predict(safe_grid).to_numpy().reshape(xx.shape)

                    contour_cfg = get_plot_cfg("DoE contour")
                    fig_contour, ax = plt.subplots(figsize=(contour_cfg["fig_w"], contour_cfg["fig_h"]))
                    cs = ax.contourf(xx, yy, zz, levels=20, cmap="viridis")
                    ax.contour(xx, yy, zz, levels=10, colors=contour_cfg["primary_color"], linewidths=max(0.6, contour_cfg["aux_line_width"] * 0.7))
                    fig_contour.colorbar(cs, ax=ax, label=response)
                    ax.scatter(
                        d[xfac],
                        d[yfac],
                        c="white",
                        edgecolor=contour_cfg["primary_color"],
                        s=contour_cfg["marker_size"],
                    )
                    apply_ax_style(ax, f"Contour plot for {response}", xfac, yfac, plot_key="DoE contour")
                    st.pyplot(fig_contour)

                    fig_surface = plt.figure(figsize=(contour_cfg["fig_w"], contour_cfg["fig_h"] + 0.5))
                    ax3 = fig_surface.add_subplot(111, projection="3d")
                    surf = ax3.plot_surface(xx, yy, zz, cmap="viridis", edgecolor="none", alpha=0.88)
                    ax3.scatter(d[xfac], d[yfac], d[response], c="black", s=max(12, contour_cfg["marker_size"] * 0.45))
                    ax3.set_xlabel(xfac)
                    ax3.set_ylabel(yfac)
                    ax3.set_zlabel(response)
                    ax3.set_title(f"Response surface for {response}")
                    fig_surface.colorbar(surf, ax=ax3, shrink=0.68, aspect=12)
                    st.pyplot(fig_surface)

                    fig_res = residual_plot(model.fittedvalues, model.resid, xlabel="Fitted values", ylabel="Residuals", title="Residuals vs fitted")
                    st.pyplot(fig_res)
                    fig_qq = qq_plot(model.resid, title="Normal probability plot of DoE residuals")
                    st.pyplot(fig_qq)

                    export_results(
                        prefix="doe_response_surfaces",
                        report_title="Statistical Analysis Report",
                        module_name="DoE / Response Surfaces",
                        statistical_analysis="A design-of-experiments style regression model was fitted to the selected numeric response using the chosen numeric factors. Depending on the selected option, the model included linear terms only, linear plus interactions, or a quadratic response-surface structure. When a block column was supplied, block was included as a categorical effect. ANOVA, model coefficients, contour plots, surface plots, and residual diagnostics were generated from the fitted model.",
                        offer_text="This analysis offers a practical way to build simple factorial designs, quantify factor effects, inspect interactions or curvature, account for blocks, and visualize the response surface over two selected factors while fixing any remaining factors at chosen values.",
                        python_tools="Python tools used here include pandas and numpy for design construction and factor selection, itertools.product for generating full-factorial combinations, statsmodels for model fitting and ANOVA, matplotlib for contour, 3D surface, and residual diagnostic plots, openpyxl for Excel export, and reportlab for the PDF-style report.",
                        table_map={"DoE ANOVA": anova, "Coefficients": coef},
                        figure_map={
                            "Contour plot": fig_to_png_bytes(fig_contour),
                            "Response surface": fig_to_png_bytes(fig_surface),
                            "Residuals vs fitted": fig_to_png_bytes(fig_res),
                            "Normal probability plot": fig_to_png_bytes(fig_qq),
                        },
                        conclusion="The fitted DoE model can be used to assess influential factors, detect interactions or curvature, evaluate block effects, and visualize predicted response behavior across the chosen design space.",
                        decimals=decimals,
                    )
            except Exception as e:
                st.error(str(e))

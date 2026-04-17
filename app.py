import re
from io import StringIO, BytesIO
from textwrap import dedent

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

with st.sidebar.expander("Display & export settings", expanded=False):
    FIG_W = st.slider("Figure width", 6.0, 14.0, 8.5, 0.5)
    FIG_H = st.slider("Figure height", 4.0, 10.0, 5.5, 0.5)
    SHOW_LEGEND = st.checkbox("Show legend", value=True)
    LEGEND_LOC = st.selectbox("Legend location", ["best", "upper right", "upper left", "lower right", "lower left", "center left", "center right", "lower center", "upper center"], index=0)
    PRIMARY_COLOR = st.color_picker("Primary color", "#1f77b4")
    SECONDARY_COLOR = st.color_picker("Secondary color", "#ff7f0e")
    BAND_COLOR = st.color_picker("Band / area color", "#93c5fd")
    GRID_ALPHA = st.slider("Grid transparency", 0.0, 1.0, 0.25, 0.05)
    DEFAULT_DECIMALS = st.slider("Default decimals", 1, 8, 3)

st.sidebar.divider()
st.sidebar.info("Paste data from Excel. Tables, charts, Excel exports, and PDF-style reports are built into the app.")


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


def parse_one_col(text):
    df = parse_pasted_table(text, header=False)
    if df is None:
        return np.array([])
    return to_numeric(df.iloc[:, 0]).dropna().to_numpy()


def get_numeric_columns(df, threshold=0.7):
    out = []
    for col in df.columns:
        converted = to_numeric(df[col])
        if converted.notna().mean() >= threshold:
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
def apply_ax_style(ax, title, xlabel, ylabel, legend=None):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=GRID_ALPHA)
    if SHOW_LEGEND and legend:
        ax.legend(frameon=False, loc=LEGEND_LOC)


def residual_plot(fitted, residuals, xlabel="Fitted", ylabel="Residuals", title="Residuals vs fitted"):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.scatter(fitted, residuals, color=PRIMARY_COLOR, s=42)
    ax.axhline(0, color="#111827", lw=1.3, ls="--")
    apply_ax_style(ax, title, xlabel, ylabel)
    return fig


def qq_plot(residuals, title="Normal probability plot of residuals"):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.get_lines()[0].set_markerfacecolor(PRIMARY_COLOR)
    ax.get_lines()[0].set_markeredgecolor(PRIMARY_COLOR)
    ax.get_lines()[1].set_color(SECONDARY_COLOR)
    apply_ax_style(ax, title, "Theoretical quantiles", "Ordered residuals")
    return fig


# -------------------------------------------------
# Statistical helpers
# -------------------------------------------------
def fit_linear(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    intercept, slope = beta
    fitted = X @ beta
    resid = y - fitted
    df = n - 2
    s = np.sqrt(np.sum(resid**2) / df)
    ss_tot = np.sum((y - y.mean()) ** 2)
    ss_res = np.sum(resid**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"intercept": intercept, "slope": slope, "XtX_inv": XtX_inv, "fitted": fitted, "resid": resid, "df": df, "s": s, "r2": r2}


def predict_intervals(model, x_vals, alpha=0.05):
    x_vals = np.asarray(x_vals, dtype=float)
    X0 = np.column_stack([np.ones(len(x_vals)), x_vals])
    fit = model["intercept"] + model["slope"] * x_vals
    h = np.sum((X0 @ model["XtX_inv"]) * X0, axis=1)
    tcrit = t.ppf(1 - alpha / 2, model["df"])
    se_mean = model["s"] * np.sqrt(h)
    se_pred = model["s"] * np.sqrt(1 + h)
    return pd.DataFrame({
        "X": x_vals,
        "Fitted": fit,
        "Lower CI": fit - tcrit * se_mean,
        "Upper CI": fit + tcrit * se_mean,
        "Lower PI": fit - tcrit * se_pred,
        "Upper PI": fit + tcrit * se_pred,
    })


def estimate_shelf_life(model, limit, decreasing=True, confidence=0.95, x_upper=100):
    xg = np.linspace(0, x_upper, 1500)
    X0 = np.column_stack([np.ones(len(xg)), xg])
    fit = model["intercept"] + model["slope"] * xg
    h = np.sum((X0 @ model["XtX_inv"]) * X0, axis=1)
    tcrit = t.ppf(confidence, model["df"])  # one-sided
    band = fit - tcrit * model["s"] * np.sqrt(h) if decreasing else fit + tcrit * model["s"] * np.sqrt(h)
    idx = np.where(band <= limit)[0] if decreasing else np.where(band >= limit)[0]
    shelf = float(xg[idx[0]]) if len(idx) else np.nan
    return shelf, xg, fit, band


def tolerance_interval_normal(data, p=0.95, conf=0.95, two_sided=True):
    data = np.asarray(data, dtype=float)
    n = len(data)
    mean = data.mean()
    sd = data.std(ddof=1)
    if n < 2:
        return np.nan, np.nan, np.nan
    if two_sided:
        g = norm.ppf((1 + p) / 2)
        k = g * np.sqrt((n - 1) * (1 + 1 / n) / chi2.ppf(1 - conf, n - 1))
        return mean, mean - k * sd, mean + k * sd
    zp = norm.ppf(p)
    k = nct.ppf(conf, n - 1, np.sqrt(n) * zp) / np.sqrt(n)
    return mean, mean - k * sd, mean + k * sd


def draw_conf_ellipse(scores, ax, edgecolor=PRIMARY_COLOR):
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
    app_header("📊 App 01 - Descriptive Statistics", "Paste a table from Excel, choose variables and grouping columns, and generate report-ready summaries.")
    data_input = st.text_area("Data (paste with headers)", height=220)
    decimals = st.slider("Decimals", 1, 8, DEFAULT_DECIMALS, key="desc_dec")

    if data_input:
        df = parse_pasted_table(data_input, header=True)
        if df is not None and not df.empty:
            st.success(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")
            with st.expander("Preview data"):
                st.dataframe(df, use_container_width=True)
            numeric_cols = get_numeric_columns(df)
            all_cols = list(df.columns)
            c1, c2, c3 = st.columns(3)
            with c1:
                selected_vars = st.multiselect("Variables", numeric_cols, default=numeric_cols)
            with c2:
                group1 = st.selectbox("Group by 1", ["(None)"] + all_cols)
            with c3:
                group2 = st.selectbox("Group by 2", ["(None)"] + all_cols)

            if st.button("Run descriptive statistics", type="primary"):
                tmp = df.copy()
                for v in selected_vars:
                    tmp[v] = to_numeric(tmp[v])
                groups = [g for g in [group1, group2] if g != "(None)"]

                def calc_stats(x):
                    return pd.Series({
                        "N": x.count(),
                        "Mean": x.mean(),
                        "Std. Deviation": x.std(ddof=1),
                        "Minimum": x.min(),
                        "Median": x.median(),
                        "Maximum": x.max(),
                        "CV (%)": (x.std(ddof=1) / x.mean() * 100) if x.mean() != 0 else np.nan,
                    })

                if groups:
                    out = tmp.groupby(groups)[selected_vars].apply(lambda g: g.apply(calc_stats)).unstack(level=-1)
                    if len(selected_vars) == 1:
                        out.columns = out.columns.droplevel(0)
                    out = out.reset_index()
                else:
                    out = tmp[selected_vars].apply(calc_stats).T.reset_index().rename(columns={"index": "Variable"})
                report_table(out, "Descriptive statistics", decimals)
                export_results(
                    prefix="descriptive_statistics",
                    report_title="Statistical Analysis Report",
                    module_name="Descriptive Statistics",
                    statistical_analysis="This module summarizes one or more quantitative variables using count, mean, standard deviation, minimum, median, maximum, and coefficient of variation. When grouping columns are selected, the summaries are produced within each group defined by the pasted headers.",
                    offer_text="It offers a fast way to characterize distributions, compare central tendency and variability across groups, and prepare clean summary tables for technical reports or development updates.",
                    python_tools="Python modules used in this analysis include pandas and numpy for data cleaning and calculations, Streamlit for the interface, matplotlib for optional plotting in other modules, openpyxl for Excel export, and reportlab for PDF-style report generation.",
                    table_map={"Descriptive Statistics": out},
                    decimals=decimals,
                )


# -------------------------------------------------
# App 02 Regression Intervals
# -------------------------------------------------
elif app_selection == "02 - Regression Intervals":
    app_header("📈 App 02 - Regression Intervals", "Fit a linear regression model and generate confidence and prediction intervals with diagnostics.")
    c1, c2 = st.columns([1.4, 1])
    with c1:
        xy_input = st.text_area("Paste X and Y data (with or without headers)", height=220)
    with c2:
        x_pred_text = st.text_area("X values for prediction (optional)")
        conf = st.slider("Confidence level (%)", 80, 99, 95)
        decimals = st.slider("Decimals", 1, 8, DEFAULT_DECIMALS, key="reg_dec")

    if xy_input:
        try:
            df, x_label, y_label = parse_xy(xy_input)
            x, y = df["x"].to_numpy(), df["y"].to_numpy()
            model = fit_linear(x, y)
            alpha = 1 - conf / 100
            pred_x = parse_x_values(x_pred_text) if x_pred_text.strip() else x
            pred = predict_intervals(model, pred_x, alpha)
            pred = pred.merge(df.rename(columns={"x": "X", "y": "Actual Y"}), how="left", on="X")
            pred = pred[["X", "Actual Y", "Fitted", "Lower CI", "Upper CI", "Lower PI", "Upper PI"]]

            model_tbl = pd.DataFrame({
                "Intercept": [model["intercept"]],
                "Slope": [model["slope"]],
                "Residual SD": [model["s"]],
                "R²": [model["r2"]],
                "N": [len(x)],
            })
            report_table(model_tbl, "Regression model summary", decimals)
            report_table(pred, f"Predictions with {conf}% confidence and prediction intervals", decimals)

            xg = np.linspace(x.min(), x.max(), 220)
            grid = predict_intervals(model, xg, alpha)
            fig_main, ax = plt.subplots(figsize=(FIG_W, FIG_H))
            ax.scatter(x, y, color=PRIMARY_COLOR, s=46, label="Observed")
            ax.plot(xg, grid["Fitted"], color="#111827", lw=2.2, label="Fit")
            ax.fill_between(xg, grid["Lower CI"], grid["Upper CI"], color=BAND_COLOR, alpha=0.28, label=f"{conf}% CI")
            ax.fill_between(xg, grid["Lower PI"], grid["Upper PI"], color=SECONDARY_COLOR, alpha=0.12, label=f"{conf}% PI")
            apply_ax_style(ax, "Linear regression with intervals", x_label, y_label, legend=True)
            st.pyplot(fig_main)

            fig_res = residual_plot(model["fitted"], model["resid"], xlabel="Fitted values", ylabel="Residuals", title="Residuals vs fitted")
            st.pyplot(fig_res)
            fig_qq = qq_plot(model["resid"], title="Normal probability plot of regression residuals")
            st.pyplot(fig_qq)

            conclusion = f"A simple linear regression was fitted to {len(x)} observations. The estimated slope was {model['slope']:.{decimals}f} and R² was {model['r2']:.{decimals}f}. Confidence intervals describe uncertainty in the mean fitted response, while prediction intervals describe the range for a future single observation."
            export_results(
                prefix="regression_intervals",
                report_title="Statistical Analysis Report",
                module_name="Regression Intervals",
                statistical_analysis="A simple linear regression model was fitted to the pasted X and Y data. Model fit was summarized by the intercept, slope, residual standard deviation, and coefficient of determination (R²). Confidence intervals for the fitted mean response and prediction intervals for future observations were then calculated using the t distribution.",
                offer_text="This analysis offers a direct way to quantify linear trends, estimate expected response values at selected X points, compare observed values against fitted values, and distinguish between uncertainty in the mean response and dispersion of individual future values.",
                python_tools="Python tools used here include pandas and numpy for handling pasted data and matrix calculations, scipy.stats for t-based interval calculations and probability plotting, matplotlib for regression and residual figures, openpyxl for Excel export, and reportlab for the PDF-style report.",
                table_map={"Model Summary": model_tbl, "Predictions": pred},
                figure_map={
                    "Regression with confidence and prediction intervals": fig_to_png_bytes(fig_main),
                    "Residuals vs fitted": fig_to_png_bytes(fig_res),
                    "Normal probability plot": fig_to_png_bytes(fig_qq),
                },
                conclusion=conclusion,
                decimals=decimals,
            )
        except Exception as e:
            st.error(str(e))


# -------------------------------------------------
# App 03 Shelf Life Estimator
# -------------------------------------------------
elif app_selection == "03 - Shelf Life Estimator":
    app_header("⏳ App 03 - Shelf Life Estimator", "Estimate shelf life from a linear stability trend and a one-sided confidence band.")
    c1, c2 = st.columns([1.4, 1])
    with c1:
        xy_input = st.text_area("Paste Time and Response data (with or without headers)", height=220)
    with c2:
        limit = st.number_input("Specification limit", value=90.0)
        direction = st.selectbox("Degradation direction", ["Response decreases toward lower limit", "Response increases toward upper limit"])
        conf = st.slider("One-sided confidence (%)", 80, 99, 95)
        decimals = st.slider("Decimals", 1, 8, DEFAULT_DECIMALS, key="sl_dec")

    if xy_input:
        try:
            df, x_label, y_label = parse_xy(xy_input)
            x, y = df["x"].to_numpy(), df["y"].to_numpy()
            model = fit_linear(x, y)
            decreasing = direction.startswith("Response decreases")
            shelf, xg, fitg, band = estimate_shelf_life(model, limit, decreasing=decreasing, confidence=conf / 100, x_upper=max(x.max() * 1.5, x.max() + 1))
            pred = predict_intervals(model, x, 1 - conf / 100)
            table = pd.DataFrame({
                x_label: x,
                f"Actual {y_label}": y,
                f"Fitted {y_label}": pred["Fitted"],
                "Lower CI": pred["Lower CI"],
                "Upper CI": pred["Upper CI"],
                "Lower PI": pred["Lower PI"],
                "Upper PI": pred["Upper PI"],
            })
            summary = pd.DataFrame({
                "Intercept": [model["intercept"]],
                "Slope": [model["slope"]],
                "Residual SD": [model["s"]],
                "R²": [model["r2"]],
                "Estimated Shelf Life": [shelf],
            })
            report_table(summary, "Shelf-life regression summary", decimals)
            report_table(table, "Observed and fitted values", decimals)

            fig_main, ax = plt.subplots(figsize=(FIG_W, FIG_H))
            ax.scatter(x, y, color=PRIMARY_COLOR, s=46, label="Observed")
            ax.plot(xg, fitg, color="#111827", lw=2.2, label="Fit")
            ax.plot(xg, band, color="#dc2626", lw=2.0, ls="--", label=f"{conf}% confidence band")
            ax.axhline(limit, color=SECONDARY_COLOR, lw=2, ls=":", label="Specification limit")
            if not np.isnan(shelf):
                ax.axvline(shelf, color="#16a34a", lw=2, ls="--", label=f"Shelf life = {shelf:.{decimals}f}")
            apply_ax_style(ax, "Shelf-life estimation", x_label, y_label, legend=True)
            st.pyplot(fig_main)

            fig_res = residual_plot(model["fitted"], model["resid"], xlabel="Fitted values", ylabel="Residuals", title="Residuals vs fitted")
            st.pyplot(fig_res)
            fig_qq = qq_plot(model["resid"], title="Normal probability plot of stability residuals")
            st.pyplot(fig_qq)

            conclusion = f"Shelf life was estimated as the first time at which the one-sided confidence band crossed the specification limit. The estimated shelf life was {('-' if np.isnan(shelf) else f'{shelf:.{decimals}f}')}."
            export_results(
                prefix="shelf_life_estimator",
                report_title="Statistical Analysis Report",
                module_name="Shelf Life Estimator",
                statistical_analysis="A simple linear regression model was applied to the response-versus-time stability data. A one-sided confidence band around the fitted mean response was then constructed. Shelf life was estimated as the earliest time point at which the relevant one-sided confidence limit crossed the specification limit.",
                offer_text="This analysis offers a practical way to quantify a stability trend, visualize uncertainty, and derive a conservative shelf-life estimate aligned with common regression-based stability approaches.",
                python_tools="Python tools used here include pandas and numpy for data preparation, scipy.stats for confidence-band calculations, matplotlib for fitted, residual, and normal probability plots, openpyxl for Excel export, and reportlab for the PDF-style report.",
                table_map={"Shelf Life Summary": summary, "Observed and Fitted Values": table},
                figure_map={
                    "Shelf-life regression plot": fig_to_png_bytes(fig_main),
                    "Residuals vs fitted": fig_to_png_bytes(fig_res),
                    "Normal probability plot": fig_to_png_bytes(fig_qq),
                },
                conclusion=conclusion,
                decimals=decimals,
            )
        except Exception as e:
            st.error(str(e))


# -------------------------------------------------
# App 04 Dissolution Comparison (f2)
# -------------------------------------------------
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
            anova = anova_lm(model, typ=2).reset_index().rename(columns={"index": "Source", "sum_sq": "Sum of Squares", "df": "df", "F": "F-Statistic", "PR(>F)": "P-Value"})
            anova["SS (%)"] = anova["Sum of Squares"] / anova["Sum of Squares"].sum() * 100
            summary = d.groupby(["FactorA", "FactorB"])["Response"].agg(["count", "mean", "std", "min", "max"]).reset_index()
            summary.columns = [factor_a, factor_b, "N", "Mean", "Std. Deviation", "Minimum", "Maximum"]
            report_table(anova, "Two-way ANOVA table", decimals)
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
                statistical_analysis="A two-way analysis of variance was fitted to the selected response variable using two chosen categorical factors and their interaction. Sums of squares, F statistics, and p-values were computed from the linear model, and residual diagnostics were generated to support assessment of model assumptions.",
                offer_text="This analysis offers a direct way to quantify the main effects of two factors, test whether their interaction is present, compare cell means, and visualize whether factor effects are consistent across levels of the other factor.",
                python_tools="Python tools used here include pandas and numpy for column selection and aggregation, statsmodels.formula.api and statsmodels.stats.anova for model fitting and ANOVA calculations, matplotlib for interaction and residual plots, openpyxl for Excel export, and reportlab for the PDF-style report.",
                table_map={"ANOVA": anova, "Cell Summary": summary},
                figure_map={
                    "Interaction plot": fig_to_png_bytes(fig_inter),
                    "Residuals vs fitted": fig_to_png_bytes(fig_res),
                    "Normal probability plot": fig_to_png_bytes(fig_qq),
                },
                conclusion="The ANOVA table reports whether Factor A, Factor B, and their interaction contributed significantly to variation in the selected response.",
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
                eig = pd.DataFrame({"Principal Component": ["PC1", "PC2"], "Eigenvalue": pca.explained_variance_, "Variance Explained (%)": exp, "Cumulative Variance (%)": np.cumsum(exp)})
                load_df = pd.DataFrame({"Variable": vars_sel, "PC1": loadings[:, 0], "PC2": loadings[:, 1]})
                report_table(eig, "Eigenvalues and explained variance", decimals)
                report_table(load_df, "Loading matrix", decimals)

                scores_df = pd.DataFrame({"PC1": scores[:, 0], "PC2": scores[:, 1]}, index=X.index)
                if label_col != "(None)":
                    scores_df["Label"] = df.loc[X.index, label_col].astype(str).values
                if group_col != "(None)":
                    scores_df["Group"] = df.loc[X.index, group_col].astype(str).values

                fig_scores, ax = plt.subplots(figsize=(FIG_W, FIG_H))
                if group_col != "(None)":
                    unique_groups = scores_df["Group"].unique()
                    for grp in unique_groups:
                        m = scores_df["Group"] == grp
                        ax.scatter(scores_df.loc[m, "PC1"], scores_df.loc[m, "PC2"], s=46, label=str(grp))
                        draw_conf_ellipse(scores_df.loc[m, ["PC1", "PC2"]].to_numpy(), ax)
                else:
                    ax.scatter(scores_df["PC1"], scores_df["PC2"], s=46, color=PRIMARY_COLOR)
                    draw_conf_ellipse(scores_df[["PC1", "PC2"]].to_numpy(), ax)
                if label_col != "(None)":
                    for _, row in scores_df.iterrows():
                        ax.text(row["PC1"], row["PC2"], str(row["Label"]), fontsize=8)
                ax.axhline(0, color="#64748b", lw=1)
                ax.axvline(0, color="#64748b", lw=1)
                apply_ax_style(ax, "PCA score plot", f"PC1 ({exp[0]:.1f}% var)", f"PC2 ({exp[1]:.1f}% var)", legend=(group_col != "(None)"))
                st.pyplot(fig_scores)

                fig_load, ax2 = plt.subplots(figsize=(FIG_W, FIG_H))
                ax2.axhline(0, color="#64748b", lw=1)
                ax2.axvline(0, color="#64748b", lw=1)
                for i, var in enumerate(vars_sel):
                    ax2.arrow(0, 0, loadings[i, 0], loadings[i, 1], head_width=0.03, length_includes_head=True, color=PRIMARY_COLOR)
                    ax2.text(loadings[i, 0], loadings[i, 1], var)
                apply_ax_style(ax2, "PCA loading plot", "PC1", "PC2")
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
                    conclusion="The PCA score plot summarizes sample-level multivariate patterns, while the loading plot shows which variables are driving separation along PC1 and PC2.",
                    decimals=decimals,
                )
        except Exception as e:
            st.error(str(e))


# -------------------------------------------------
# App 09 DoE / Response Surfaces
# -------------------------------------------------
elif app_selection == "09 - DoE / Response Surfaces":
    app_header("🧪 App 09 - DoE / Response Surfaces", "Fit linear, interaction, or quadratic DoE models and visualize contour and surface plots.")
    data_input = st.text_area("Paste data with headers", height=240)
    decimals = st.slider("Decimals", 1, 8, DEFAULT_DECIMALS, key="doe_dec")
    if data_input:
        try:
            df = parse_pasted_table(data_input, header=True)
            num_cols = get_numeric_columns(df)
            c1, c2, c3 = st.columns([1.35, 1, 1])
            with c1:
                factors = st.multiselect("Numeric factors", num_cols, default=num_cols[: min(2, len(num_cols))])
            with c2:
                response = st.selectbox("Response", [c for c in num_cols if c not in factors] or num_cols)
            with c3:
                model_type = st.selectbox("Model type", ["linear", "interaction", "quadratic"])

            if len(factors) >= 2:
                d = df[factors + [response]].copy()
                for c in factors + [response]:
                    d[c] = to_numeric(d[c])
                d = d.dropna()
                safe_factor_names = [f"F{i+1}" for i in range(len(factors))]
                rename_map = {orig: safe for orig, safe in zip(factors, safe_factor_names)}
                safe_df = d.rename(columns=rename_map).rename(columns={response: "Response"})
                formula = doe_formula(safe_factor_names, model_type=model_type)
                model = smf.ols(formula, data=safe_df).fit()
                anova = anova_lm(model, typ=2).reset_index().rename(columns={"index": "Source", "sum_sq": "Sum of Squares", "df": "df", "F": "F-Statistic", "PR(>F)": "P-Value"})
                anova["SS (%)"] = anova["Sum of Squares"] / anova["Sum of Squares"].sum() * 100
                coef = pd.DataFrame({"Term": model.params.index, "Coefficient": model.params.values, "P-Value": model.pvalues.values})
                report_table(anova, f"DoE ANOVA ({model_type} model)", decimals)
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

                x_vals = np.linspace(d[xfac].min(), d[xfac].max(), 40)
                y_vals = np.linspace(d[yfac].min(), d[yfac].max(), 40)
                xx, yy = np.meshgrid(x_vals, y_vals)
                grid = pd.DataFrame({xfac: xx.ravel(), yfac: yy.ravel()})
                for f in other_factors:
                    grid[f] = fixed_vals[f]
                safe_grid = grid.rename(columns=rename_map)
                zz = model.predict(safe_grid).to_numpy().reshape(xx.shape)

                fig_contour, ax = plt.subplots(figsize=(FIG_W, FIG_H))
                cs = ax.contourf(xx, yy, zz, levels=20, cmap="viridis")
                fig_contour.colorbar(cs, ax=ax, label=response)
                ax.scatter(d[xfac], d[yfac], c="white", edgecolor="black", s=34)
                apply_ax_style(ax, f"Contour plot for {response}", xfac, yfac)
                st.pyplot(fig_contour)

                fig_surface = plt.figure(figsize=(FIG_W, FIG_H + 0.5))
                ax3 = fig_surface.add_subplot(111, projection="3d")
                surf = ax3.plot_surface(xx, yy, zz, cmap="viridis", edgecolor="none", alpha=0.88)
                ax3.scatter(d[xfac], d[yfac], d[response], c="black", s=26)
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
                    statistical_analysis="A design-of-experiments style regression model was fitted to the selected numeric response using the chosen numeric factors. Depending on the selected option, the model included linear terms only, linear plus interactions, or a quadratic response-surface structure. ANOVA, model coefficients, contour plots, surface plots, and residual diagnostics were generated from the fitted model.",
                    offer_text="This analysis offers a way to quantify factor effects, inspect interactions, model curvature, and visualize the response surface over two selected factors while fixing any remaining factors at chosen values.",
                    python_tools="Python tools used here include pandas and numpy for selecting factor and response columns, statsmodels for model fitting and ANOVA, matplotlib for contour, 3D surface, and residual diagnostic plots, openpyxl for Excel export, and reportlab for the PDF-style report.",
                    table_map={"DoE ANOVA": anova, "Coefficients": coef},
                    figure_map={
                        "Contour plot": fig_to_png_bytes(fig_contour),
                        "Response surface": fig_to_png_bytes(fig_surface),
                        "Residuals vs fitted": fig_to_png_bytes(fig_res),
                        "Normal probability plot": fig_to_png_bytes(fig_qq),
                    },
                    conclusion="The fitted DoE model can be used to assess influential factors, detect interactions or curvature, and visualize predicted response behavior across the chosen design space.",
                    decimals=decimals,
                )
        except Exception as e:
            st.error(str(e))

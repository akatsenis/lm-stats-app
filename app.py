import re
from io import StringIO

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
from statsmodels.stats.oneway import anova_oneway

from sklearn.decomposition import PCA

# -----------------------------
# Page config & style
# -----------------------------
st.set_page_config(page_title="lm Stats Suite", page_icon="🔬", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.0rem; padding-bottom: 2rem;}
    .app-header {
        border:1px solid #e2e8f0;
        border-radius:12px;
        padding:16px 20px;
        background: linear-gradient(90deg, #f8fafc 0%, #ffffff 100%);
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .app-title {font-size: 1.7rem; font-weight: 700; margin-bottom: 0.2rem;}
    .app-sub {color:#475569; font-size:0.95rem;}
    .metric-card {
        border:1px solid #e2e8f0; border-radius:12px; padding:14px 16px; background:#fff;
    }
    .report-table table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.95rem;
        margin-bottom: 1rem;
        background-color: white;
    }
    .report-table caption {
        text-align: left;
        font-weight: 700;
        font-size: 1rem;
        margin-bottom: 0.55rem;
        color: #111827;
    }
    .report-table thead th {
        border-top: 2px solid #111827;
        border-bottom: 1px solid #111827;
        padding: 8px 12px;
        text-align: center;
        background-color: #f8fafc;
    }
    .report-table tbody td {
        padding: 8px 12px;
        text-align: center;
        border: none;
    }
    .report-table tbody tr:last-child td {
        border-bottom: 2px solid #111827;
    }
    div[data-testid='stMetric'] {
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 10px 12px;
        background-color: #fff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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


# -----------------------------
# General helpers
# -----------------------------
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
            if df.shape[1] >= 1:
                if header:
                    df.columns = [str(c).strip() for c in df.columns]
                return df.dropna(how="all").reset_index(drop=True)
        except Exception:
            continue
    return None


def parse_xy(text):
    df = parse_pasted_table(text, header=False)
    if df is None or df.shape[1] < 2:
        raise ValueError("Paste two columns: X and Y.")
    df = df.iloc[:, :2].copy()
    df.columns = ["x", "y"]
    # remove header row if non-numeric
    if to_numeric(df.iloc[0]).isna().any():
        df = df.iloc[1:].reset_index(drop=True)
    df["x"] = to_numeric(df["x"])
    df["y"] = to_numeric(df["y"])
    df = df.dropna().sort_values("x").reset_index(drop=True)
    if len(df) < 3 or df["x"].nunique() < 2:
        raise ValueError("At least 3 valid rows and 2 unique X values are required.")
    return df


def parse_one_col(text):
    df = parse_pasted_table(text, header=False)
    if df is None:
        return np.array([])
    vals = to_numeric(df.iloc[:, 0]).dropna().to_numpy()
    return vals


def parse_x_values(text):
    parts = re.split(r"[\s,;\t]+", str(text).strip())
    vals = []
    for p in parts:
        if p:
            vals.append(float(p))
    return np.array(vals, dtype=float)


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


def report_table(df, caption="", decimals=3):
    disp = df.copy()
    sty = (
        disp.style
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
    st.markdown(f"<div class='report-table'>{sty.to_html()}</div>", unsafe_allow_html=True)


def csv_download(df, name="table.csv"):
    st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), file_name=name, mime="text/csv")


# -----------------------------
# Regression helpers
# -----------------------------
def fit_linear(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    intercept, slope = beta
    fit = X @ beta
    resid = y - fit
    df = n - 2
    s = np.sqrt(np.sum(resid ** 2) / df)
    ss_tot = np.sum((y - y.mean()) ** 2)
    ss_res = np.sum(resid ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {
        "intercept": intercept,
        "slope": slope,
        "XtX_inv": XtX_inv,
        "fit": fit,
        "resid": resid,
        "s": s,
        "df": df,
        "r2": r2,
        "xbar": x.mean(),
        "n": n,
    }


def predict_intervals(model, x_vals, alpha=0.05):
    x_vals = np.asarray(x_vals, dtype=float)
    X0 = np.column_stack([np.ones(len(x_vals)), x_vals])
    fit = model["intercept"] + model["slope"] * x_vals
    h = np.sum((X0 @ model["XtX_inv"]) * X0, axis=1)
    tcrit = t.ppf(1 - alpha / 2, model["df"])
    se_mean = model["s"] * np.sqrt(h)
    se_pred = model["s"] * np.sqrt(1 + h)
    return pd.DataFrame({
        "x": x_vals,
        "fit": fit,
        "ci_lower": fit - tcrit * se_mean,
        "ci_upper": fit + tcrit * se_mean,
        "pi_lower": fit - tcrit * se_pred,
        "pi_upper": fit + tcrit * se_pred,
    })


def shelf_life(model, limit, side="lower", alpha=0.05, x_max=None):
    # approximate by evaluating one-sided confidence band on fine grid
    x_grid = np.linspace(0, x_max if x_max is not None else 1.5 *  max(1, model["n"]), 1000)
    X0 = np.column_stack([np.ones(len(x_grid)), x_grid])
    fit = model["intercept"] + model["slope"] * x_grid
    h = np.sum((X0 @ model["XtX_inv"]) * X0, axis=1)
    tcrit = t.ppf(1 - alpha, model["df"])  # one-sided
    se = model["s"] * np.sqrt(h)
    if side == "lower":
        band = fit - tcrit * se
        idx = np.where(band <= limit)[0]
    else:
        band = fit + tcrit * se
        idx = np.where(band >= limit)[0]
    return (x_grid[idx[0]] if len(idx) else np.nan), x_grid, band


# -----------------------------
# DoE helpers
# -----------------------------
def build_formula(response, factors, model_type="interaction"):
    q = lambda x: f'Q("{x}")'
    if model_type == "linear":
        terms = [q(f) for f in factors]
    elif model_type == "interaction":
        terms = [q(f) for f in factors]
        for i in range(len(factors)):
            for j in range(i + 1, len(factors)):
                terms.append(f"{q(factors[i])}:{q(factors[j])}")
    elif model_type == "quadratic":
        terms = [q(f) for f in factors]
        for i in range(len(factors)):
            terms.append(f"I({q(factors[i])}**2)")
        for i in range(len(factors)):
            for j in range(i + 1, len(factors)):
                terms.append(f"{q(factors[i])}:{q(factors[j])}")
    return f'{q(response)} ~ ' + ' + '.join(terms)


def norm_ellipse(scores, ax, color="tab:blue"):
    if scores.shape[0] < 3:
        return
    cov = np.cov(scores[:, 0], scores[:, 1])
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    width, height = 2 * np.sqrt(5.991 * eigvals)  # 95% ellipse for 2 dof
    ell = Ellipse(xy=scores.mean(axis=0), width=width, height=height, angle=angle,
                  edgecolor=color, facecolor='none', lw=2)
    ax.add_patch(ell)


def tolerance_interval_normal(data, p=0.95, conf=0.95, two_sided=True):
    data = np.asarray(data, dtype=float)
    n = len(data)
    mean = data.mean()
    sd = data.std(ddof=1)
    if n < 2:
        return np.nan, np.nan, np.nan
    if two_sided:
        g = norm.ppf((1 + p) / 2)
        k = g * np.sqrt((n - 1) * (1 + 1/n) / chi2.ppf(1 - conf, n - 1))
        return mean, mean - k * sd, mean + k * sd
    else:
        zp = norm.ppf(p)
        k = nct.ppf(conf, n - 1, np.sqrt(n) * zp) / np.sqrt(n)
        return mean, mean - k * sd, mean + k * sd


# -----------------------------
# Sidebar
# -----------------------------
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

st.sidebar.divider()
st.sidebar.info("Paste data directly from Excel into the tools. Tables and figures are formatted for reports.")


# -----------------------------
# App 01 Descriptive Statistics
# -----------------------------
if app_selection == "01 - Descriptive Statistics":
    app_header("📊 App 01 - Descriptive Statistics", "Paste data from Excel and generate report-ready summary tables.")

    data_input = st.text_area("Data (paste with headers from Excel)", height=220)
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
            decimals = st.slider("Decimals", 1, 8, 3)

            if st.button("Run descriptive statistics", type="primary"):
                if not selected_vars:
                    st.error("Please select at least one numeric variable.")
                else:
                    tmp = df.copy()
                    for v in selected_vars:
                        tmp[v] = to_numeric(tmp[v])

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

                    groups = [g for g in [group1, group2] if g != "(None)"]
                    if groups:
                        out = tmp.groupby(groups)[selected_vars].apply(lambda g: g.apply(calc_stats)).unstack(level=-1)
                        if len(selected_vars) == 1:
                            out.columns = out.columns.droplevel(0)
                        out = out.reset_index()
                    else:
                        out = tmp[selected_vars].apply(calc_stats).T.reset_index().rename(columns={"index": "Variable"})

                    report_table(out, "Descriptive statistics", decimals=decimals)
                    csv_download(out, "descriptive_statistics.csv")
        else:
            st.error("Could not parse the pasted data.")


# -----------------------------
# App 02 Regression Intervals
# -----------------------------
elif app_selection == "02 - Regression Intervals":
    app_header("📈 App 02 - Regression Intervals", "Fit a straight line and compute confidence and prediction intervals.")

    col1, col2 = st.columns([1.3, 1])
    with col1:
        xy_input = st.text_area("Paste X and Y data (two columns)", height=220)
    with col2:
        x_pred_text = st.text_area("X values for predictions (space/comma separated)", "")
        conf = st.slider("Confidence level (%)", 80, 99, 95)
        decimals = st.slider("Decimals", 1, 8, 3)

    if xy_input:
        try:
            df = parse_xy(xy_input)
            x = df["x"].to_numpy()
            y = df["y"].to_numpy()
            model = fit_linear(x, y)
            alpha = 1 - conf / 100
            x_grid = np.linspace(x.min(), x.max(), 200)
            pred_grid = predict_intervals(model, x_grid, alpha)

            model_tbl = pd.DataFrame({
                "Intercept": [model["intercept"]],
                "Slope": [model["slope"]],
                "Residual SD": [model["s"]],
                "R²": [model["r2"]],
                "N": [len(x)]
            })
            report_table(model_tbl, "Regression model summary", decimals=decimals)

            if x_pred_text.strip():
                x_vals = parse_x_values(x_pred_text)
            else:
                x_vals = x
            pred = predict_intervals(model, x_vals, alpha)
            pred = pred.merge(df.rename(columns={"y": "Actual Y"}), how="left", left_on="x", right_on="x")
            pred = pred[["x", "Actual Y", "fit", "ci_lower", "ci_upper", "pi_lower", "pi_upper"]]
            pred.columns = ["X Value", "Actual Y", "Fitted Y", "Lower CI", "Upper CI", "Lower PI", "Upper PI"]
            report_table(pred, f"Predicted values with {conf}% confidence and prediction intervals", decimals=decimals)
            csv_download(pred, "regression_intervals.csv")

            fig, ax = plt.subplots(figsize=(8.5, 5.5))
            ax.scatter(x, y, c="#1f77b4", s=45, label="Observed")
            ax.plot(x_grid, pred_grid["fit"], color="#111827", lw=2.2, label="Fit")
            ax.fill_between(x_grid, pred_grid["ci_lower"], pred_grid["ci_upper"], color="#60a5fa", alpha=0.25, label=f"{conf}% CI")
            ax.fill_between(x_grid, pred_grid["pi_lower"], pred_grid["pi_upper"], color="#cbd5e1", alpha=0.35, label=f"{conf}% PI")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_title("Linear regression with confidence and prediction intervals")
            ax.grid(alpha=0.25)
            ax.legend(frameon=False)
            st.pyplot(fig)

        except Exception as e:
            st.error(str(e))


# -----------------------------
# App 03 Shelf Life Estimator
# -----------------------------
elif app_selection == "03 - Shelf Life Estimator":
    app_header("⏳ App 03 - Shelf Life Estimator", "Estimate shelf life by linear regression and one-sided confidence band.")

    col1, col2 = st.columns([1.3, 1])
    with col1:
        xy_input = st.text_area("Paste Time and Response data (two columns)", height=220)
    with col2:
        limit = st.number_input("Specification limit", value=90.0)
        direction = st.selectbox("Degradation direction", ["Response decreases toward lower limit", "Response increases toward upper limit"])
        conf = st.slider("One-sided confidence (%)", 80, 99, 95)
        decimals = st.slider("Decimals", 1, 8, 3, key="shelf_dec")

    if xy_input:
        try:
            df = parse_xy(xy_input)
            x = df["x"].to_numpy()
            y = df["y"].to_numpy()
            model = fit_linear(x, y)
            x_max = max(x.max() * 1.5, x.max() + 1)
            side = "lower" if direction.startswith("Response decreases") else "upper"
            shelf, grid, band = shelf_life(model, limit, side=side, alpha=1-conf/100, x_max=x_max)
            pred = predict_intervals(model, x, alpha=1-conf/100)
            table = pd.DataFrame({
                "Time": x,
                "Actual Response": y,
                "Fitted Response": pred["fit"],
                "Lower CI": pred["ci_lower"],
                "Upper CI": pred["ci_upper"],
                "Lower PI": pred["pi_lower"],
                "Upper PI": pred["pi_upper"],
            })
            report_table(table, f"Observed and fitted values with {conf}% intervals", decimals=decimals)

            summary = pd.DataFrame({
                "Intercept": [model["intercept"]],
                "Slope": [model["slope"]],
                "Residual SD": [model["s"]],
                "R²": [model["r2"]],
                "Estimated Shelf Life": [shelf],
            })
            report_table(summary, "Shelf-life regression summary", decimals=decimals)
            csv_download(table, "shelf_life_results.csv")

            xg = np.linspace(x.min(), x_max, 250)
            fitg = model["intercept"] + model["slope"] * xg
            X0 = np.column_stack([np.ones(len(xg)), xg])
            h = np.sum((X0 @ model["XtX_inv"]) * X0, axis=1)
            tcrit = t.ppf(conf/100 if side == "lower" else conf/100, model["df"])
            se = model["s"] * np.sqrt(h)
            lower = fitg - t.ppf(conf/100, model["df"]) * se
            upper = fitg + t.ppf(conf/100, model["df"]) * se

            fig, ax = plt.subplots(figsize=(8.5, 5.5))
            ax.scatter(x, y, s=45, c="#1f77b4", label="Observed")
            ax.plot(xg, fitg, c="#111827", lw=2.2, label="Fit")
            if side == "lower":
                ax.plot(grid, band, c="#dc2626", ls="--", lw=2, label=f"{conf}% lower confidence band")
            else:
                ax.plot(grid, band, c="#dc2626", ls="--", lw=2, label=f"{conf}% upper confidence band")
            ax.axhline(limit, color="#ea580c", ls=":", lw=2, label="Specification limit")
            if not np.isnan(shelf):
                ax.axvline(shelf, color="#16a34a", ls="--", lw=2, label=f"Estimated shelf life = {shelf:.{decimals}f}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Response")
            ax.set_title("Shelf-life estimation")
            ax.grid(alpha=0.25)
            ax.legend(frameon=False)
            st.pyplot(fig)
        except Exception as e:
            st.error(str(e))


# -----------------------------
# App 04 Dissolution Comparison (f2)
# -----------------------------
elif app_selection == "04 - Dissolution Comparison (f2)":
    app_header("💊 App 04 - Dissolution Comparison (f2)", "Compare reference and test dissolution profiles using the similarity factor f₂.")

    st.markdown("Paste **Reference** and **Test** tables with the first column as time and the remaining columns as unit values.")
    c1, c2 = st.columns(2)
    with c1:
        ref_text = st.text_area("Reference data", height=220)
    with c2:
        test_text = st.text_area("Test data", height=220)

    decimals = st.slider("Decimals", 1, 8, 2, key="f2dec")

    if ref_text and test_text:
        try:
            ref = parse_pasted_table(ref_text, header=True)
            test = parse_pasted_table(test_text, header=True)
            ref.iloc[:, 0] = to_numeric(ref.iloc[:, 0])
            test.iloc[:, 0] = to_numeric(test.iloc[:, 0])
            ref = ref.dropna().reset_index(drop=True)
            test = test.dropna().reset_index(drop=True)

            if not np.allclose(ref.iloc[:, 0], test.iloc[:, 0]):
                raise ValueError("Time points in reference and test must match.")

            time = to_numeric(ref.iloc[:, 0])
            ref_vals = ref.iloc[:, 1:].apply(to_numeric)
            test_vals = test.iloc[:, 1:].apply(to_numeric)
            if ref_vals.shape[1] < 1 or test_vals.shape[1] < 1:
                raise ValueError("Provide at least one unit column for reference and test.")

            sumdf = pd.DataFrame({
                "Time": time,
                "Ref. Mean": ref_vals.mean(axis=1),
                "Ref. SD": ref_vals.std(axis=1, ddof=1),
                "Ref. CV (%)": ref_vals.std(axis=1, ddof=1) / ref_vals.mean(axis=1) * 100,
                "Test Mean": test_vals.mean(axis=1),
                "Test SD": test_vals.std(axis=1, ddof=1),
                "Test CV (%)": test_vals.std(axis=1, ddof=1) / test_vals.mean(axis=1) * 100,
            })
            sumdf["Absolute Difference"] = (sumdf["Ref. Mean"] - sumdf["Test Mean"]).abs()
            sumdf["Squared Difference"] = (sumdf["Ref. Mean"] - sumdf["Test Mean"]) ** 2
            n = len(sumdf)
            f2 = 50 * np.log10((1 + sumdf["Squared Difference"].mean()) ** -0.5 * 100)

            report_table(sumdf, "Dissolution profile summary", decimals=decimals)
            f2_tbl = pd.DataFrame({"Similarity Factor f₂": [f2], "Conclusion": ["Similar" if f2 >= 50 else "Not Similar"]})
            report_table(f2_tbl, "Overall f₂ assessment", decimals=decimals)
            csv_download(sumdf, "dissolution_f2_summary.csv")

            fig, ax = plt.subplots(figsize=(8.5, 5.5))
            ax.errorbar(time, sumdf["Ref. Mean"], yerr=sumdf["Ref. SD"], fmt='o-', capsize=4, lw=2, label="Reference")
            ax.errorbar(time, sumdf["Test Mean"], yerr=sumdf["Test SD"], fmt='s-', capsize=4, lw=2, label="Test")
            ax.set_xlabel("Time")
            ax.set_ylabel("Dissolved (%)")
            ax.set_title(f"Dissolution profiles (f₂ = {f2:.{decimals}f})")
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.25)
            ax.legend(frameon=False)
            st.pyplot(fig)
        except Exception as e:
            st.error(str(e))


# -----------------------------
# App 05 Two-Sample Tests
# -----------------------------
elif app_selection == "05 - Two-Sample Tests":
    app_header("⚖️ App 05 - Two-Sample Tests", "Independent or paired comparison with parametric and non-parametric options.")

    mode = st.radio("Comparison type", ["Independent samples", "Paired samples"], horizontal=True)
    c1, c2 = st.columns(2)
    with c1:
        x_text = st.text_area("Sample 1", height=220)
    with c2:
        y_text = st.text_area("Sample 2", height=220)
    alpha = st.slider("Significance level α", 0.001, 0.10, 0.05, step=0.001)
    decimals = st.slider("Decimals", 1, 8, 3, key="twos_dec")

    if x_text and y_text:
        try:
            x = parse_one_col(x_text)
            y = parse_one_col(y_text)
            if len(x) < 2 or len(y) < 2:
                raise ValueError("Each sample must contain at least two numeric values.")
            if mode == "Paired samples" and len(x) != len(y):
                raise ValueError("Paired samples require equal length.")

            def ad_test(a):
                stat, p = normal_ad(a)
                return stat, p, (p >= alpha)

            a1, p1, n1 = ad_test(x)
            a2, p2, n2 = ad_test(y)
            lev_stat, lev_p = stats.levene(x, y)
            equal_var = lev_p >= alpha

            desc = pd.DataFrame({
                "Sample": ["Sample 1", "Sample 2"],
                "N": [len(x), len(y)],
                "Mean": [np.mean(x), np.mean(y)],
                "Std. Deviation": [np.std(x, ddof=1), np.std(y, ddof=1)],
                "Median": [np.median(x), np.median(y)],
                "Minimum": [np.min(x), np.min(y)],
                "Maximum": [np.max(x), np.max(y)],
                "AD A*": [a1, a2],
                "AD P-Value": [p1, p2],
                "Normal?": ["Yes" if n1 else "No", "Yes" if n2 else "No"],
            })
            report_table(desc, "Sample summary and normality check", decimals=decimals)

            if mode == "Independent samples":
                t_stat, t_p = stats.ttest_ind(x, y, equal_var=equal_var)
                mw_stat, mw_p = stats.mannwhitneyu(x, y, alternative="two-sided")
                result = pd.DataFrame({
                    "Test": ["Levene test", "Student/Welch t-test", "Mann–Whitney U"],
                    "Statistic": [lev_stat, t_stat, mw_stat],
                    "P-Value": [lev_p, t_p, mw_p],
                    "Conclusion": [
                        "Equal variances" if equal_var else "Unequal variances",
                        "Significant" if t_p < alpha else "Not significant",
                        "Significant" if mw_p < alpha else "Not significant",
                    ]
                })
            else:
                diff = x - y
                ad_d, p_d, n_d = ad_test(diff)
                t_stat, t_p = stats.ttest_rel(x, y)
                w_stat, w_p = stats.wilcoxon(x, y)
                result = pd.DataFrame({
                    "Test": ["AD test of paired differences", "Paired t-test", "Wilcoxon signed-rank"],
                    "Statistic": [ad_d, t_stat, w_stat],
                    "P-Value": [p_d, t_p, w_p],
                    "Conclusion": [
                        "Normal differences" if n_d else "Non-normal differences",
                        "Significant" if t_p < alpha else "Not significant",
                        "Significant" if w_p < alpha else "Not significant",
                    ]
                })

            report_table(result, f"Two-sample test results (α = {alpha})", decimals=decimals)
            csv_download(result, "two_sample_tests.csv")

            fig, ax = plt.subplots(figsize=(8.5, 5.5))
            ax.boxplot([x, y], labels=["Sample 1", "Sample 2"], patch_artist=True)
            ax.set_title("Two-sample comparison")
            ax.set_ylabel("Value")
            ax.grid(axis='y', alpha=0.25)
            st.pyplot(fig)

        except Exception as e:
            st.error(str(e))


# -----------------------------
# App 06 Two-Way ANOVA
# -----------------------------
elif app_selection == "06 - Two-Way ANOVA":
    app_header("📐 App 06 - Two-Way ANOVA", "Analyze two factors and their interaction for a numeric response.")

    data_input = st.text_area("Paste data with headers (Factor A, Factor B, Response)", height=220)
    decimals = st.slider("Decimals", 1, 8, 3, key="anova2_dec")
    if data_input:
        try:
            df = parse_pasted_table(data_input, header=True)
            all_cols = list(df.columns)
            c1, c2, c3 = st.columns(3)
            with c1:
                fa = st.selectbox("Factor A", all_cols)
            with c2:
                fb = st.selectbox("Factor B", [c for c in all_cols if c != fa])
            with c3:
                response = st.selectbox("Response", [c for c in all_cols if c not in [fa, fb]])

            d = df[[fa, fb, response]].copy()
            d[response] = to_numeric(d[response])
            d = d.dropna()

            model = smf.ols(f'Q("{response}") ~ C(Q("{fa}")) * C(Q("{fb}"))', data=d).fit()
            anova = anova_lm(model, typ=2).reset_index().rename(columns={"index": "Source", "sum_sq": "Sum of Squares", "df": "df", "F": "F", "PR(>F)": "P-Value"})
            anova["SS (%)"] = anova["Sum of Squares"] / anova["Sum of Squares"].sum() * 100
            report_table(anova, "Two-way ANOVA table", decimals=decimals)

            means = d.groupby([fa, fb])[response].agg(["count", "mean", "std", "min", "max"]).reset_index()
            means.columns = [fa, fb, "N", "Mean", "Std. Deviation", "Minimum", "Maximum"]
            report_table(means, "Cell summary statistics", decimals=decimals)
            csv_download(anova, "two_way_anova.csv")

            fig, ax = plt.subplots(figsize=(8.5, 5.5))
            for level in d[fb].unique():
                subset = d[d[fb] == level]
                mean_by_fa = subset.groupby(fa)[response].mean().reset_index()
                ax.plot(mean_by_fa[fa].astype(str), mean_by_fa[response], marker='o', lw=2, label=f"{fb} = {level}")
            ax.set_xlabel(fa)
            ax.set_ylabel(response)
            ax.set_title("Interaction plot")
            ax.grid(alpha=0.25)
            ax.legend(frameon=False)
            st.pyplot(fig)
        except Exception as e:
            st.error(str(e))


# -----------------------------
# App 07 Tolerance & Confidence Intervals
# -----------------------------
elif app_selection == "07 - Tolerance & Confidence Intervals":
    app_header("🎯 App 07 - Tolerance & Confidence Intervals", "Compute confidence intervals and normal-theory tolerance intervals for one or two samples.")

    c1, c2 = st.columns(2)
    with c1:
        x_text = st.text_area("Sample 1", height=220)
    with c2:
        y_text = st.text_area("Sample 2 (optional)", height=220)

    confidence = st.slider("Confidence level for CI/TI (%)", 80, 99, 95)
    coverage = st.slider("Population coverage for TI (%)", 80, 99, 95)
    paired = st.checkbox("Paired samples for mean difference CI", value=False)
    decimals = st.slider("Decimals", 1, 8, 3, key="ti_dec")

    if x_text:
        try:
            x = parse_one_col(x_text)
            y = parse_one_col(y_text) if y_text.strip() else None
            alpha = 1 - confidence / 100
            p = coverage / 100
            conf = confidence / 100

            def mean_ci(a):
                n = len(a)
                m = np.mean(a)
                s = np.std(a, ddof=1)
                se = s / np.sqrt(n)
                tcrit = t.ppf(1 - alpha/2, n-1)
                return m, m - tcrit*se, m + tcrit*se, s

            m1, l1, u1, s1 = mean_ci(x)
            m, tl1, tu1 = tolerance_interval_normal(x, p=p, conf=conf, two_sided=True)
            summary = [{
                "Sample": "Sample 1", "N": len(x), "Mean": m1, "Std. Deviation": s1,
                f"{confidence}% CI Lower": l1, f"{confidence}% CI Upper": u1,
                f"{coverage}%/{confidence}% TI Lower": tl1, f"{coverage}%/{confidence}% TI Upper": tu1
            }]

            if y is not None and len(y) > 1:
                m2, l2, u2, s2 = mean_ci(y)
                m, tl2, tu2 = tolerance_interval_normal(y, p=p, conf=conf, two_sided=True)
                summary.append({
                    "Sample": "Sample 2", "N": len(y), "Mean": m2, "Std. Deviation": s2,
                    f"{confidence}% CI Lower": l2, f"{confidence}% CI Upper": u2,
                    f"{coverage}%/{confidence}% TI Lower": tl2, f"{coverage}%/{confidence}% TI Upper": tu2
                })
            out = pd.DataFrame(summary)
            report_table(out, "Confidence and tolerance intervals", decimals=decimals)

            if y is not None and len(y) > 1:
                if paired:
                    if len(x) != len(y):
                        raise ValueError("Paired CI requires equal sample sizes.")
                    d = x - y
                    md = np.mean(d)
                    sd = np.std(d, ddof=1)
                    se = sd / np.sqrt(len(d))
                    tcrit = t.ppf(1 - alpha/2, len(d)-1)
                    l, u = md - tcrit*se, md + tcrit*se
                    diff_tbl = pd.DataFrame({"Mean Difference": [md], f"{confidence}% CI Lower": [l], f"{confidence}% CI Upper": [u]})
                else:
                    nx, ny = len(x), len(y)
                    mx, my = np.mean(x), np.mean(y)
                    sx2, sy2 = np.var(x, ddof=1), np.var(y, ddof=1)
                    se = np.sqrt(sx2/nx + sy2/ny)
                    dfw = (sx2/nx + sy2/ny)**2 / ((sx2/nx)**2/(nx-1) + (sy2/ny)**2/(ny-1))
                    tcrit = t.ppf(1 - alpha/2, dfw)
                    d = mx - my
                    l, u = d - tcrit*se, d + tcrit*se
                    diff_tbl = pd.DataFrame({"Mean Difference": [d], f"{confidence}% CI Lower": [l], f"{confidence}% CI Upper": [u]})
                report_table(diff_tbl, "Confidence interval for mean difference", decimals=decimals)
                csv_download(out, "tolerance_confidence_intervals.csv")

            fig, ax = plt.subplots(figsize=(8.5, 5.5))
            data = [x] if y is None else [x, y]
            labels = ["Sample 1"] if y is None else ["Sample 1", "Sample 2"]
            ax.boxplot(data, labels=labels, patch_artist=True)
            ax.set_ylabel("Value")
            ax.set_title("Sample distributions")
            ax.grid(axis='y', alpha=0.25)
            st.pyplot(fig)
        except Exception as e:
            st.error(str(e))


# -----------------------------
# App 08 PCA Analysis
# -----------------------------
elif app_selection == "08 - PCA Analysis":
    app_header("🌐 App 08 - PCA Analysis", "Principal component analysis with scores and loadings.")

    data_input = st.text_area("Paste data with headers (numeric variables plus optional label/group columns)", height=240)
    decimals = st.slider("Decimals", 1, 8, 3, key="pca_dec")
    if data_input:
        try:
            df = parse_pasted_table(data_input, header=True)
            num_cols = get_numeric_columns(df)
            all_cols = list(df.columns)
            c1, c2, c3 = st.columns([1.2, 1, 1])
            with c1:
                vars_sel = st.multiselect("Numeric variables", num_cols, default=num_cols)
            with c2:
                label_col = st.selectbox("Label column (optional)", ["(None)"] + all_cols)
            with c3:
                group_col = st.selectbox("Group column (optional)", ["(None)"] + all_cols)

            if len(vars_sel) < 2:
                st.warning("Select at least two numeric variables.")
            else:
                X = df[vars_sel].apply(to_numeric).dropna()
                z = (X - X.mean()) / X.std(ddof=1)
                pca = PCA(n_components=2)
                scores = pca.fit_transform(z)
                loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
                exp = pca.explained_variance_ratio_ * 100

                eig = pd.DataFrame({
                    "Principal Component": ["PC1", "PC2"],
                    "Eigenvalue": pca.explained_variance_,
                    "Variance Explained (%)": exp,
                    "Cumulative Variance (%)": np.cumsum(exp)
                })
                report_table(eig, "Eigenvalues and explained variance", decimals=decimals)

                load_df = pd.DataFrame({
                    "Variable": vars_sel,
                    "PC1": loadings[:, 0],
                    "PC2": loadings[:, 1],
                })
                report_table(load_df, "Loadings", decimals=decimals)
                csv_download(load_df, "pca_loadings.csv")

                scores_df = pd.DataFrame({"PC1": scores[:, 0], "PC2": scores[:, 1]})
                if label_col != "(None)":
                    scores_df["Label"] = df.loc[X.index, label_col].values
                if group_col != "(None)":
                    scores_df["Group"] = df.loc[X.index, group_col].values

                fig, ax = plt.subplots(figsize=(8.5, 5.5))
                if group_col != "(None)":
                    groups = scores_df["Group"].astype(str).unique()
                    for g in groups:
                        m = scores_df["Group"].astype(str) == g
                        ax.scatter(scores_df.loc[m, "PC1"], scores_df.loc[m, "PC2"], label=g, s=45)
                        norm_ellipse(scores_df.loc[m, ["PC1", "PC2"]].to_numpy(), ax)
                else:
                    ax.scatter(scores_df["PC1"], scores_df["PC2"], s=45)
                    norm_ellipse(scores_df[["PC1", "PC2"]].to_numpy(), ax)
                if label_col != "(None)":
                    for _, row in scores_df.iterrows():
                        ax.text(row["PC1"], row["PC2"], str(row["Label"]), fontsize=8)
                ax.axhline(0, color='grey', lw=1)
                ax.axvline(0, color='grey', lw=1)
                ax.set_xlabel(f"PC1 ({exp[0]:.1f}% var)")
                ax.set_ylabel(f"PC2 ({exp[1]:.1f}% var)")
                ax.set_title("PCA score plot")
                ax.grid(alpha=0.25)
                if group_col != "(None)":
                    ax.legend(frameon=False)
                st.pyplot(fig)

                fig2, ax2 = plt.subplots(figsize=(8.5, 5.5))
                ax2.axhline(0, color='grey', lw=1)
                ax2.axvline(0, color='grey', lw=1)
                for i, var in enumerate(vars_sel):
                    ax2.arrow(0, 0, loadings[i, 0], loadings[i, 1], head_width=0.03, length_includes_head=True)
                    ax2.text(loadings[i, 0], loadings[i, 1], var)
                ax2.set_xlabel("PC1")
                ax2.set_ylabel("PC2")
                ax2.set_title("PCA loading plot")
                ax2.grid(alpha=0.25)
                st.pyplot(fig2)
        except Exception as e:
            st.error(str(e))


# -----------------------------
# App 09 DoE / Response Surfaces
# -----------------------------
elif app_selection == "09 - DoE / Response Surfaces":
    app_header("🧪 App 09 - DoE / Response Surfaces", "Fit linear, interaction, or quadratic models and generate contour/surface plots.")

    data_input = st.text_area("Paste data with headers (factor columns and one or more responses)", height=240)
    decimals = st.slider("Decimals", 1, 8, 3, key="doe_dec")
    if data_input:
        try:
            df = parse_pasted_table(data_input, header=True)
            num_cols = get_numeric_columns(df)
            all_cols = list(df.columns)
            c1, c2, c3 = st.columns([1.4, 1.1, 1])
            with c1:
                factors = st.multiselect("Select numeric factors", num_cols, default=num_cols[:2])
            with c2:
                response = st.selectbox("Response", [c for c in all_cols if c not in factors] or all_cols)
            with c3:
                model_type = st.selectbox("Model type", ["linear", "interaction", "quadratic"])

            if len(factors) < 2:
                st.warning("Select at least two numeric factors to generate response surfaces.")
            else:
                d = df[factors + [response]].copy()
                for c in factors + [response]:
                    d[c] = to_numeric(d[c])
                d = d.dropna()
                formula = build_formula(response, factors, model_type)
                model = smf.ols(formula, data=d).fit()
                anova = anova_lm(model, typ=2).reset_index().rename(columns={"index": "Source", "sum_sq": "Sum of Squares", "df": "df", "F": "F", "PR(>F)": "P-Value"})
                anova["SS (%)"] = anova["Sum of Squares"] / anova["Sum of Squares"].sum() * 100
                report_table(anova, f"DoE ANOVA ({model_type} model)", decimals=decimals)

                coef = model.params.reset_index()
                coef.columns = ["Term", "Coefficient"]
                coef["P-Value"] = model.pvalues.values
                report_table(coef, "Model coefficients", decimals=decimals)
                csv_download(coef, "doe_coefficients.csv")

                xfac = st.selectbox("X-axis factor", factors, index=0)
                yfac = st.selectbox("Y-axis factor", [f for f in factors if f != xfac], index=0)
                other_factors = [f for f in factors if f not in [xfac, yfac]]
                fixed = {}
                if other_factors:
                    st.markdown("**Fixed levels for other factors**")
                    cols = st.columns(len(other_factors))
                    for i, f in enumerate(other_factors):
                        fixed[f] = cols[i].number_input(f, value=float(d[f].mean()))

                x_vals = np.linspace(d[xfac].min(), d[xfac].max(), 40)
                y_vals = np.linspace(d[yfac].min(), d[yfac].max(), 40)
                xx, yy = np.meshgrid(x_vals, y_vals)
                grid = pd.DataFrame({xfac: xx.ravel(), yfac: yy.ravel()})
                for f in other_factors:
                    grid[f] = fixed[f]
                z = model.predict(grid).to_numpy().reshape(xx.shape)

                fig, ax = plt.subplots(figsize=(8.5, 5.5))
                cn = ax.contourf(xx, yy, z, levels=20, cmap="viridis")
                plt.colorbar(cn, ax=ax, label=response)
                ax.scatter(d[xfac], d[yfac], c='white', edgecolor='black', s=35)
                ax.set_xlabel(xfac)
                ax.set_ylabel(yfac)
                ax.set_title(f"Contour plot for {response}")
                st.pyplot(fig)

                fig3d = plt.figure(figsize=(8.5, 6))
                ax3 = fig3d.add_subplot(111, projection='3d')
                surf = ax3.plot_surface(xx, yy, z, cmap='viridis', edgecolor='none', alpha=0.88)
                ax3.scatter(d[xfac], d[yfac], d[response], c='black', s=30)
                ax3.set_xlabel(xfac)
                ax3.set_ylabel(yfac)
                ax3.set_zlabel(response)
                ax3.set_title(f"Response surface for {response}")
                fig3d.colorbar(surf, ax=ax3, shrink=0.65, aspect=12)
                st.pyplot(fig3d)
        except Exception as e:
            st.error(str(e))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import re
from scipy.stats import t

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Labomed Stats Suite", page_icon="🔬", layout="wide")

# ==========================================
# SHARED HELPERS (Used by all Apps)
# ==========================================

def display_formal_table(df, caption, decimals=3, custom_headers=None):
    display_df = df.copy()
    formal_headers = {
        "x": "Time / X", "y": "Actual Y", "fit": "Fitted Y", 
        "ci_lower": "Lower CI", "ci_upper": "Upper CI", 
        "pi_lower": "Lower PI", "pi_upper": "Upper PI",
        "n": "N", "mean": "Mean", "sd": "Std. Deviation",
        "cv_pct": "CV (%)", "median": "Median", "min": "Min", "max": "Max"
    }
    if custom_headers:
        display_df.columns = [custom_headers.get(col, formal_headers.get(col, str(col).replace("_", " ").title())) for col in display_df.columns]
    else:
        display_df.columns = [formal_headers.get(col, str(col).replace("_", " ").title()) for col in display_df.columns]

    styles = [
        dict(selector="caption", props=[("text-align", "left"), ("font-size", "16px"), ("font-weight", "bold"), ("color", "black"), ("padding-bottom", "10px")]),
        dict(selector="thead th", props=[("border-top", "2px solid black"), ("border-bottom", "1px solid black"), ("background-color", "white"), ("color", "black"), ("font-weight", "bold"), ("text-align", "center"), ("padding", "8px 12px")]),
        dict(selector="tbody td", props=[("background-color", "white"), ("border", "none"), ("text-align", "center"), ("padding", "8px 12px")]),
        dict(selector="tbody tr", props=[("background-color", "white")]),
        dict(selector="tbody tr:last-child td", props=[("border-bottom", "2px solid black")])
    ]
    styled_html = (display_df.style.hide(axis="index").set_caption(caption).set_table_styles(styles).format(precision=decimals, na_rep="-").to_html())
    st.markdown(styled_html, unsafe_allow_html=True)

@st.cache_data
def parse_pasted_data(text):
    if not text.strip(): return None
    parsers = [
        lambda s: pd.read_csv(StringIO(s), sep="\t", engine="python"),
        lambda s: pd.read_csv(StringIO(s), sep=",", engine="python"),
        lambda s: pd.read_csv(StringIO(s), sep=";", engine="python"),
        lambda s: pd.read_csv(StringIO(s), sep=r"\s+", engine="python"),
    ]
    for parser in parsers:
        try:
            df = parser(text)
            if df.shape[1] >= 1:
                df.columns = [str(c).strip() for c in df.columns]
                for col in df.columns:
                    if df[col].dtype == object: df[col] = df[col].astype(str).str.strip()
                return df.dropna(how="all").reset_index(drop=True)
        except: continue
    return None

def fit_linear_model(x, y):
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - (X @ beta)
    s = np.sqrt(np.sum(resid**2) / (n - 2))
    r2 = 1 - (np.sum(resid**2) / np.sum((y - np.mean(y))**2))
    return {"intercept": beta[0], "slope": beta[1], "XtX_inv": XtX_inv, "s": s, "df": n-2, "r2": r2}

def predict_intervals(model, x_vals, conf=0.95, side="upper"):
    Xg = np.column_stack([np.ones(len(x_vals)), x_vals])
    yhat = Xg @ np.array([model["intercept"], model["slope"]])
    h = np.einsum("ij,jk,ik->i", Xg, model["XtX_inv"], Xg)
    if side == "two-sided":
        tcrit = t.ppf(1 - (1-conf)/2, model["df"])
    else:
        tcrit = t.ppf(conf, model["df"])
    ci_err = tcrit * model["s"] * np.sqrt(h)
    pi_err = tcrit * model["s"] * np.sqrt(1 + h)
    return pd.DataFrame({"x": x_vals, "fit": yhat, "ci_lower": yhat-ci_err, "ci_upper": yhat+ci_err, "pi_lower": yhat-pi_err, "pi_upper": yhat+pi_err})

# --- MASTER NAVIGATION SIDEBAR ---
st.sidebar.title("🔬 Labomed Stats")
app_selection = st.sidebar.radio("Navigation", [
    "01 - Descriptive Statistics",
    "02 - Linear Regression Intervals",
    "03 - Shelf Life Estimator",
    "04 - Dissolution Comparison (f2)",
    "05 - Two-Sample Tests",
    "06 - Two-Way ANOVA",
    "07 - Tolerance & Confidence Intervals",
    "08 - PCA Analysis",
    "09 - Design of Experiments (DOE)"
])

# ==========================================
# APP 01: DESCRIPTIVE STATISTICS
# ==========================================
if app_selection == "01 - Descriptive Statistics":
    st.title("📊 App 01 - Descriptive Statistics")
    data_input = st.text_area("Data (Paste with headers from Excel)", height=200)
    if data_input:
        df = parse_pasted_data(data_input)
        if df is not None:
            numeric_cols = [c for c in df.columns if pd.to_numeric(df[c].astype(str).str.replace("%",""), errors='coerce').notna().mean() >= 0.7]
            col1, col2, col3 = st.columns(3)
            with col1: selected_vars = st.multiselect("Variables", numeric_cols, default=numeric_cols)
            with col2: group1 = st.selectbox("Group 1", ["(None)"] + list(df.columns))
            with col3: group2 = st.selectbox("Group 2", ["(None)"] + list(df.columns))
            decimals = st.slider("Decimals", 1, 8, 3)
            if st.button("Run Stats"):
                active_groups = [g for g in [group1, group2] if g != "(None)"]
                def calc_stats(x):
                    return pd.Series({'n': x.count(), 'mean': x.mean(), 'sd': x.std(ddof=1), 'min': x.min(), 'median': x.median(), 'max': x.max(), 'cv_pct': (x.std(ddof=1)/x.mean()*100) if x.mean()!=0 else np.nan})
                if active_groups:
                    res = df.groupby(active_groups)[selected_vars].apply(lambda g: g.apply(calc_stats)).unstack(level=-1).reset_index()
                else:
                    res = df[selected_vars].apply(calc_stats).T.reset_index().rename(columns={'index': 'Variable'})
                display_formal_table(res, "Summary Statistics", decimals)

# ==========================================
# APP 02: LINEAR REGRESSION INTERVALS
# ==========================================
elif app_selection == "02 - Linear Regression Intervals":
    st.title("📈 App 02 - Linear Regression Intervals")
    c1, c2 = st.columns([2, 1])
    with c1: data_in = st.text_area("Regression Data", "3\t0.18\n6\t0.38\n9\t0.48\n12\t0.70\n18\t0.95")
    with c2: pred_in = st.text_area("Predict X", "24\n36\n48")
    col1, col2, col3 = st.columns(3)
    with col1: side = st.selectbox("Side", ["upper", "lower", "two-sided"])
    with col2: interval_type = st.selectbox("Type", ["pi", "ci", "both"])
    with col3: conf = st.slider("Confidence", 0.80, 0.99, 0.95)
    decimals = st.slider("Decimals", 1, 8, 4)

    if st.button("Run Regression"):
        df_reg = parse_pasted_data(data_in)
        df_reg.columns = ['x', 'y']
        model = fit_linear_model(df_reg['x'], df_reg['y'])
        grid_x = np.linspace(df_reg['x'].min(), df_reg['x'].max()*1.5, 100)
        grid_res = predict_intervals(model, grid_x, conf, side)
        
        sns.set_theme(style="white")
        fig, ax = plt.subplots()
        ax.plot(df_reg['x'], df_reg['y'], 'ko', label="Data")
        ax.plot(grid_x, grid_res['fit'], 'k-', label="Fit")
        if interval_type in ['ci', 'both']: ax.fill_between(grid_x, grid_res['ci_lower'], grid_res['ci_upper'], alpha=0.2, label="CI")
        if interval_type in ['pi', 'both']: ax.plot(grid_x, grid_res['pi_upper'], 'r--', label="PI")
        ax.legend()
        st.pyplot(fig)
        
        pred_pts = [float(p) for p in pred_in.split() if p.strip()]
        all_x = sorted(list(set(df_reg['x'].tolist() + pred_pts)))
        final_df = predict_intervals(model, np.array(all_x), conf, side)
        final_df = pd.merge(pd.DataFrame({"x": all_x}), df_reg, on="x", how="left").merge(final_df, on="x")
        display_formal_table(final_df, "Table 1: Regression Analysis", decimals)

# ==========================================
# APP 03: SHELF LIFE ESTIMATOR
# ==========================================
elif app_selection == "03 - Shelf Life Estimator":
    st.title("⏳ App 03 - Shelf Life Estimator")
    c1, c2 = st.columns([2, 1])
    with c1: data_in = st.text_area("Stability Data", "0\t100\n3\t99.2\n6\t98.4\n9\t97.8\n12\t97.0\n18\t95.6\n24\t94.8")
    with c2: pred_in = st.text_area("Future Points", "30\n36\n48")
    col1, col2, col3 = st.columns(3)
    with col1: side = st.selectbox("Spec Side", ["lower", "upper"])
    with col2: basis = st.selectbox("Basis", ["ci", "pi", "fit"])
    with col3: limit = st.number_input("Spec Limit", value=90.0)
    conf = st.slider("Confidence (1-sided)", 0.80, 0.99, 0.95)
    decimals = st.slider("Decimals", 1, 8, 4)

    if st.button("Calculate"):
        df_sl = parse_pasted_data(data_in)
        df_sl.columns = ['x', 'y']
        model = fit_linear_model(df_sl['x'], df_sl['y'])
        grid_x = np.linspace(0, max(df_sl['x'].max()*2, 48), 1000)
        res = predict_intervals(model, grid_x, conf, "upper" if side=="upper" else "lower")
        
        bound_col = "fit" if basis == "fit" else ("ci_lower" if side == "lower" else "ci_upper") if basis == "ci" else ("pi_lower" if side == "lower" else "pi_upper")
        
        # Simple crossing logic
        def find_cross(xv, yv, lim):
            for i in range(len(yv)-1):
                if (yv[i]-lim)*(yv[i+1]-lim) <= 0:
                    return xv[i] + (lim - yv[i]) * (xv[i+1]-xv[i])/(yv[i+1]-yv[i])
            return None
        
        sl = find_cross(grid_x, res[bound_col].values, limit)
        
        fig, ax = plt.subplots()
        ax.plot(df_sl['x'], df_sl['y'], 'ko')
        ax.plot(grid_x, res['fit'], 'k-')
        ax.plot(grid_x, res[bound_col], 'r-', label="Shelf Life Bound")
        ax.axhline(limit, color='g', ls='--')
        if sl: ax.axvline(sl, color='g', ls=':')
        st.pyplot(fig)
        
        st.success(f"Estimated Shelf Life: {f'{sl:.2f}' if sl else 'Not reached'}")
        
        pts = [float(p) for p in pred_in.split() if p.strip()]
        all_x = sorted(list(set(df_sl['x'].tolist() + pts)))
        table_res = predict_intervals(model, np.array(all_x), conf, "two-sided")
        final_df = pd.merge(pd.DataFrame({"x": all_x}), df_sl, on="x", how="left").merge(table_res, on="x")
        display_formal_table(final_df, "Table 1: Stability Details", decimals)

# ==========================================
# PLACEHOLDERS
# ==========================================
else:
    st.title(app_selection)
    st.info("Module logic pending...")

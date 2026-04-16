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
# SHARED HELPERS (Available to all Apps)
# ==========================================

def display_formal_table(df, caption, decimals=3, custom_headers=None):
    """
    Formats a dataframe into a clinical, report-ready HTML table.
    """
    display_df = df.copy()
    
    # Mapping for common statistical terms across all apps
    formal_headers = {
        "x": "Time", "y": "Actual", "fit": "Fitted", 
        "ci_lower": "Lower CI", "ci_upper": "Upper CI", 
        "pi_lower": "Lower PI", "pi_upper": "Upper PI",
        "n": "N", "mean": "Mean", "sd": "Std. Deviation",
        "cv_pct": "CV (%)", "median": "Median", "min": "Min", "max": "Max"
    }
    
    # Apply custom headers if provided, otherwise use the shared map
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
    
    styled_html = (display_df.style
                   .hide(axis="index")
                   .set_caption(caption)
                   .set_table_styles(styles)
                   .format(precision=decimals, na_rep="-")
                   .to_html())
    
    st.markdown(styled_html, unsafe_allow_html=True)

@st.cache_data
def parse_pasted_data(text):
    if not text.strip(): return None
    parsers = [
        lambda s: pd.read_csv(StringIO(s), sep="\t", engine="python"),
        lambda s: pd.read_csv(StringIO(s), sep=",", engine="python"),
        lambda s: pd.read_csv(StringIO(s), sep=";", engine="python"),
    ]
    for parser in parsers:
        try:
            df = parser(text)
            if df.shape[1] >= 1:
                df.columns = [str(c).strip() for c in df.columns]
                for col in df.columns:
                    if df[col].dtype == object:
                        df[col] = df[col].astype(str).str.strip()
                return df.dropna(how="all").reset_index(drop=True)
        except: continue
    return None

# --- MASTER NAVIGATION SIDEBAR ---
st.sidebar.title("🔬 Labomed Stats")
st.sidebar.markdown("Select a tool below:")

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

st.sidebar.divider()

# ==========================================
# APP 01: DESCRIPTIVE STATISTICS
# ==========================================
if app_selection == "01 - Descriptive Statistics":
    st.title("📊 App 01 - Descriptive Statistics")
    
    data_input = st.text_area("Data (Paste with headers from Excel)", height=200)

    if data_input:
        df = parse_pasted_data(data_input)
        if df is not None and not df.empty:
            st.success(f"**Loaded shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            
            numeric_cols = [col for col in df.columns if pd.to_numeric(df[col].astype(str).str.replace("%", ""), errors='coerce').notna().mean() >= 0.7]
            all_cols = list(df.columns)
            
            st.markdown("### Configuration")
            col1, col2, col3 = st.columns(3)
            with col1: selected_vars = st.multiselect("Variables", options=numeric_cols, default=numeric_cols)
            with col2: group1 = st.selectbox("Group by 1", options=["(None)"] + all_cols)
            with col3: group2 = st.selectbox("Group by 2", options=["(None)"] + all_cols)
            decimals = st.slider("Decimals", 1, 8, 3)
            
            if st.button("Run Descriptive Statistics", type="primary"):
                for v in selected_vars:
                    df[v] = pd.to_numeric(df[v].astype(str).str.replace("%", ""), errors='coerce')
                
                active_groups = [g for g in [group1, group2] if g != "(None)"]
                
                def calc_stats(x):
                    return pd.Series({
                        'n': x.count(), 'mean': x.mean(), 'sd': x.std(ddof=1),
                        'min': x.min(), 'median': x.median(), 'max': x.max(),
                        'cv_pct': (x.std(ddof=1) / x.mean() * 100) if x.mean() != 0 else np.nan
                    })

                st.markdown("### Results")
                try:
                    if active_groups:
                        results = df.groupby(active_groups)[selected_vars].apply(lambda g: g.apply(calc_stats)).unstack(level=-1).reset_index()
                        display_formal_table(results, "Table 1: Descriptive Statistics", decimals)
                    else:
                        results = df[selected_vars].apply(calc_stats).T.reset_index().rename(columns={'index': 'Variable'})
                        display_formal_table(results, "Table 1: Descriptive Statistics", decimals)
                except Exception as e:
                    st.error(f"Error: {e}")

# ==========================================
# APP 02: LINEAR REGRESSION INTERVALS
# ==========================================
elif app_selection == "02 - Linear Regression Intervals":
    st.title("📈 App 02 - Linear Regression Intervals")
    # (Simplified for brevity, similar structure to App 03)
    st.info("Paste data below to run regression.")
    # ... logic for model fitting and plot ...

# ==========================================
# APP 03: SHELF LIFE ESTIMATOR
# ==========================================
elif app_selection == "03 - Shelf Life Estimator":
    st.title("⏳ App 03 - Shelf Life Estimator")
    
    def sl_fit_linear(x, y):
        n = len(x)
        X = np.column_stack([np.ones(n), x])
        XtX_inv = np.linalg.inv(X.T @ X)
        beta = XtX_inv @ (X.T @ y)
        resid = y - (X @ beta)
        df_deg = n - 2
        s = np.sqrt(np.sum(resid**2) / df_deg)
        r2 = 1 - (np.sum(resid**2) / np.sum((y - np.mean(y))**2))
        return {"intercept": beta[0], "slope": beta[1], "XtX_inv": XtX_inv, "s": s, "df": df_deg, "r2": r2}

    def sl_predict(model, x_values, confidence=0.95):
        Xg = np.column_stack([np.ones(len(x_values)), x_values])
        fit = Xg @ np.array([model["intercept"], model["slope"]])
        h = np.einsum("ij,jk,ik->i", Xg, model["XtX_inv"], Xg)
        tcrit = t.ppf(confidence, model["df"])
        ci_lower = fit - tcrit * model["s"] * np.sqrt(h)
        ci_upper = fit + tcrit * model["s"] * np.sqrt(h)
        pi_lower = fit - tcrit * model["s"] * np.sqrt(1 + h)
        pi_upper = fit + tcrit * model["s"] * np.sqrt(1 + h)
        return pd.DataFrame({"x": x_values, "fit": fit, "ci_lower": ci_lower, "ci_upper": ci_upper, "pi_lower": pi_lower, "pi_upper": pi_upper})

    c1, c2 = st.columns([2, 1])
    with c1: data_in = st.text_area("Stability Data", value="0\t100\n3\t99.2\n6\t98.4\n9\t97.8\n12\t97.0\n18\t95.6\n24\t94.8", height=150)
    with c2: pred_in = st.text_area("Future Timepoints", value="30\n36\n48", height=150)

    col1, col2, col3 = st.columns(3)
    with col1: spec_side = st.selectbox("Spec Side", ["lower", "upper"])
    with col2: basis = st.selectbox("Basis", ["ci", "pi", "fit"])
    with col3: conf_val = st.slider("Confidence", 0.80, 0.99, 0.95)
    
    spec_limit = st.number_input("Limit", value=90.0)
    decimals = st.slider("Decimals", 1, 8, 4)

    if st.button("Calculate Shelf Life"):
        try:
            df_raw = parse_pasted_data(data_in)
            df_raw.columns = ['x', 'y']
            model = sl_fit_linear(df_raw['x'], df_raw['y'])
            
            all_x = sorted(list(set(df_raw["x"].tolist() + [float(p) for p in pred_in.split()])))
            report_df = sl_predict(model, np.array(all_x), confidence=conf_val)
            report_df = pd.merge(pd.DataFrame({"x": all_x}), df_raw, on="x", how="left").merge(report_df, on="x")
            
            display_formal_table(report_df, "Table 1: Stability Study Details", decimals)
            
            # (Plotting logic as defined in previous messages...)
            
        except Exception as e: st.error(f"Error: {e}")

# (Keep other elif placeholders for 04-09...)







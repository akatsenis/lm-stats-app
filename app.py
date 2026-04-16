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

st.sidebar.divider()
st.sidebar.info("Upload or paste data directly from Excel into the tools.")

# ==========================================
# APP 01: DESCRIPTIVE STATISTICS
# ==========================================
    
if app_selection == "01 - Descriptive Statistics":
    
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

    def get_numeric_columns(df):
        num_cols = []
        for col in df.columns:
            converted = pd.to_numeric(df[col].astype(str).str.replace("%", ""), errors='coerce')
            if converted.notna().mean() >= 0.7: num_cols.append(col)
        return num_cols

    st.title("📊 App 01 - Descriptive Statistics")
    st.markdown("Paste data from Excel, optionally select grouping columns, and get summary statistics.")
    
    data_input = st.text_area("Data (Paste with headers from Excel)", height=200)

    if data_input:
        df = parse_pasted_data(data_input)
        if df is not None and not df.empty:
            st.success(f"**Loaded shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            with st.expander("Preview Loaded Data"):
                st.dataframe(df.head(10), use_container_width=True)
                
            numeric_cols = get_numeric_columns(df)
            all_cols = list(df.columns)
            
            st.markdown("### Configuration")
            col1, col2, col3 = st.columns(3)
            with col1: selected_vars = st.multiselect("Variables", options=numeric_cols, default=numeric_cols)
            with col2: group1 = st.selectbox("Group by 1", options=["(None)"] + all_cols)
            with col3: group2 = st.selectbox("Group by 2", options=["(None)"] + all_cols)
                
            decimals = st.slider("Decimals", min_value=1, max_value=8, value=3)
            
            if st.button("Run Descriptive Statistics", type="primary"):
                if not selected_vars:
                    st.error("Please select at least one numeric variable.")
                else:
                    for v in selected_vars:
                        df[v] = pd.to_numeric(df[v].astype(str).str.replace("%", ""), errors='coerce')
                    
                    active_groups = [g for g in [group1, group2] if g != "(None)"]
                    
                    def calc_stats(x):
                        return pd.Series({
                            'N': x.count(), 'Mean': x.mean(), 'Std. Dev': x.std(ddof=1),
                            'Min': x.min(), 'Median': x.median(), 'Max': x.max(),
                            'CV (%)': (x.std(ddof=1) / x.mean() * 100) if x.mean() != 0 else np.nan
                        })

                    st.markdown("### Results")
                    try:
                        if active_groups:
                            results = df.groupby(active_groups)[selected_vars].apply(lambda g: g.apply(calc_stats)).unstack(level=-1)
                            if len(selected_vars) == 1: results.columns = results.columns.droplevel(0)
                        else:
                            results = df[selected_vars].apply(calc_stats).T
                            
                        st.dataframe(results.style.format(precision=decimals, na_rep="-"), use_container_width=True)
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.error("Could not parse data.")

# ==========================================
# APP 02: LINEAR REGRESSION INTERVALS
# ==========================================
elif app_selection == "02 - Linear Regression Intervals":
    st.title("📈 App 02 - Linear Regression Intervals")
    st.markdown("Paste Excel data to generate models, CI/PI bands, and predict values.")

    # --- Helper Functions ---
    def display_formal_table(df, caption, decimals=3):
        display_df = df.copy()
        formal_headers = {
            "x": "X Value", "y": "Actual Y", "fit": "Fitted Y", 
            "ci_lower": "Lower CI", "ci_upper": "Upper CI", 
            "pi_lower": "Lower PI", "pi_upper": "Upper PI"
        }
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

    def _to_numeric_clean(series):
        return pd.to_numeric(series.astype(str).str.strip().str.replace("%", "", regex=False), errors="coerce")

    def parse_xy_data(text):
        text = str(text).strip()
        if not text: raise ValueError("Paste X and Y data into the data box.")
        parsers = [
            lambda s: pd.read_csv(StringIO(s), sep="\t", header=None, engine="python"),
            lambda s: pd.read_csv(StringIO(s), sep=",", header=None, engine="python"),
            lambda s: pd.read_csv(StringIO(s), sep=";", header=None, engine="python"),
            lambda s: pd.read_csv(StringIO(s), sep=r"\s+", header=None, engine="python"),
        ]
        df = None
        for parser in parsers:
            try:
                trial = parser(text)
                if trial.shape[1] >= 2:
                    df = trial.copy()
                    break
            except: pass
        if df is None or df.shape[1] < 2: raise ValueError("Could not read two columns. Paste two Excel columns: X and Y.")
        df = df.iloc[:, :2].copy()
        df.columns = ["x", "y"]
        first_row_numeric = _to_numeric_clean(df.iloc[0])
        if first_row_numeric.isna().any(): df = df.iloc[1:].reset_index(drop=True)
        df["x"] = _to_numeric_clean(df["x"])
        df["y"] = _to_numeric_clean(df["y"])
        df = df.dropna().sort_values("x").reset_index(drop=True)
        if len(df) < 3: raise ValueError("At least 3 valid rows are required.")
        if df["x"].nunique() < 2: raise ValueError("X values must not all be the same.")
        return df

    def parse_prediction_points(text):
        text = str(text).strip()
        if not text: return np.array([], dtype=float)
        parts = re.split(r"[\s,\t;]+", text)
        vals = [float(p.strip()) for p in parts if p.strip()]
        return np.array(vals, dtype=float)

    def fit_linear_model(x, y):
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        n = len(x)
        X = np.column_stack([np.ones(n), x])
        XtX_inv = np.linalg.inv(X.T @ X)
        beta = XtX_inv @ (X.T @ y)
        intercept, slope = beta
        y_fit = X @ beta
        resid = y - y_fit
        df = n - 2
        s = np.sqrt(np.sum(resid**2) / df)
        ss_tot = np.sum((y - np.mean(y))**2)
        ss_res = np.sum(resid**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return {"intercept": intercept, "slope": slope, "XtX_inv": XtX_inv, "s": s, "df": df, "r2": r2, "x": x, "y": y, "y_fit": y_fit}

    def predict_with_intervals(model, x_values, confidence=0.95, side="upper"):
        x_values = np.asarray(x_values, dtype=float).ravel()
        Xg = np.column_stack([np.ones(len(x_values)), x_values])
        beta = np.array([model["intercept"], model["slope"]])
        yhat = Xg @ beta
        h = np.einsum("ij,jk,ik->i", Xg, model["XtX_inv"], Xg)
        se_mean = model["s"] * np.sqrt(h)
        se_pred = model["s"] * np.sqrt(1 + h)
        alpha = 1 - confidence
        tcrit = t.ppf(1 - alpha / 2, model["df"]) if side == "two-sided" else t.ppf(confidence, model["df"])
        ci_lower = yhat - tcrit * se_mean
        ci_upper = yhat + tcrit * se_mean
        pi_lower = yhat - tcrit * se_pred
        pi_upper = yhat + tcrit * se_pred
        return pd.DataFrame({"x": x_values, "fit": yhat, "ci_lower": ci_lower, "ci_upper": ci_upper, "pi_lower": pi_lower, "pi_upper": pi_upper})

    def _find_crossing(xv, yv, limit):
        d = yv - limit
        idx = np.where(d[:-1] * d[1:] <= 0)[0]
        if len(idx) == 0: return None
        i = idx[0]
        x1, x2, y1, y2 = xv[i], xv[i + 1], yv[i], yv[i + 1]
        return x1 if y2 == y1 else x1 + (limit - y1) * (x2 - x1) / (y2 - y1)

    def plot_regression(data_df, model, grid_df, confidence, interval, side, title, xlabel, ylabel, point_label, y_suffix, spec_enabled, spec_limit, spec_label, crossing_on):
        sns.set_theme(style="white", palette="muted")
        plt.rcParams.update({"font.family": "sans-serif", "axes.titlesize": 14, "axes.titleweight": "bold", "axes.labelsize": 12, "figure.dpi": 300})
        x = data_df["x"].to_numpy()
        y = data_df["y"].to_numpy()
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.scatter(x, y, color="#2c3e50", s=50, alpha=0.8, label=point_label, zorder=3)
        ax.plot(grid_df["x"], grid_df["fit"], color="#2c3e50", lw=2, label="Fitted Line")

        if interval in ["ci", "both"]:
            if side == "two-sided":
                ax.fill_between(grid_df["x"], grid_df["ci_lower"], grid_df["ci_upper"], color="#3498db", alpha=0.2, label="Confidence Interval (CI)")
                ax.plot(grid_df["x"], grid_df["ci_upper"], color="#3498db", ls="--", lw=1.2)
                ax.plot(grid_df["x"], grid_df["ci_lower"], color="#3498db", ls="--", lw=1.2)
            elif side == "upper":
                ax.fill_between(grid_df["x"], grid_df["fit"], grid_df["ci_upper"], color="#3498db", alpha=0.2, label="Upper CI")
                ax.plot(grid_df["x"], grid_df["ci_upper"], color="#3498db", ls="--", lw=1.4, label="_nolegend_")
            elif side == "lower":
                ax.fill_between(grid_df["x"], grid_df["ci_lower"], grid_df["fit"], color="#3498db", alpha=0.2, label="Lower CI")
                ax.plot(grid_df["x"], grid_df["ci_lower"], color="#3498db", ls="--", lw=1.4, label="_nolegend_")

        if interval in ["pi", "both"]:
            if side == "two-sided":
                ax.fill_between(grid_df["x"], grid_df["pi_lower"], grid_df["pi_upper"], color="#e74c3c", alpha=0.15, label="Prediction Interval (PI)")
                ax.plot(grid_df["x"], grid_df["pi_upper"], color="#e74c3c", ls=(0, (4, 4)), lw=1.2)
                ax.plot(grid_df["x"], grid_df["pi_lower"], color="#e74c3c", ls=(0, (4, 4)), lw=1.2)
            elif side == "upper":
                ax.fill_between(grid_df["x"], grid_df["fit"], grid_df["pi_upper"], color="#e74c3c", alpha=0.15, label="Upper PI")
                ax.plot(grid_df["x"], grid_df["pi_upper"], color="#e74c3c", ls=(0, (4, 4)), lw=1.4, label="_nolegend_")
            elif side == "lower":
                ax.fill_between(grid_df["x"], grid_df["pi_lower"], grid_df["fit"], color="#e74c3c", alpha=0.15, label="Lower PI")
                ax.plot(grid_df["x"], grid_df["pi_lower"], color="#e74c3c", ls=(0, (4, 4)), lw=1.4, label="_nolegend_")

        crossing_x = None
        if spec_enabled and spec_limit is not None:
            ax.axhline(spec_limit, color="#27ae60", ls="--", lw=1.5, label=f"Limit ({spec_label})")
            curve_map = {"fit": grid_df["fit"].to_numpy(), "ci_upper": grid_df["ci_upper"].to_numpy(), "ci_lower": grid_df["ci_lower"].to_numpy(), "pi_upper": grid_df["pi_upper"].to_numpy(), "pi_lower": grid_df["pi_lower"].to_numpy()}
            if crossing_on == "auto":
                crossing_on = "pi_upper" if side == "upper" else "pi_lower" if side == "lower" else "pi_upper" if interval in ["both", "pi"] else "ci_upper" if side == "upper" else "ci_lower" if side == "lower" else "ci_upper"
            if crossing_on in curve_map:
                crossing_x = _find_crossing(grid_df["x"].to_numpy(), curve_map[crossing_on], spec_limit)
                if crossing_x is not None:
                    ax.axvline(crossing_x, color="#27ae60", ls=":", lw=1.5)
                    ymin, ymax = ax.get_ylim()
                    ax.text(crossing_x, ymin + 0.05 * (ymax - ymin), f" {crossing_x:.2f}", color="#27ae60", ha="left", va="bottom", fontsize=11, weight="bold", bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

            xmin, xmax = grid_df["x"].min(), grid_df["x"].max()
            ymax_data = max(grid_df["fit"].max(), grid_df["ci_upper"].max(), grid_df["pi_upper"].max(), y.max())
            ymin_data = min(grid_df["fit"].min(), grid_df["ci_lower"].min(), grid_df["pi_lower"].min(), y.min())
            pad = 0.02 * (ymax_data - ymin_data if ymax_data > ymin_data else 1)
            ax.text(xmin + (xmax - xmin) * 0.02, spec_limit + pad, f"{spec_label} = {spec_limit:.1f}{y_suffix}", ha="left", va="bottom", fontsize=11, color="#27ae60", weight="bold", bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3))

        if y_suffix: ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, pos: f"{v:.1f}{y_suffix}"))
        if not title.strip():
            s1 = {"upper": "Upper One-Sided", "lower": "Lower One-Sided", "two-sided": "Two-Sided"}[side]
            s2 = {"ci": "Confidence Intervals", "pi": "Prediction Intervals", "both": "Confidence and Prediction Intervals"}[interval]
            title = f"{s1} {s2} ({confidence:.0%})"

        ax.set_title(title, pad=15)
        ax.set_xlabel(xlabel, labelpad=10)
        ax.set_ylabel(ylabel, labelpad=10)
        sns.despine()
        ax.legend(frameon=False, loc="best")
        plt.tight_layout()
        st.pyplot(fig) 
        return crossing_x

    def parse_optional_float(txt):
        try: return float(str(txt).strip()) if str(txt).strip() else None
        except: return None

    # --- UI LAYOUT ---
    col1, col2 = st.columns([2, 1])
    with col1:
        data_input = st.text_area("Data (Paste 2 columns: X and Y)", value="3\t0.18\n6\t0.38\n9\t0.48\n12\t0.70\n18\t0.95", height=150)
    with col2:
        pred_input = st.text_area("Predict X (Optional future times)", value="24\n36\n48\n60", height=150)

    st.markdown("### Configuration")
    c1, c2, c3 = st.columns(3)
    with c1: interval_val = st.selectbox("Interval", options=["ci", "pi", "both"], format_func=lambda x: {"ci":"CI", "pi":"PI", "both":"Both"}[x], index=1)
    with c2: side_val = st.selectbox("Side", options=["upper", "lower", "two-sided"], format_func=lambda x: x.title(), index=0)
    with c3: conf_val = st.slider("Confidence", min_value=0.80, max_value=0.99, value=0.95, step=0.01)

    c4, c5, c6, c7 = st.columns(4)
    with c4: title_val = st.text_input("Plot Title", value="")
    with c5: xlab_val = st.text_input("X Label", value="Time")
    with c6: ylab_val = st.text_input("Y Label", value="Response")
    with c7: point_val = st.text_input("Point Label", value="Data")

    c8, c9, c10, c11 = st.columns(4)
    with c8: y_suff_val = st.text_input("Y Suffix", value="%")
    with c9: xmin_val = st.text_input("X Min", value="")
    with c10: xmax_val = st.text_input("X Max", value="40")
    with c11: decimals_val = st.slider("Decimals", 1, 8, 4)

    st.markdown("### Specifications")
    s1, s2, s3, s4 = st.columns(4)
    with s1: spec_en_val = st.checkbox("Use spec limit", value=True)
    with s2: spec_limit_val = st.text_input("Spec Value", value="3.0")
    with s3: spec_label_val = st.text_input("Spec Label", value="US")
    with s4: cross_val = st.selectbox("Crossing on", options=["auto", "fit", "ci_upper", "ci_lower", "pi_upper", "pi_lower"], format_func=lambda x: x.replace("_", " ").title())

    if st.button("Run Regression Analysis", type="primary"):
        try:
            data_df = parse_xy_data(data_input)
            pred_x = parse_prediction_points(pred_input)

            x_all_max = data_df["x"].max()
            if len(pred_x) > 0: x_all_max = max(x_all_max, np.max(pred_x))

            x_min = parse_optional_float(xmin_val)
            x_max = parse_optional_float(xmax_val)
            if x_min is None: x_min = min(0, data_df["x"].min())
            if x_max is None: x_max = x_all_max * 1.15 if x_all_max != 0 else 1
            if x_max <= x_min: st.error("X max must be greater than X min.")
            else:
                grid_x = np.linspace(x_min, x_max, 500)
                model = fit_linear_model(data_df["x"], data_df["y"])
                grid_df = predict_with_intervals(model, grid_x, confidence=conf_val, side=side_val)

                st.markdown("### Regression Plot")
                crossing_x = plot_regression(data_df, model, grid_df, conf_val, interval_val, side_val, title_val, xlab_val, ylab_val, point_val, y_suff_val, spec_en_val, parse_optional_float(spec_limit_val) if spec_en_val else None, spec_label_val, cross_val)

                eq_html = f"""
                <div style="border-left: 4px solid #2980b9; padding: 10px 15px; background-color: #f8f9fa; margin-bottom: 20px;">
                    <h4 style="margin-top: 0;">Regression Model Summary</h4>
                    <b>Equation:</b> y = {model['intercept']:.{decimals_val}f} + {model['slope']:.{decimals_val}f} &times; x<br>
                    <b>R²:</b> {model['r2']:.{decimals_val}f}<br>
                    <b>Residual SD (s):</b> {model['s']:.{decimals_val}f}<br>
                    <b>Degrees of Freedom:</b> {model['df']}
                """
                if crossing_x is not None:
                    eq_html += f"<br><br><b>Crossing Point:</b> x = <span style='color: #27ae60; font-weight: bold;'>{crossing_x:.{decimals_val}f}</span>"
                eq_html += "</div>"
                st.markdown(eq_html, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                display_formal_table(data_df, "Table 1: Parsed Input Data", decimals_val)

                new_pred_x = np.setdiff1d(pred_x, data_df["x"].to_numpy())
                if len(new_pred_x) > 0:
                    new_pts_df = pd.DataFrame({"x": new_pred_x, "y": np.nan})
                    combined_pts_df = pd.concat([data_df[["x", "y"]], new_pts_df], ignore_index=True)
                else:
                    combined_pts_df = data_df[["x", "y"]].copy()

                combined_pts_df = combined_pts_df.sort_values("x").reset_index(drop=True)
                unique_x = combined_pts_df["x"].unique()
                intervals_df = predict_with_intervals(model, unique_x, confidence=conf_val, side=side_val)
                final_table_df = pd.merge(combined_pts_df, intervals_df, on="x", how="left")
                col_order = ['x', 'y', 'fit', 'ci_lower', 'ci_upper', 'pi_lower', 'pi_upper']
                final_table_df = final_table_df[[c for c in col_order if c in final_table_df.columns]]

                st.markdown("<br>", unsafe_allow_html=True)
                display_formal_table(final_table_df, "Table 2: Fitted Values & Intervals", decimals_val)

        except Exception as e:
            st.error(f"Error: {e}")

# ==========================================
# APP 03: SHELF LIFE ESTIMATOR
# ==========================================
elif app_selection == "03 - Shelf Life Estimator":
    st.title("📈 App 03 - Shelf Life Estimator")
    st.info("We will paste the code for this app here next!")

# ==========================================
# APP 04: DISSOLUTION COMPARISON
# ==========================================
elif app_selection == "04 - Dissolution Comparison (f2)":
    st.title("💊 App 04 - Dissolution Comparison (f2)")
    st.info("We will paste the code for this app here next!")

# ==========================================
# APP 05: TWO-SAMPLE TESTS
# ==========================================
elif app_selection == "05 - Two-Sample Tests":
    st.title("⚖️ App 05 - Two-Sample Tests")
    st.info("We will paste the code for this app here next!")

# ==========================================
# APP 06: TWO-WAY ANOVA
# ==========================================
elif app_selection == "06 - Two-Way ANOVA":
    st.title("📐 App 06 - Two-Way ANOVA")
    st.info("We will paste the code for this app here next!")

# ==========================================
# APP 07: TOLERANCE & CONFIDENCE INTERVALS
# ==========================================
elif app_selection == "07 - Tolerance & Confidence Intervals":
    st.title("🎯 App 07 - Tolerance & Confidence Intervals")
    st.info("We will paste the code for this app here next!")

# ==========================================
# APP 08: PCA ANALYSIS
# ==========================================
elif app_selection == "08 - PCA Analysis":
    st.title("🌐 App 08 - PCA Analysis")
    st.info("We will paste the code for this app here next!")

# ==========================================
# APP 09: DESIGN OF EXPERIMENTS (DOE)
# ==========================================
elif app_selection == "09 - Design of Experiments (DOE)":
    st.title("🧪 App 09 - Design of Experiments (DOE)")
    st.markdown("Define factors and responses, generate a standard design matrix, and analyze main effects and interactions.")
    
    default_factors = "Temp, 20, 40\nTime, 10, 30\nCatalyst, 1, 5"
    default_responses = "Yield\nImpurity"
    
    col1, col2 = st.columns(2)
    with col1: factors_input = st.text_area("Factors (Name, Low, High)", value=default_factors, height=120)
    with col2: responses_input = st.text_area("Responses (One per line)", value=default_responses, height=120)
        
    col3, col4, col5 = st.columns(3)
    with col3: design_type = st.selectbox("Design type", options=["Full Factorial (2-level)", "Fractional Factorial", "Plackett-Burman"])
    with col4: center_points = st.number_input("Center points", min_value=0, value=3, step=1)
    with col5: replicates = st.number_input("Replicates", min_value=1, value=1, step=1)
        
    col6, col7, col8 = st.columns(3)
    with col6: plot_title = st.text_input("Plot title", value="Pareto Chart")
    with col7: y_label = st.text_input("Y label", value="Standardized Effect")
    with col8: decimals = st.slider("Decimals", min_value=1, max_value=8, value=4)
        
    btn_col1, btn_col2 = st.columns([1, 1])
    with btn_col1: generate_btn = st.button("Generate Design Matrix", type="secondary", use_container_width=True)
    with btn_col2: analyze_btn = st.button("Analyze Responses", type="primary", use_container_width=True)
        
    st.divider()
    
    if generate_btn: st.info("Design Matrix generation logic will go here once you provide the Python code!")
    if analyze_btn: st.info("Response Analysis and plotting logic will go here once you provide the Python code!")

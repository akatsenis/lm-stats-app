import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="lm Stats Suite", page_icon="🔬", layout="wide")

# --- MASTER NAVIGATION SIDEBAR ---
st.sidebar.title("🔬 lm Stats")
st.sidebar.markdown("Select a tool below:")

app_selection = st.sidebar.radio("Navigation", [
    "01 - Descriptive Statistics",
    "02 - Shelf Life Estimator",
    "03 - Dissolution Comparison (f2)",
    "04 - Two-Sample Tests",
    "05 - Two-Way ANOVA",
    "06 - Tolerance & Confidence Intervals",
    "07 - PCA Analysis"
])

st.sidebar.divider()
st.sidebar.info("Upload or paste data directly from Excel into the tools.")

# ==========================================
# APP 01: DESCRIPTIVE STATISTICS
# ==========================================
if app_selection == "01 - Descriptive Statistics":
    
    # Helper functions specific to App 01
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

    # App 01 UI
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
# APP 02: SHELF LIFE ESTIMATOR
# ==========================================
elif app_selection == "02 - Shelf Life Estimator":
    st.title("📈 App 02 - Shelf Life Estimator")
    st.info("We will paste the code for this app here next!")

# ==========================================
# APP 03: DISSOLUTION COMPARISON
# ==========================================
elif app_selection == "03 - Dissolution Comparison (f2)":
    st.title("💊 App 03 - Dissolution Comparison (f2)")
    st.info("We will paste the code for this app here next!")

# ==========================================
# APP 04: TWO-SAMPLE TESTS
# ==========================================
elif app_selection == "04 - Two-Sample Tests":
    st.title("⚖️ App 04 - Two-Sample Tests")
    st.info("We will paste the code for this app here next!")

# ==========================================
# APP 05: TWO-WAY ANOVA
# ==========================================
elif app_selection == "05 - Two-Way ANOVA":
    st.title("📐 App 05 - Two-Way ANOVA")
    st.info("We will paste the code for this app here next!")

# ==========================================
# APP 06: TOLERANCE & CONFIDENCE INTERVALS
# ==========================================
elif app_selection == "06 - Tolerance & Confidence Intervals":
    st.title("🎯 App 06 - Tolerance & Confidence Intervals")
    st.info("We will paste the code for this app here next!")

# ==========================================
# APP 07: PCA ANALYSIS
# ==========================================
elif app_selection == "07 - PCA Analysis":
    st.title("🌐 App 07 - PCA Analysis")
    st.info("We will paste the code for this app here next!")

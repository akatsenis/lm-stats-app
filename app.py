import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from io import StringIO
from scipy import stats
from scipy.stats import t, norm, gaussian_kde, chi2, nct
from matplotlib.patches import Ellipse

from statsmodels.stats.diagnostic import normal_ad
from statsmodels.stats.oneway import anova_oneway
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

from sklearn.decomposition import PCA


def sl_to_numeric(series):
    return pd.to_numeric(
        series.astype(str).str.strip().str.replace("%", "", regex=False),
        errors="coerce"
    )

def sl_parse_xy_data(text):
    text = str(text).strip()
    if not text:
        raise ValueError("Paste two Excel columns: Time and Response.")

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
                df = trial.iloc[:, :2].copy()
                break
        except:
            pass

    if df is None or df.shape[1] < 2:
        raise ValueError("Could not read two columns. Paste Time and Response from Excel.")

    df.columns = ["x", "y"]

    first_row_num = sl_to_numeric(df.iloc[0])
    if first_row_num.isna().any():
        df = df.iloc[1:].reset_index(drop=True)

    df["x"] = sl_to_numeric(df["x"])
    df["y"] = sl_to_numeric(df["y"])
    df = df.dropna().sort_values("x").reset_index(drop=True)

    if len(df) < 3:
        raise ValueError("At least 3 valid rows are required.")
    if df["x"].nunique() < 2:
        raise ValueError("Time values must not all be the same.")

    return df

def sl_parse_x_values(text):
    text = str(text).strip()
    if not text:
        return np.array([], dtype=float)

    parts = re.split(r"[\s,\t;]+", text)
    vals = []
    for p in parts:
        p = p.strip()
        if p:
            vals.append(float(p))
    return np.array(vals, dtype=float)

def sl_fit_linear(x, y):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    n = len(x)
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)

    intercept, slope = beta
    fitted = X @ beta
    resid = y - fitted
    df = n - 2
    s = np.sqrt(np.sum(resid**2) / df)

    ss_tot = np.sum((y - np.mean(y))**2)
    ss_res = np.sum(resid**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "intercept": intercept,
        "slope": slope,
        "XtX_inv": XtX_inv,
        "fitted": fitted,
        "resid": resid,
        "s": s,
        "df": df,
        "r2": r2
    }

def sl_predict(model, x_values, confidence=0.95, one_sided=True):
    x_values = np.asarray(x_values, dtype=float).ravel()
    Xg = np.column_stack([np.ones(len(x_values)), x_values])

    beta = np.array([model["intercept"], model["slope"]])
    fit = Xg @ beta

    h = np.einsum("ij,jk,ik->i", Xg, model["XtX_inv"], Xg)
    se_mean = model["s"] * np.sqrt(h)
    se_pred = model["s"] * np.sqrt(1 + h)

    alpha = 1 - confidence
    tcrit = t.ppf(confidence, model["df"]) if one_sided else t.ppf(1 - alpha / 2, model["df"])

    ci_lower = fit - tcrit * se_mean
    ci_upper = fit + tcrit * se_mean
    pi_lower = fit - tcrit * se_pred
    pi_upper = fit + tcrit * se_pred

    return pd.DataFrame({
        "x": x_values,
        "fit": fit,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "pi_lower": pi_lower,
        "pi_upper": pi_upper
    })

def sl_find_crossing(xv, yv, limit):
    xv = np.asarray(xv, dtype=float)
    yv = np.asarray(yv, dtype=float)
    d = yv - limit

    if d[0] == 0:
        return xv[0]

    for i in range(len(d) - 1):
        if d[i] == 0:
            return xv[i]
        if d[i] * d[i + 1] < 0:
            x1, x2 = xv[i], xv[i + 1]
            y1, y2 = yv[i], yv[i + 1]
            if y2 == y1:
                return x1
            return x1 + (limit - y1) * (x2 - x1) / (y2 - y1)

    return None

def sl_get_bound_column(spec_side, shelf_basis):
    if shelf_basis == "fit":
        return "fit"
    if shelf_basis == "ci":
        return "ci_lower" if spec_side == "lower" else "ci_upper"
    if shelf_basis == "pi":
        return "pi_lower" if spec_side == "lower" else "pi_upper"
    raise ValueError("Invalid shelf-life basis.")

def sl_plot(
    data_df,
    grid_df,
    spec_side,
    spec_limit,
    shelf_basis,
    show_ci_band,
    show_pi_band,
    title,
    xlabel,
    ylabel,
    point_label,
    y_suffix,
    spec_label
):
    x = data_df["x"].to_numpy()
    y = data_df["y"].to_numpy()

    fig, ax = plt.subplots(figsize=(11, 6.5))

    if show_pi_band:
        ax.fill_between(
            grid_df["x"], grid_df["pi_lower"], grid_df["pi_upper"],
            color="red", alpha=0.08, label="PI band"
        )
        ax.plot(grid_df["x"], grid_df["pi_lower"], color="red", lw=1.0, ls=(0, (4, 4)))
        ax.plot(grid_df["x"], grid_df["pi_upper"], color="red", lw=1.0, ls=(0, (4, 4)))

    if show_ci_band:
        ax.fill_between(
            grid_df["x"], grid_df["ci_lower"], grid_df["ci_upper"],
            color="royalblue", alpha=0.12, label="CI band"
        )
        ax.plot(grid_df["x"], grid_df["ci_lower"], color="royalblue", lw=1.0, ls="--")
        ax.plot(grid_df["x"], grid_df["ci_upper"], color="royalblue", lw=1.0, ls="--")

    ax.scatter(x, y, color="black", s=40, label=point_label, zorder=3)
    ax.plot(grid_df["x"], grid_df["fit"], color="black", lw=1.6, label="Fitted line")

    bound_col = sl_get_bound_column(spec_side, shelf_basis)
    bound_color = {"fit": "black", "ci": "royalblue", "pi": "red"}[shelf_basis]
    bound_label = {
        "fit": "Shelf-life line (fit)",
        "ci": f"Shelf-life bound ({'lower' if spec_side == 'lower' else 'upper'} CI)",
        "pi": f"Shelf-life bound ({'lower' if spec_side == 'lower' else 'upper'} PI)"
    }[shelf_basis]

    if shelf_basis != "fit":
        ax.plot(grid_df["x"], grid_df[bound_col], color=bound_color, lw=2.2, label=bound_label)

    ax.axhline(spec_limit, color="black", ls="--", lw=1.2, label=spec_label)

    shelf_life = sl_find_crossing(grid_df["x"].to_numpy(), grid_df[bound_col].to_numpy(), spec_limit)

    if shelf_life is not None:
        ax.axvline(shelf_life, color="black", ls="--", lw=1.2)

    ymax_data = max(
        np.max(y),
        np.max(grid_df["fit"]),
        np.max(grid_df["ci_upper"]),
        np.max(grid_df["pi_upper"])
    )
    ymin_data = min(
        np.min(y),
        np.min(grid_df["fit"]),
        np.min(grid_df["ci_lower"]),
        np.min(grid_df["pi_lower"])
    )
    pad = 0.03 * (ymax_data - ymin_data if ymax_data > ymin_data else 1)

    ax.text(
        grid_df["x"].max() * 0.98,
        spec_limit + pad,
        f"{spec_label} = {spec_limit:.3f}{y_suffix}",
        ha="right",
        va="bottom",
        fontsize=11
    )

    if shelf_life is not None:
        ax.text(
            shelf_life,
            ymin_data + pad,
            f"{shelf_life:.2f}",
            ha="right",
            va="bottom",
            fontsize=11
        )

    if y_suffix:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, pos: f"{v:.1f}{y_suffix}"))

    if not str(title).strip():
        side_txt = "Lower Spec" if spec_side == "lower" else "Upper Spec"
        basis_txt = {"fit": "Fit", "ci": "Confidence Bound", "pi": "Prediction Bound"}[shelf_basis]
        title = f"Shelf Life Estimator ({side_txt}, {basis_txt})"

    ax.set_title(title, fontsize=16, weight="bold")
    ax.set_xlabel(xlabel, fontsize=12, weight="bold")
    ax.set_ylabel(ylabel, fontsize=12, weight="bold")
    ax.legend(frameon=False, loc="best")
    ax.grid(False)
    plt.tight_layout()
    plt.show()

    return shelf_life, bound_col

def dis_make_unique(names):
    out = []
    seen = {}
    for i, n in enumerate(names):
        n = str(n).strip()
        if n == "" or n.lower() == "nan":
            n = f"Col{i+1}"
        if n in seen:
            seen[n] += 1
            n = f"{n}_{seen[n]}"
        else:
            seen[n] = 1
        out.append(n)
    return out

def dis_to_numeric(series):
    return pd.to_numeric(
        series.astype(str).str.strip().str.replace("%", "", regex=False),
        errors="coerce"
    )

def dis_parse_profile_table(text):
    text = str(text).strip()
    if not text:
        raise ValueError("Paste a dissolution table.")

    parsers = [
        lambda s: pd.read_csv(StringIO(s), sep="\t", header=None, engine="python"),
        lambda s: pd.read_csv(StringIO(s), sep=",", header=None, engine="python"),
        lambda s: pd.read_csv(StringIO(s), sep=";", header=None, engine="python"),
    ]

    df_raw = None
    for parser in parsers:
        try:
            trial = parser(text)
            if trial.shape[1] >= 2:
                df_raw = trial.copy()
                break
        except:
            pass

    if df_raw is None or df_raw.shape[1] < 2:
        raise ValueError("Could not read the pasted table. Use at least 2 columns: Time and one or more units.")

    df_raw = df_raw.dropna(how="all").reset_index(drop=True)

    first_row = df_raw.iloc[0].astype(str).str.strip()
    first_row_numeric = pd.to_numeric(first_row, errors="coerce").notna().all()

    if first_row_numeric:
        df = df_raw.copy()
        df.columns = ["Time"] + [f"Unit{i}" for i in range(1, df.shape[1])]
    else:
        df = df_raw.iloc[1:].reset_index(drop=True).copy()
        header = dis_make_unique(first_row.tolist())
        df.columns = header
        df.rename(columns={df.columns[0]: "Time"}, inplace=True)

    df.columns = dis_make_unique(df.columns)

    df["Time"] = dis_to_numeric(df["Time"])

    unit_cols = [c for c in df.columns if c != "Time"]
    for c in unit_cols:
        df[c] = dis_to_numeric(df[c])

    df = df.dropna(subset=["Time"]).copy()
    df = df.loc[df[unit_cols].notna().any(axis=1)].copy()
    df = df.sort_values("Time").reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("No valid dissolution rows found after cleaning.")

    return df

def dis_profile_summary(df):
    unit_cols = [c for c in df.columns if c != "Time"]
    out = pd.DataFrame({"Time": df["Time"].to_numpy()})

    values = df[unit_cols].to_numpy(dtype=float)
    out["n_units"] = np.sum(np.isfinite(values), axis=1)
    out["mean"] = np.nanmean(values, axis=1)
    out["sd"] = np.nanstd(values, axis=1, ddof=1)

    sd_mask = out["n_units"] <= 1
    out.loc[sd_mask, "sd"] = np.nan

    out["cv_pct"] = 100 * out["sd"] / out["mean"]
    out.loc[out["mean"] == 0, "cv_pct"] = np.nan

    return out

def dis_merge_profiles(ref_summary, test_summary):
    merged = ref_summary.merge(
        test_summary,
        on="Time",
        how="inner",
        suffixes=("_ref", "_test")
    )
    merged = merged.sort_values("Time").reset_index(drop=True)

    if len(merged) == 0:
        raise ValueError("Reference and Test have no common timepoints.")

    return merged

def dis_select_points(merged, include_zero=True, cutoff_mode="all", threshold=85.0):
    use = merged.copy()

    if not include_zero:
        use = use.loc[use["Time"] != 0].copy()

    if len(use) == 0:
        raise ValueError("No timepoints left after filtering.")

    first_both_ge_idx = None
    for i in range(len(use)):
        if use.loc[i, "mean_ref"] >= threshold and use.loc[i, "mean_test"] >= threshold:
            first_both_ge_idx = i
            break

    if cutoff_mode == "apply_85" and first_both_ge_idx is not None:
        use = use.iloc[:first_both_ge_idx + 1].copy()

    if len(use) < 3:
        raise ValueError("At least 3 selected timepoints are needed to calculate f2.")

    return use.reset_index(drop=True), first_both_ge_idx

def dis_calc_f2(ref_means, test_means):
    ref_means = np.asarray(ref_means, dtype=float)
    test_means = np.asarray(test_means, dtype=float)

    n = len(ref_means)
    if n < 1:
        return np.nan

    msd = np.mean((ref_means - test_means) ** 2)
    f2 = 50 * np.log10(100 / np.sqrt(1 + msd))
    return f2

def dis_get_unit_cols(df):
    return [c for c in df.columns if c != "Time"]

def dis_get_selected_matrix(df, selected_times):
    sub = df[df["Time"].isin(selected_times)].copy()
    sub = sub.sort_values("Time").reset_index(drop=True)

    times_sorted = np.sort(np.asarray(selected_times, dtype=float))
    if len(sub) != len(times_sorted) or not np.allclose(sub["Time"].to_numpy(dtype=float), times_sorted):
        raise ValueError("Selected timepoints could not be aligned back to the original profile table.")

    unit_cols = dis_get_unit_cols(df)
    mat = sub[unit_cols].to_numpy(dtype=float)
    return mat, unit_cols

def dis_fda_checks(ref_df, test_df, merged, selected, threshold=85.0, include_zero=False):
    ref_units = len(dis_get_unit_cols(ref_df))
    test_units = len(dis_get_unit_cols(test_df))

    same_original_times = np.array_equal(
        np.sort(ref_df["Time"].to_numpy(dtype=float)),
        np.sort(test_df["Time"].to_numpy(dtype=float))
    )

    same_selected_time_count = len(selected)
    at_least_12 = (ref_units >= 12) and (test_units >= 12)
    at_least_3_points = same_selected_time_count >= 3

    both_ge = (selected["mean_ref"] >= threshold) & (selected["mean_test"] >= threshold)
    n_post85_kept = int(both_ge.sum())
    one_post85_ok = n_post85_kept <= 1

    selected_nonzero = selected[selected["Time"] > 0].copy()
    if include_zero:
        selected_nonzero = selected.copy()

    early_cv_ref = np.nan
    early_cv_test = np.nan
    later_max_cv_ref = np.nan
    later_max_cv_test = np.nan
    cv_ok = True

    if len(selected_nonzero) > 0:
        early_cv_ref = selected_nonzero.iloc[0]["cv_pct_ref"]
        early_cv_test = selected_nonzero.iloc[0]["cv_pct_test"]

        later_ref = selected_nonzero.iloc[1:]["cv_pct_ref"].dropna()
        later_test = selected_nonzero.iloc[1:]["cv_pct_test"].dropna()

        later_max_cv_ref = later_ref.max() if len(later_ref) > 0 else np.nan
        later_max_cv_test = later_test.max() if len(later_test) > 0 else np.nan

        if pd.notna(early_cv_ref) and early_cv_ref > 20:
            cv_ok = False
        if pd.notna(early_cv_test) and early_cv_test > 20:
            cv_ok = False
        if pd.notna(later_max_cv_ref) and later_max_cv_ref > 10:
            cv_ok = False
        if pd.notna(later_max_cv_test) and later_max_cv_test > 10:
            cv_ok = False

    fda_tbl = pd.DataFrame([
        {"Criterion": "Same original timepoints in both profiles", "Pass": "Yes" if same_original_times else "No"},
        {"Criterion": "At least 12 units in Reference and Test", "Pass": "Yes" if at_least_12 else "No"},
        {"Criterion": "At least 3 selected timepoints for f2", "Pass": "Yes" if at_least_3_points else "No"},
        {"Criterion": "No more than one selected point after both are ≥ threshold", "Pass": "Yes" if one_post85_ok else "No"},
        {"Criterion": "CV at earlier selected timepoint ≤ 20% and later ≤ 10%", "Pass": "Yes" if cv_ok else "No"},
    ])

    detail_tbl = pd.DataFrame([{
        "Reference_units": ref_units,
        "Test_units": test_units,
        "Selected_timepoints": same_selected_time_count,
        "Selected_points_where_both_ge_threshold": n_post85_kept,
        "Earlier_CV_ref": early_cv_ref,
        "Earlier_CV_test": early_cv_test,
        "Later_max_CV_ref": later_max_cv_ref,
        "Later_max_CV_test": later_max_cv_test,
    }])

    conventional_ok = bool((fda_tbl["Pass"] == "Yes").all())

    return fda_tbl, detail_tbl, conventional_ok

def dis_bootstrap_f2(ref_mat, test_mat, n_boot=2000, seed=123):
    rng = np.random.default_rng(seed)

    n_ref = ref_mat.shape[1]
    n_test = test_mat.shape[1]

    out = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx_ref = rng.integers(0, n_ref, size=n_ref)
        idx_test = rng.integers(0, n_test, size=n_test)

        ref_mean = np.nanmean(ref_mat[:, idx_ref], axis=1)
        test_mean = np.nanmean(test_mat[:, idx_test], axis=1)

        out[b] = dis_calc_f2(ref_mean, test_mean)

    return out

def dis_jackknife_f2(ref_mat, test_mat):
    vals = []

    n_ref = ref_mat.shape[1]
    n_test = test_mat.shape[1]

    for j in range(n_ref):
        keep = [i for i in range(n_ref) if i != j]
        if len(keep) >= 1:
            ref_mean = np.nanmean(ref_mat[:, keep], axis=1)
            test_mean = np.nanmean(test_mat, axis=1)
            vals.append(dis_calc_f2(ref_mean, test_mean))

    for j in range(n_test):
        keep = [i for i in range(n_test) if i != j]
        if len(keep) >= 1:
            ref_mean = np.nanmean(ref_mat, axis=1)
            test_mean = np.nanmean(test_mat[:, keep], axis=1)
            vals.append(dis_calc_f2(ref_mean, test_mean))

    return np.asarray(vals, dtype=float)

def dis_bca_interval(theta_hat, boot_vals, jack_vals, conf=0.90):
    boot_vals = np.asarray(boot_vals, dtype=float)
    boot_vals = boot_vals[np.isfinite(boot_vals)]
    jack_vals = np.asarray(jack_vals, dtype=float)
    jack_vals = jack_vals[np.isfinite(jack_vals)]

    if len(boot_vals) < 10:
        return np.nan, np.nan, np.nan, np.nan

    alpha = 1 - conf

    prop_less = np.mean(boot_vals < theta_hat)
    eps = 1 / (2 * len(boot_vals))
    prop_less = np.clip(prop_less, eps, 1 - eps)
    z0 = norm.ppf(prop_less)

    if len(jack_vals) < 3:
        a = 0.0
    else:
        jack_mean = np.mean(jack_vals)
        num = np.sum((jack_mean - jack_vals) ** 3)
        den = 6 * (np.sum((jack_mean - jack_vals) ** 2) ** 1.5)
        a = num / den if den > 0 else 0.0

    z_low = norm.ppf(alpha / 2)
    z_high = norm.ppf(1 - alpha / 2)

    adj_low = norm.cdf(z0 + (z0 + z_low) / (1 - a * (z0 + z_low)))
    adj_high = norm.cdf(z0 + (z0 + z_high) / (1 - a * (z0 + z_high)))

    low = np.quantile(boot_vals, adj_low)
    high = np.quantile(boot_vals, adj_high)

    return low, high, z0, a

def dis_percentile_interval(boot_vals, conf=0.90):
    alpha = 1 - conf
    low = np.quantile(boot_vals, alpha / 2)
    high = np.quantile(boot_vals, 1 - alpha / 2)
    return low, high

def dis_plot_profiles(ref_df, test_df, ref_summary, test_summary, selected, show_units=True, title="Dissolution Profiles", ylabel="% Dissolved"):
    fig, ax = plt.subplots(figsize=(10, 6))

    ref_unit_cols = [c for c in ref_df.columns if c != "Time"]
    test_unit_cols = [c for c in test_df.columns if c != "Time"]

    if show_units:
        for c in ref_unit_cols:
            ax.plot(ref_df["Time"], ref_df[c], alpha=0.22, linewidth=1)
        for c in test_unit_cols:
            ax.plot(test_df["Time"], test_df[c], alpha=0.22, linewidth=1)

    ax.plot(ref_summary["Time"], ref_summary["mean"], marker="o", linewidth=2, label="Reference mean")
    ax.plot(test_summary["Time"], test_summary["mean"], marker="o", linewidth=2, label="Test mean")

    ax.scatter(selected["Time"], selected["mean_ref"], marker="s", s=55, label="Selected ref points")
    ax.scatter(selected["Time"], selected["mean_test"], marker="s", s=55, label="Selected test points")

    ax.set_xlabel("Time", fontsize=12, weight="bold")
    ax.set_ylabel(ylabel, fontsize=12, weight="bold")
    ax.set_title(title, fontsize=15, weight="bold")
    ax.legend(frameon=False)
    ax.grid(False)
    plt.tight_layout()
    plt.show()

def dis_plot_bootstrap_f2_distribution(
    boot_vals,
    observed_f2,
    ci_low=None,
    ci_high=None,
    ci_label="90% CI",
    title="Distribution of f2 Similarity Factor",
    x_min=50,
    x_max=100
):
    boot_vals = np.asarray(boot_vals, dtype=float)
    boot_vals = boot_vals[np.isfinite(boot_vals)]

    if len(boot_vals) < 5:
        print("Not enough bootstrap values to draw the distribution plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#e6e6e6")
    ax.set_facecolor("#f2f2f2")

    sd_boot = np.std(boot_vals, ddof=1)

    if sd_boot > 0:
        kde = gaussian_kde(boot_vals)
        x_lo = min(x_min, np.min(boot_vals) - 2 * sd_boot, observed_f2 - 5)
        x_hi = max(x_max, np.max(boot_vals) + 2 * sd_boot, observed_f2 + 5)

        if ci_low is not None:
            x_lo = min(x_lo, ci_low - 3)
        if ci_high is not None:
            x_hi = max(x_hi, ci_high + 3)

        x_grid = np.linspace(x_lo, x_hi, 600)
        y_grid = kde(x_grid)
        ax.plot(x_grid, y_grid, color="black", linewidth=1.6)
        y_top = float(np.max(y_grid))
    else:
        ax.axvline(observed_f2, color="black", linewidth=1.6)
        y_top = 1.0

    ax.axvline(observed_f2, color="black", linestyle=(0, (4, 4)), linewidth=1.4)

    if ci_low is not None:
        ax.axvline(ci_low, color="#00aa44", linestyle=(0, (4, 4)), linewidth=1.4)
    if ci_high is not None:
        ax.axvline(ci_high, color="#00aa44", linestyle=(0, (4, 4)), linewidth=1.4)

    ax.text(
        observed_f2 - 0.8,
        y_top * 0.52,
        f"Original f2, {observed_f2:.1f}",
        rotation=90,
        ha="right",
        va="center",
        fontsize=11,
        color="black",
        weight="bold"
    )

    if ci_low is not None:
        ax.text(
            ci_low - 0.8,
            y_top * 0.58,
            f"{ci_label} Lower CI, {ci_low:.1f}",
            rotation=90,
            ha="right",
            va="center",
            fontsize=11,
            color="#00aa44",
            weight="bold"
        )

    if ci_high is not None:
        ax.text(
            ci_high + 0.8,
            y_top * 0.58,
            f"{ci_label} Upper CI, {ci_high:.1f}",
            rotation=90,
            ha="left",
            va="center",
            fontsize=11,
            color="#00aa44",
            weight="bold"
        )

    ax.set_title(title, fontsize=16, weight="bold", pad=18)
    ax.set_xlabel("f2 Values", fontsize=12, weight="bold")
    ax.set_ylabel("Density", fontsize=12, weight="bold")
    ax.set_xlim(x_min, x_max)
    ax.grid(False)
    plt.tight_layout()
    plt.show()

def ts_make_unique(names):
    out = []
    seen = {}
    for i, n in enumerate(names):
        n = str(n).strip()
        if n == "" or n.lower() == "nan":
            n = f"Col{i+1}"
        if n in seen:
            seen[n] += 1
            n = f"{n}_{seen[n]}"
        else:
            seen[n] = 1
        out.append(n)
    return out

def ts_parse_raw_table(text):
    text = str(text).strip()
    if not text:
        raise ValueError("Paste data from Excel.")

    parsers = [
        lambda s: pd.read_csv(StringIO(s), sep="\t", header=None, engine="python"),
        lambda s: pd.read_csv(StringIO(s), sep=",", header=None, engine="python"),
        lambda s: pd.read_csv(StringIO(s), sep=";", header=None, engine="python"),
    ]

    df = None
    for parser in parsers:
        try:
            trial = parser(text)
            if trial.shape[1] >= 2:
                df = trial.copy()
                break
        except:
            pass

    if df is None or df.shape[1] < 2:
        raise ValueError("Could not read the pasted table.")

    df = df.dropna(how="all").reset_index(drop=True)
    return df

def ts_promote_header_if_needed(df_raw):
    first_row = df_raw.iloc[0].astype(str).str.strip()
    first_row_numeric = pd.to_numeric(first_row, errors="coerce").notna()

    if first_row_numeric.all():
        df = df_raw.copy()
        df.columns = [f"Col{i+1}" for i in range(df.shape[1])]
    else:
        df = df_raw.iloc[1:].reset_index(drop=True).copy()
        df.columns = ts_make_unique(first_row.tolist())

    df.columns = ts_make_unique(df.columns)
    return df

def ts_to_numeric(series):
    return pd.to_numeric(
        series.astype(str).str.strip().str.replace("%", "", regex=False),
        errors="coerce"
    )

def ts_guess_long_columns(df):
    numericish = []
    nonnumericish = []

    for col in df.columns:
        num = ts_to_numeric(df[col])
        frac_numeric = num.notna().mean()
        if frac_numeric >= 0.7:
            numericish.append(col)
        else:
            nonnumericish.append(col)

    response_col = numericish[-1] if len(numericish) > 0 else df.columns[-1]
    group_col = nonnumericish[0] if len(nonnumericish) > 0 else df.columns[0]

    if group_col == response_col and len(df.columns) >= 2:
        for c in df.columns:
            if c != response_col:
                group_col = c
                break

    return group_col, response_col

def ts_prepare_wide(df):
    groups = {}
    for col in df.columns:
        vals = ts_to_numeric(df[col]).dropna().to_numpy(dtype=float)
        if len(vals) > 0:
            groups[str(col)] = vals

    if len(groups) < 2:
        raise ValueError("Wide format requires at least two non-empty numeric columns.")

    for name, vals in groups.items():
        if len(vals) < 2:
            raise ValueError(f"Group '{name}' has fewer than 2 values.")

    long_df = pd.concat(
        [pd.DataFrame({"Group": name, "Response": vals}) for name, vals in groups.items()],
        ignore_index=True
    )

    return long_df, groups

def ts_prepare_wide_paired(df):
    if df.shape[1] != 2:
        raise ValueError("Paired mode currently requires exactly 2 columns in wide format.")

    col1, col2 = df.columns[0], df.columns[1]

    paired_df = pd.DataFrame({
        str(col1): ts_to_numeric(df[col1]),
        str(col2): ts_to_numeric(df[col2])
    }).dropna().reset_index(drop=True)

    if len(paired_df) < 2:
        raise ValueError("At least 2 complete pairs are required.")

    x1 = paired_df.iloc[:, 0].to_numpy(dtype=float)
    x2 = paired_df.iloc[:, 1].to_numpy(dtype=float)

    groups = {
        str(col1): x1,
        str(col2): x2
    }

    long_df = pd.concat(
        [
            pd.DataFrame({"Group": str(col1), "Response": x1}),
            pd.DataFrame({"Group": str(col2), "Response": x2})
        ],
        ignore_index=True
    )

    return long_df, groups, str(col1), str(col2), x1, x2

def ts_prepare_long(df, group_col, response_col):
    dfa = df[[group_col, response_col]].copy()
    dfa.columns = ["Group", "Response"]

    dfa["Group"] = dfa["Group"].astype(str).str.strip()
    dfa["Response"] = ts_to_numeric(dfa["Response"])
    dfa = dfa.dropna(subset=["Group", "Response"])
    dfa = dfa[dfa["Group"] != ""].reset_index(drop=True)

    if dfa["Group"].nunique() < 2:
        raise ValueError("Long format requires at least two group levels.")

    groups = {
        g: sub["Response"].to_numpy(dtype=float)
        for g, sub in dfa.groupby("Group", sort=False)
    }

    for name, vals in groups.items():
        if len(vals) < 2:
            raise ValueError(f"Group '{name}' has fewer than 2 values.")

    return dfa, groups

def ts_group_summary(long_df):
    return (
        long_df.groupby("Group", sort=False)["Response"]
        .agg(
            n="size",
            mean="mean",
            sd="std",
            median="median",
            min="min",
            max="max"
        )
        .reset_index()
    )

def ts_normality_table(groups, alpha):
    rows = []
    for name, vals in groups.items():
        try:
            a2_raw, p = normal_ad(vals)
            n = len(vals)
            a2_star = a2_raw * (1 + 0.75 / n + 2.25 / (n ** 2))
        except:
            a2_star, p = np.nan, np.nan

        rows.append({
            "Group": name,
            "n": len(vals),
            "AD_A_star": a2_star,
            "AD_pvalue": p,
            "Normal_at_alpha": "Yes" if (pd.notna(p) and p >= alpha) else "No"
        })
    return pd.DataFrame(rows)

def ts_paired_difference_normality_table(x1, x2, alpha, name1="Group 1", name2="Group 2"):
    diff = np.asarray(x1, dtype=float) - np.asarray(x2, dtype=float)

    try:
        a2_raw, p = normal_ad(diff)
        n = len(diff)
        a2_star = a2_raw * (1 + 0.75 / n + 2.25 / (n ** 2))
    except:
        a2_star, p = np.nan, np.nan

    return pd.DataFrame([{
        "Comparison": f"{name1} - {name2}",
        "n_pairs": len(diff),
        "AD_A_star": a2_star,
        "AD_pvalue": p,
        "Normal_at_alpha": "Yes" if (pd.notna(p) and p >= alpha) else "No"
    }])

def ts_variance_test(groups):
    arrays = list(groups.values())
    stat, p = stats.levene(*arrays, center="median")
    return stat, p

def ts_choose_test(groups, alpha, route_mode, design="independent", paired_data=None):
    k = len(groups)

    if design == "paired":
        if paired_data is None:
            raise ValueError("Paired data are required for paired design.")

        name1, name2, x1, x2 = paired_data
        norm_tbl = ts_paired_difference_normality_table(x1, x2, alpha, name1, name2)
        diff_normal = bool(norm_tbl["Normal_at_alpha"].iloc[0] == "Yes")

        if route_mode == "auto":
            return ("paired_t" if diff_normal else "wilcoxon"), norm_tbl, np.nan, np.nan

        if route_mode in ["parametric_equal", "welch"]:
            return "paired_t", norm_tbl, np.nan, np.nan

        if route_mode == "nonparametric":
            return "wilcoxon", norm_tbl, np.nan, np.nan

        raise ValueError("Invalid route mode for paired design.")

    norm_tbl = ts_normality_table(groups, alpha)
    all_normal = bool((norm_tbl["Normal_at_alpha"] == "Yes").all())

    lev_stat, lev_p = ts_variance_test(groups)
    equal_var = lev_p >= alpha

    if route_mode == "auto":
        if k == 2:
            if all_normal and equal_var:
                return "pooled_t", norm_tbl, lev_stat, lev_p
            elif all_normal and not equal_var:
                return "welch_t", norm_tbl, lev_stat, lev_p
            else:
                return "mannwhitney", norm_tbl, lev_stat, lev_p
        else:
            if all_normal and equal_var:
                return "anova", norm_tbl, lev_stat, lev_p
            elif all_normal and not equal_var:
                return "welch_anova", norm_tbl, lev_stat, lev_p
            else:
                return "kruskal", norm_tbl, lev_stat, lev_p

    if route_mode == "parametric_equal":
        return ("pooled_t" if k == 2 else "anova"), norm_tbl, lev_stat, lev_p

    if route_mode == "welch":
        return ("welch_t" if k == 2 else "welch_anova"), norm_tbl, lev_stat, lev_p

    if route_mode == "nonparametric":
        return ("mannwhitney" if k == 2 else "kruskal"), norm_tbl, lev_stat, lev_p

    raise ValueError("Invalid route mode.")

def ts_run_test(groups, selected_test, paired_data=None):
    names = list(groups.keys())
    arrays = [groups[n] for n in names]

    if selected_test == "paired_t":
        if paired_data is None:
            raise ValueError("Paired data are required for paired t-test.")
        name1, name2, x1, x2 = paired_data
        stat, p = stats.ttest_rel(x1, x2)
        diffs = x1 - x2
        return {
            "Test": "Paired t-test",
            "Statistic_name": "t",
            "Statistic": stat,
            "pvalue": p,
            "Mean_difference": float(np.mean(diffs))
        }

    if selected_test == "wilcoxon":
        if paired_data is None:
            raise ValueError("Paired data are required for Wilcoxon signed-rank test.")
        name1, name2, x1, x2 = paired_data
        stat, p = stats.wilcoxon(x1, x2, alternative="two-sided", method="auto")
        diffs = x1 - x2
        return {
            "Test": "Wilcoxon signed-rank test",
            "Statistic_name": "W",
            "Statistic": stat,
            "pvalue": p,
            "Median_difference": float(np.median(diffs))
        }

    if selected_test == "pooled_t":
        stat, p = stats.ttest_ind(arrays[0], arrays[1], equal_var=True)
        return {
            "Test": "Pooled two-sample t-test",
            "Statistic_name": "t",
            "Statistic": stat,
            "pvalue": p
        }

    if selected_test == "welch_t":
        stat, p = stats.ttest_ind(arrays[0], arrays[1], equal_var=False)
        return {
            "Test": "Welch t-test",
            "Statistic_name": "t",
            "Statistic": stat,
            "pvalue": p
        }

    if selected_test == "mannwhitney":
        stat, p = stats.mannwhitneyu(arrays[0], arrays[1], alternative="two-sided", method="auto")
        return {
            "Test": "Mann-Whitney U test",
            "Statistic_name": "U",
            "Statistic": stat,
            "pvalue": p
        }

    if selected_test == "anova":
        stat, p = stats.f_oneway(*arrays)
        return {
            "Test": "One-way ANOVA",
            "Statistic_name": "F",
            "Statistic": stat,
            "pvalue": p
        }

    if selected_test == "welch_anova":
        res = anova_oneway(arrays, use_var="unequal")
        return {
            "Test": "Welch ANOVA",
            "Statistic_name": "F",
            "Statistic": float(res.statistic),
            "pvalue": float(res.pvalue)
        }

    if selected_test == "kruskal":
        stat, p = stats.kruskal(*arrays)
        return {
            "Test": "Kruskal-Wallis test",
            "Statistic_name": "H",
            "Statistic": stat,
            "pvalue": p
        }

    raise ValueError("Unknown test.")

def ts_plot_groups(long_df, title="Group Comparison", ylabel="Response", show_points=True, design="independent"):
    groups = list(pd.unique(long_df["Group"]))
    data = [long_df.loc[long_df["Group"] == g, "Response"].to_numpy() for g in groups]

    colors = ["black", "firebrick"]

    plt.figure(figsize=(9.5, 6))

    bp = plt.boxplot(
        data,
        tick_labels=groups,
        showmeans=False,
        patch_artist=False,
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        medianprops=dict(linewidth=1.8),
        flierprops=dict(marker="o", markersize=4)
    )

    for i, box in enumerate(bp["boxes"]):
        color = colors[i % len(colors)]
        box.set_color(color)

    for i, whisker in enumerate(bp["whiskers"]):
        color = colors[(i // 2) % len(colors)]
        whisker.set_color(color)

    for i, cap in enumerate(bp["caps"]):
        color = colors[(i // 2) % len(colors)]
        cap.set_color(color)

    for i, median in enumerate(bp["medians"]):
        color = colors[i % len(colors)]
        median.set_color(color)

    for i, flier in enumerate(bp["fliers"]):
        color = colors[i % len(colors)]
        flier.set_markeredgecolor(color)
        flier.set_markerfacecolor(color)

    if show_points:
        rng = np.random.default_rng(12345)
        for i, vals in enumerate(data, start=1):
            color = colors[(i - 1) % len(colors)]
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            plt.scatter(
                np.full(len(vals), i) + jitter,
                vals,
                alpha=0.45,
                s=20,
                color=color,
                zorder=3
            )

    if design == "paired" and len(groups) == 2 and len(data[0]) == len(data[1]):
        x1, x2 = 1, 2
        for y1, y2 in zip(data[0], data[1]):
            plt.plot([x1, x2], [y1, y2], color="gray", alpha=0.5, linewidth=1.0, zorder=1)

    for i, vals in enumerate(data, start=1):
        color = colors[(i - 1) % len(colors)]
        mean_val = np.mean(vals)
        plt.scatter(
            i,
            mean_val,
            marker="x",
            s=90,
            linewidths=2.0,
            color=color,
            zorder=4
        )

    plt.xlabel("Group", fontsize=12, weight="bold")
    plt.ylabel(ylabel, fontsize=12, weight="bold")
    plt.title(title, fontsize=15, weight="bold")
    plt.grid(False)
    plt.tight_layout()
    plt.show()

def twa_parse_pasted_table(text):
    text = str(text).strip()
    if not text:
        raise ValueError("Paste a table from Excel.")

    parsers = [
        lambda s: pd.read_csv(StringIO(s), sep="\t", engine="python"),
        lambda s: pd.read_csv(StringIO(s), sep=",", engine="python"),
        lambda s: pd.read_csv(StringIO(s), sep=";", engine="python"),
    ]

    df = None
    for parser in parsers:
        try:
            trial = parser(text)
            if trial.shape[1] >= 3:
                df = trial.copy()
                break
        except:
            pass

    if df is None or df.shape[1] < 3:
        raise ValueError("Could not read the table. Paste at least 3 columns with headers.")

    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all").reset_index(drop=True)

    return df

def twa_clean_dataframe(df):
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].astype(str).str.strip()
    return out

def twa_guess_columns(df):
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(pd.to_numeric(df[c], errors="coerce"))]
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

    response = numeric_cols[-1] if numeric_cols else df.columns[-1]
    factors = [c for c in df.columns if c != response]

    factor_a = factors[0] if len(factors) >= 1 else df.columns[0]
    factor_b = factors[1] if len(factors) >= 2 else df.columns[1] if len(df.columns) > 1 else df.columns[0]

    return response, factor_a, factor_b

def twa_prepare_analysis_df(df, response_col, factor_a_col, factor_b_col):
    dfa = df[[response_col, factor_a_col, factor_b_col]].copy()
    dfa.columns = ["Response", "FactorA", "FactorB"]

    dfa["Response"] = pd.to_numeric(
        dfa["Response"].astype(str).str.replace("%", "", regex=False).str.strip(),
        errors="coerce"
    )

    dfa["FactorA"] = dfa["FactorA"].astype(str).str.strip()
    dfa["FactorB"] = dfa["FactorB"].astype(str).str.strip()

    dfa = dfa.dropna(subset=["Response", "FactorA", "FactorB"]).reset_index(drop=True)

    if len(dfa) < 4:
        raise ValueError("Not enough valid rows after cleaning.")
    if dfa["FactorA"].nunique() < 2:
        raise ValueError("Factor A must have at least 2 levels.")
    if dfa["FactorB"].nunique() < 2:
        raise ValueError("Factor B must have at least 2 levels.")

    return dfa

def twa_fit_model(df_analysis, include_interaction=True, typ=2):
    formula = "Response ~ C(FactorA) + C(FactorB)"
    if include_interaction:
        formula += " + C(FactorA):C(FactorB)"

    model = smf.ols(formula, data=df_analysis).fit()
    anova_tbl = anova_lm(model, typ=typ).reset_index().rename(columns={"index": "Source"})

    if "sum_sq" in anova_tbl.columns:
        total_ss = anova_tbl["sum_sq"].sum()
        if total_ss > 0:
            anova_tbl["Pct_SS"] = 100 * anova_tbl["sum_sq"] / total_ss

    return model, anova_tbl

def twa_make_group_summary(df_analysis):
    summary = (
        df_analysis
        .groupby(["FactorA", "FactorB"], dropna=False)
        .agg(
            n=("Response", "size"),
            mean=("Response", "mean"),
            sd=("Response", "std"),
            sem=("Response", lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan),
            min=("Response", "min"),
            max=("Response", "max")
        )
        .reset_index()
    )
    return summary

def twa_make_means_table(df_analysis):
    means = df_analysis.pivot_table(
        index="FactorA",
        columns="FactorB",
        values="Response",
        aggfunc="mean"
    )
    return means

def twa_make_counts_table(df_analysis):
    counts = df_analysis.pivot_table(
        index="FactorA",
        columns="FactorB",
        values="Response",
        aggfunc="size"
    )
    return counts

def twa_plot_interaction(df_analysis, ylabel="Response", title="Interaction Plot", show_points=True):
    means = (
        df_analysis
        .groupby(["FactorA", "FactorB"], dropna=False)["Response"]
        .mean()
        .reset_index()
    )

    factor_a_levels = list(pd.unique(df_analysis["FactorA"]))
    factor_b_levels = list(pd.unique(df_analysis["FactorB"]))

    plt.figure(figsize=(9.5, 6))

    x = np.arange(len(factor_a_levels))

    for b in factor_b_levels:
        sub = means[means["FactorB"] == b].copy()
        yvals = []
        for a in factor_a_levels:
            match = sub.loc[sub["FactorA"] == a, "Response"]
            yvals.append(match.iloc[0] if len(match) > 0 else np.nan)

        plt.plot(x, yvals, marker="o", linewidth=1.8, label=str(b))

    if show_points:
        jitter = np.linspace(-0.08, 0.08, max(2, len(factor_b_levels)))
        for j, b in enumerate(factor_b_levels):
            sub_raw = df_analysis[df_analysis["FactorB"] == b]
            for i, a in enumerate(factor_a_levels):
                vals = sub_raw.loc[sub_raw["FactorA"] == a, "Response"].to_numpy()
                if len(vals) > 0:
                    plt.scatter(np.full(len(vals), x[i] + jitter[j]), vals, alpha=0.35, s=18)

    plt.xticks(x, factor_a_levels)
    plt.xlabel("Factor A", fontsize=12, weight="bold")
    plt.ylabel(ylabel, fontsize=12, weight="bold")
    plt.title(title, fontsize=15, weight="bold")
    plt.legend(title="Factor B", frameon=False)
    plt.grid(False)
    plt.tight_layout()
    plt.show()

def tc_make_unique(names):
    out = []
    seen = {}
    for i, n in enumerate(names):
        n = str(n).strip()
        if n == "" or n.lower() == "nan":
            n = f"Col{i+1}"
        if n in seen:
            seen[n] += 1
            n = f"{n}_{seen[n]}"
        else:
            seen[n] = 1
        out.append(n)
    return out

def tc_parse_raw_table(text):
    text = str(text).strip()
    if not text:
        raise ValueError("Paste data from Excel.")

    parsers = [
        lambda s: pd.read_csv(StringIO(s), sep="\t", header=None, engine="python"),
        lambda s: pd.read_csv(StringIO(s), sep=",", header=None, engine="python"),
        lambda s: pd.read_csv(StringIO(s), sep=";", header=None, engine="python"),
    ]

    df = None
    for parser in parsers:
        try:
            trial = parser(text)
            if trial.shape[1] >= 2:
                df = trial.copy()
                break
        except:
            pass

    if df is None or df.shape[1] < 2:
        raise ValueError("Could not read the pasted table.")

    df = df.dropna(how="all").reset_index(drop=True)
    return df

def tc_promote_header_if_needed(df_raw):
    first_row = df_raw.iloc[0].astype(str).str.strip()
    first_row_numeric = pd.to_numeric(first_row, errors="coerce").notna()

    if first_row_numeric.all():
        df = df_raw.copy()
        df.columns = [f"Col{i+1}" for i in range(df.shape[1])]
    else:
        df = df_raw.iloc[1:].reset_index(drop=True).copy()
        df.columns = tc_make_unique(first_row.tolist())

    df.columns = tc_make_unique(df.columns)
    return df

def tc_to_numeric(series):
    return pd.to_numeric(
        series.astype(str).str.strip().str.replace("%", "", regex=False),
        errors="coerce"
    )

def tc_guess_long_columns(df):
    numericish = []
    nonnumericish = []

    for col in df.columns:
        num = tc_to_numeric(df[col])
        frac_numeric = num.notna().mean()
        if frac_numeric >= 0.7:
            numericish.append(col)
        else:
            nonnumericish.append(col)

    response_col = numericish[-1] if len(numericish) > 0 else df.columns[-1]
    group_col = nonnumericish[0] if len(nonnumericish) > 0 else df.columns[0]

    if group_col == response_col and len(df.columns) >= 2:
        for c in df.columns:
            if c != response_col:
                group_col = c
                break

    return group_col, response_col

def tc_prepare_wide(df):
    groups = {}
    for col in df.columns:
        vals = tc_to_numeric(df[col]).dropna().to_numpy(dtype=float)
        if len(vals) > 0:
            groups[str(col)] = vals

    if len(groups) != 2:
        raise ValueError("Wide format for this app must contain exactly two non-empty numeric columns.")
    for name, vals in groups.items():
        if len(vals) < 2:
            raise ValueError(f"Group '{name}' has fewer than 2 values.")

    long_df = pd.concat(
        [pd.DataFrame({"Group": name, "Response": vals}) for name, vals in groups.items()],
        ignore_index=True
    )
    return long_df, groups

def tc_prepare_long(df, group_col, response_col):
    dfa = df[[group_col, response_col]].copy()
    dfa.columns = ["Group", "Response"]

    dfa["Group"] = dfa["Group"].astype(str).str.strip()
    dfa["Response"] = tc_to_numeric(dfa["Response"])
    dfa = dfa.dropna(subset=["Group", "Response"])
    dfa = dfa[dfa["Group"] != ""].reset_index(drop=True)

    groups = {
        g: sub["Response"].to_numpy(dtype=float)
        for g, sub in dfa.groupby("Group", sort=False)
    }

    if len(groups) != 2:
        raise ValueError("Long format for this app must contain exactly two groups.")
    for name, vals in groups.items():
        if len(vals) < 2:
            raise ValueError(f"Group '{name}' has fewer than 2 values.")

    return dfa, groups

def tc_ad_table(groups, alpha):
    rows = []
    for name, vals in groups.items():
        try:
            a2_raw, p = normal_ad(vals)
            n = len(vals)
            a2_star = a2_raw * (1 + 0.75 / n + 2.25 / (n ** 2))
        except:
            a2_star, p = np.nan, np.nan

        rows.append({
            "Group": name,
            "n": len(vals),
            "AD_A_star": a2_star,
            "AD_pvalue": p,
            "Normal_at_alpha": "Yes" if (pd.notna(p) and p >= alpha) else "No"
        })
    return pd.DataFrame(rows)

def tc_group_summary(groups):
    rows = []
    for name, vals in groups.items():
        vals = np.asarray(vals, dtype=float)
        rows.append({
            "Group": name,
            "n": len(vals),
            "mean": np.mean(vals),
            "sd": np.std(vals, ddof=1),
            "median": np.median(vals),
            "min": np.min(vals),
            "max": np.max(vals)
        })
    return pd.DataFrame(rows)

def tc_mean_ci(x, conf=0.95):
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = np.mean(x)
    sd = np.std(x, ddof=1)
    se = sd / np.sqrt(n)
    tcrit = t.ppf(1 - (1 - conf) / 2, n - 1)
    lcl = mean - tcrit * se
    ucl = mean + tcrit * se
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "se": se,
        "df": n - 1,
        "CI_confidence": conf,
        "CI_lower": lcl,
        "CI_upper": ucl
    }

def tc_diff_mean_ci(x1, x2, conf=0.95, method="auto", alpha_var=0.05):
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)

    n1, n2 = len(x1), len(x2)
    m1, m2 = np.mean(x1), np.mean(x2)
    s1, s2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
    diff = m1 - m2

    lev_stat, lev_p = stats.levene(x1, x2, center="median")
    equal_var = lev_p >= alpha_var

    if method == "auto":
        method_used = "pooled" if equal_var else "welch"
    else:
        method_used = method

    if method_used == "pooled":
        sp2 = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2)
        se = np.sqrt(sp2 * (1 / n1 + 1 / n2))
        df = n1 + n2 - 2
    else:
        se = np.sqrt(s1**2 / n1 + s2**2 / n2)
        num = (s1**2 / n1 + s2**2 / n2) ** 2
        den = ((s1**2 / n1) ** 2) / (n1 - 1) + ((s2**2 / n2) ** 2) / (n2 - 1)
        df = num / den

    tcrit = t.ppf(1 - (1 - conf) / 2, df)
    lcl = diff - tcrit * se
    ucl = diff + tcrit * se

    return {
        "Comparison": f"{list(['Group1','Group2'])}",
        "Mean_diff_G1_minus_G2": diff,
        "SE": se,
        "df": df,
        "Method": method_used,
        "Levene_stat": lev_stat,
        "Levene_pvalue": lev_p,
        "Equal_variance_at_alpha": "Yes" if equal_var else "No",
        "CI_confidence": conf,
        "CI_lower": lcl,
        "CI_upper": ucl
    }

def tc_one_sided_ti(x, conf=0.95, coverage=0.95, side="upper"):
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = np.mean(x)
    sd = np.std(x, ddof=1)

    zp = norm.ppf(coverage)
    k = nct.ppf(conf, df=n - 1, nc=np.sqrt(n) * zp) / np.sqrt(n)

    if side == "upper":
        lower = np.nan
        upper = mean + k * sd
    else:
        lower = mean - k * sd
        upper = np.nan

    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "Tolerance_confidence": conf,
        "Tolerance_coverage": coverage,
        "Tolerance_side": side,
        "Method": "Exact one-sided noncentral t",
        "k_factor": k,
        "TI_lower": lower,
        "TI_upper": upper
    }

def tc_two_sided_ti_howe(x, conf=0.95, coverage=0.95):
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = np.mean(x)
    sd = np.std(x, ddof=1)

    alpha = 1 - conf
    z = norm.ppf((1 + coverage) / 2)
    chi = chi2.ppf(alpha, n - 1)
    k = z * np.sqrt((n - 1) * (1 + 1 / n) / chi)

    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "Tolerance_confidence": conf,
        "Tolerance_coverage": coverage,
        "Tolerance_side": "two-sided",
        "Method": "Howe two-sided approximation",
        "k_factor": k,
        "TI_lower": mean - k * sd,
        "TI_upper": mean + k * sd
    }

def tc_ti_table(groups, conf=0.95, coverage=0.95, side="two-sided"):
    rows = []
    for name, vals in groups.items():
        if side == "two-sided":
            res = tc_two_sided_ti_howe(vals, conf=conf, coverage=coverage)
        else:
            res = tc_one_sided_ti(vals, conf=conf, coverage=coverage, side=side)
        res["Group"] = name
        rows.append(res)
    cols = [
        "Group", "n", "mean", "sd", "Tolerance_side", "Tolerance_confidence",
        "Tolerance_coverage", "Method", "k_factor", "TI_lower", "TI_upper"
    ]
    return pd.DataFrame(rows)[cols]

def tc_ci_table(groups, conf=0.95):
    rows = []
    for name, vals in groups.items():
        res = tc_mean_ci(vals, conf=conf)
        res["Group"] = name
        rows.append(res)
    cols = ["Group", "n", "mean", "sd", "se", "df", "CI_confidence", "CI_lower", "CI_upper"]
    return pd.DataFrame(rows)[cols]

def tc_plot_intervals(long_df, ci_tbl, ti_tbl, title="Intervals by Group", ylabel="Response", show_points=True):
    groups = list(ci_tbl["Group"])
    x_positions = np.arange(1, len(groups) + 1)

    fig, ax = plt.subplots(figsize=(9.5, 6))

    if show_points:
        rng = np.random.default_rng(12345)
        for i, g in enumerate(groups, start=1):
            vals = long_df.loc[long_df["Group"] == g, "Response"].to_numpy(dtype=float)
            jitter = rng.uniform(-0.06, 0.06, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, alpha=0.45, s=22)

    for i, g in enumerate(groups):
        x = x_positions[i]
        mean = float(ci_tbl.loc[ci_tbl["Group"] == g, "mean"].iloc[0])
        ci_l = float(ci_tbl.loc[ci_tbl["Group"] == g, "CI_lower"].iloc[0])
        ci_u = float(ci_tbl.loc[ci_tbl["Group"] == g, "CI_upper"].iloc[0])

        ax.vlines(x - 0.08, ci_l, ci_u, linewidth=2.4)
        ax.hlines([ci_l, ci_u], x - 0.14, x - 0.02, linewidth=2.0)
        ax.scatter([x - 0.08], [mean], marker="s", s=45)

        ti_l = ti_tbl.loc[ti_tbl["Group"] == g, "TI_lower"].iloc[0]
        ti_u = ti_tbl.loc[ti_tbl["Group"] == g, "TI_upper"].iloc[0]
        side = ti_tbl.loc[ti_tbl["Group"] == g, "Tolerance_side"].iloc[0]

        if side == "two-sided":
            ax.vlines(x + 0.08, ti_l, ti_u, linewidth=1.8, linestyles="--")
            ax.hlines([ti_l, ti_u], x + 0.02, x + 0.14, linewidth=1.6, linestyles="--")
        elif side == "upper" and pd.notna(ti_u):
            ax.vlines(x + 0.08, mean, ti_u, linewidth=1.8, linestyles="--")
            ax.hlines(ti_u, x + 0.02, x + 0.14, linewidth=1.6, linestyles="--")
        elif side == "lower" and pd.notna(ti_l):
            ax.vlines(x + 0.08, ti_l, mean, linewidth=1.8, linestyles="--")
            ax.hlines(ti_l, x + 0.02, x + 0.14, linewidth=1.6, linestyles="--")

        ax.scatter([x], [mean], s=30)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], linestyle="-", linewidth=2.4, label="Mean CI"),
        Line2D([0], [0], linestyle="--", linewidth=1.8, label="Tolerance interval/bound"),
        Line2D([0], [0], marker="o", linestyle="", label="Mean")
    ]
    if show_points:
        handles.append(Line2D([0], [0], marker="o", linestyle="", alpha=0.45, label="Raw points"))

    ax.set_xticks(x_positions)
    ax.set_xticklabels(groups)
    ax.set_xlabel("Group", fontsize=12, weight="bold")
    ax.set_ylabel(ylabel, fontsize=12, weight="bold")
    ax.set_title(title, fontsize=15, weight="bold")
    ax.legend(handles=handles, frameon=False, loc="best")
    ax.grid(False)
    plt.tight_layout()
    plt.show()

def pca_make_unique(names):
    out = []
    seen = {}
    for i, n in enumerate(names):
        n = str(n).strip()
        if n == "" or n.lower() == "nan":
            n = f"Col{i+1}"
        if n in seen:
            seen[n] += 1
            n = f"{n}_{seen[n]}"
        else:
            seen[n] = 1
        out.append(n)
    return out

def pca_parse_table(text):
    text = str(text).strip()
    if not text:
        raise ValueError("Paste a table with headers from Excel.")

    parsers = [
        lambda s: pd.read_csv(StringIO(s), sep="\t", engine="python"),
        lambda s: pd.read_csv(StringIO(s), sep=",", engine="python"),
        lambda s: pd.read_csv(StringIO(s), sep=";", engine="python"),
    ]

    df = None
    for parser in parsers:
        try:
            trial = parser(text)
            if trial.shape[1] >= 2:
                df = trial.copy()
                break
        except:
            pass

    if df is None or df.shape[1] < 2:
        raise ValueError("Could not read the pasted table.")

    df.columns = pca_make_unique(df.columns)
    df = df.dropna(how="all").reset_index(drop=True)

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    return df

def pca_numeric_candidates(df, min_fraction=0.7):
    numeric_cols = []
    for col in df.columns:
        vals = pd.to_numeric(
            df[col].astype(str).str.replace("%", "", regex=False).str.strip(),
            errors="coerce"
        )
        if vals.notna().mean() >= min_fraction:
            numeric_cols.append(col)
    return numeric_cols

def pca_prepare_matrix(df, feature_cols, label_col=None, group_col=None):
    dfa = df.copy()

    X = pd.DataFrame(index=dfa.index)
    for col in feature_cols:
        X[col] = pd.to_numeric(
            dfa[col].astype(str).str.replace("%", "", regex=False).str.strip(),
            errors="coerce"
        )

    keep = X.notna().all(axis=1)
    X = X.loc[keep].reset_index(drop=True)

    meta = pd.DataFrame(index=np.arange(len(X)))

    if label_col is None or label_col == "(None)":
        meta["Label"] = [f"Obs{i+1}" for i in range(len(X))]
    else:
        meta["Label"] = dfa.loc[keep, label_col].astype(str).reset_index(drop=True)

    if group_col is None or group_col == "(None)":
        meta["Group"] = "All"
    else:
        meta["Group"] = dfa.loc[keep, group_col].astype(str).reset_index(drop=True)

    if X.shape[0] < 2:
        raise ValueError("Not enough complete rows after removing missing values.")
    if X.shape[1] < 2:
        raise ValueError("Choose at least two numeric variables.")

    return X, meta

def pca_preprocess(X, mode):
    Xv = X.to_numpy(dtype=float)
    means = Xv.mean(axis=0)
    sds = Xv.std(axis=0, ddof=1)

    if mode == "none":
        Xp = Xv.copy()
    elif mode == "center":
        Xp = Xv - means
    elif mode == "autoscale":
        sds_safe = np.where(sds == 0, 1.0, sds)
        Xp = (Xv - means) / sds_safe
    else:
        raise ValueError("Invalid preprocessing mode.")

    prep_info = pd.DataFrame({
        "Variable": X.columns,
        "Mean": means,
        "SD": sds
    })

    return Xp, prep_info

def pca_run(X, n_components):
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)

    score_cols = [f"PC{i+1}" for i in range(scores.shape[1])]
    scores_df = pd.DataFrame(scores, columns=score_cols)

    loadings = pca.components_.T
    loading_cols = [f"PC{i+1}" for i in range(loadings.shape[1])]
    loadings_df = pd.DataFrame(loadings, index=None, columns=loading_cols)

    explained_df = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
        "Eigenvalue": pca.explained_variance_,
        "Explained_Variance_%": 100 * pca.explained_variance_ratio_,
        "Cumulative_%": 100 * np.cumsum(pca.explained_variance_ratio_)
    })

    return pca, scores_df, loadings_df, explained_df

def pca_plot_scree(explained_df, title="Scree Plot"):
    x = np.arange(1, len(explained_df) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(x, explained_df["Explained_Variance_%"].to_numpy(), marker="o")
    plt.xticks(x, [f"PC{i}" for i in x])
    plt.xlabel("Principal Component", fontsize=12, weight="bold")
    plt.ylabel("Explained Variance (%)", fontsize=12, weight="bold")
    plt.title(title, fontsize=15, weight="bold")
    plt.grid(False)
    plt.tight_layout()
    plt.show()

def pca_plot_cumulative(explained_df, title="Cumulative Explained Variance"):
    x = np.arange(1, len(explained_df) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(x, explained_df["Cumulative_%"].to_numpy(), marker="o")
    plt.xticks(x, [f"PC{i}" for i in x])
    plt.xlabel("Principal Component", fontsize=12, weight="bold")
    plt.ylabel("Cumulative Variance (%)", fontsize=12, weight="bold")
    plt.title(title, fontsize=15, weight="bold")
    plt.ylim(0, 105)
    plt.grid(False)
    plt.tight_layout()
    plt.show()

def pca_ellipse_scale(mode):
    if mode == "none":
        return None
    if mode == "3sigma":
        return 3.0
    if mode == "mahal_95":
        return np.sqrt(5.991464547107979)   # chi-square df=2, 95%
    if mode == "mahal_99":
        return np.sqrt(9.21034037197618)    # chi-square df=2, 99%
    raise ValueError("Invalid ellipse mode.")

def pca_add_confidence_ellipse(ax, x, y, mode="3sigma", label=None, linewidth=1.5, color=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return

    scale = pca_ellipse_scale(mode)
    if scale is None:
        return

    cov = np.cov(x, y)
    if cov.shape != (2, 2):
        return

    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    if np.any(vals < 0):
        return

    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width = 2 * scale * np.sqrt(vals[0])
    height = 2 * scale * np.sqrt(vals[1])

    ellipse = Ellipse(
        xy=(np.mean(x), np.mean(y)),
        width=width,
        height=height,
        angle=angle,
        fill=False,
        edgecolor=color,
        linewidth=linewidth,
        label=label
    )
    ax.add_patch(ellipse)

def pca_plot_scores(
    scores_df,
    meta_df,
    explained_df,
    show_labels=True,
    title="Scores Plot",
    ellipse_mode="none",
    ellipse_scope="overall"
):
    if scores_df.shape[1] < 2:
        raise ValueError("Need at least 2 PCs to draw a scores plot.")

    pc1_name = "PC1"
    pc2_name = "PC2"
    xlab = f"{pc1_name} ({explained_df.loc[0, 'Explained_Variance_%']:.2f}%)"
    ylab = f"{pc2_name} ({explained_df.loc[1, 'Explained_Variance_%']:.2f}%)"

    fig, ax = plt.subplots(figsize=(8, 6))

    groups = pd.unique(meta_df["Group"])
    group_colors = {}

    for g in groups:
        mask = meta_df["Group"] == g
        sc = ax.scatter(
            scores_df.loc[mask, pc1_name],
            scores_df.loc[mask, pc2_name],
            label=str(g),
            alpha=0.8,
            s=40
        )

        color = sc.get_facecolor()[0]
        group_colors[g] = color

    if show_labels:
        for i in range(len(scores_df)):
            ax.text(
                scores_df.loc[i, pc1_name],
                scores_df.loc[i, pc2_name],
                str(meta_df.loc[i, "Label"]),
                fontsize=9
            )

    if ellipse_mode != "none":
        if ellipse_scope == "overall":
            pca_add_confidence_ellipse(
                ax,
                scores_df[pc1_name].to_numpy(),
                scores_df[pc2_name].to_numpy(),
                mode=ellipse_mode,
                label="Ellipse",
                color="black"
            )
        elif ellipse_scope == "by_group":
            for g in groups:
                mask = meta_df["Group"] == g
                pca_add_confidence_ellipse(
                    ax,
                    scores_df.loc[mask, pc1_name].to_numpy(),
                    scores_df.loc[mask, pc2_name].to_numpy(),
                    mode=ellipse_mode,
                    label=f"{g} ellipse",
                    color=group_colors[g]
                )

    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)
    ax.set_xlabel(xlab, fontsize=12, weight="bold")
    ax.set_ylabel(ylab, fontsize=12, weight="bold")
    ax.set_title(title, fontsize=15, weight="bold")

    if len(groups) > 1 or str(groups[0]) != "All" or ellipse_mode != "none":
        ax.legend(frameon=False, title="Group")

    ax.grid(False)
    plt.tight_layout()
    plt.show()

def pca_plot_loadings(loadings_df, variable_names, explained_df, title="Loadings Plot"):
    if loadings_df.shape[1] < 2:
        raise ValueError("Need at least 2 PCs to draw a loadings plot.")

    pc1 = loadings_df["PC1"].to_numpy()
    pc2 = loadings_df["PC2"].to_numpy()

    plt.figure(figsize=(8, 6))
    plt.axhline(0, linewidth=1)
    plt.axvline(0, linewidth=1)

    for i, var in enumerate(variable_names):
        plt.arrow(0, 0, pc1[i], pc2[i], head_width=0.03, length_includes_head=True)
        plt.text(pc1[i], pc2[i], str(var), fontsize=10)

    lim = max(1.0, np.max(np.abs(np.concatenate([pc1, pc2]))) * 1.2)
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)

    xlab = f"PC1 ({explained_df.loc[0, 'Explained_Variance_%']:.2f}%)"
    ylab = f"PC2 ({explained_df.loc[1, 'Explained_Variance_%']:.2f}%)"

    plt.xlabel(xlab, fontsize=12, weight="bold")
    plt.ylabel(ylab, fontsize=12, weight="bold")
    plt.title(title, fontsize=15, weight="bold")
    plt.grid(False)
    plt.tight_layout()
    plt.show()



# ------------------------------
# Shared Streamlit helpers
# ------------------------------
def round_df(df, decimals=4):
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].round(decimals)
    return out

def display_df(df, decimals=4):
    st.dataframe(round_df(df, decimals), use_container_width=True)

def render_plot(plot_func, *args, **kwargs):
    plt.close("all")
    plot_func(*args, **kwargs)
    fig = plt.gcf()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

def safe_parse_float(text, default=None):
    text = str(text).strip()
    if text == "":
        return default
    return float(text)

# ------------------------------
# Sample data
# ------------------------------
SHELF_SAMPLE_DATA = "0\t100\n3\t99.2\n6\t98.4\n9\t97.8\n12\t97.0\n18\t95.6\n24\t94.8"
SHELF_SAMPLE_PRED = "30\n36\n48"

DISS_REF_SAMPLE = """Time\tR1\tR2\tR3\tR4\tR5\tR6
5\t28\t30\t27\t29\t31\t28
10\t52\t54\t51\t53\t55\t52
15\t71\t73\t70\t72\t74\t71
20\t84\t86\t83\t85\t87\t84
30\t93\t94\t92\t93\t95\t93
45\t97\t98\t96\t97\t98\t97
"""

DISS_TEST_SAMPLE = """Time\tT1\tT2\tT3\tT4\tT5\tT6
5\t26\t28\t25\t27\t29\t26
10\t49\t51\t48\t50\t52\t49
15\t69\t70\t68\t69\t71\t68
20\t82\t84\t81\t83\t85\t82
30\t92\t93\t91\t92\t94\t92
45\t97\t97\t96\t97\t98\t97
"""

TS_WIDE_SAMPLE = """Reference\tTest
98.1\t97.4
97.9\t97.8
98.4\t97.5
98.0\t97.9
98.3\t97.6
"""

TS_LONG_SAMPLE = """Group\tResponse
A\t12.1
A\t12.3
A\t11.9
B\t13.0
B\t12.8
B\t13.2
C\t11.6
C\t11.8
C\t11.7
"""

TWA_SAMPLE = """Formulation\tCondition\tResponse
A\t25C\t98.1
A\t25C\t97.9
A\t40C\t95.4
A\t40C\t95.1
B\t25C\t99.2
B\t25C\t99.0
B\t40C\t96.8
B\t40C\t96.5
C\t25C\t97.4
C\t25C\t97.7
C\t40C\t94.2
C\t40C\t94.5
"""

TC_WIDE_SAMPLE = """Population_A\tPopulation_B
98.1\t97.4
97.9\t97.8
98.4\t97.5
98.0\t97.9
98.3\t97.6
97.7\t98.1
98.5\t97.3
"""

TC_LONG_SAMPLE = """Group\tResponse
A\t98.1
A\t97.9
A\t98.4
A\t98.0
A\t98.3
B\t97.4
B\t97.8
B\t97.5
B\t97.9
B\t97.6
"""

PCA_SAMPLE = """Sample\tBatch\tVar1\tVar2\tVar3\tVar4\tVar5
S1\tA\t10.2\t5.1\t101\t0.45\t8.2
S2\tA\t10.5\t5.3\t99\t0.52\t8.0
S3\tA\t9.9\t4.8\t102\t0.49\t8.4
S4\tB\t12.1\t6.1\t110\t0.72\t7.1
S5\tB\t11.8\t5.9\t108\t0.68\t7.3
S6\tB\t12.4\t6.3\t111\t0.75\t7.0
S7\tC\t8.7\t4.2\t95\t0.31\t9.1
S8\tC\t8.9\t4.4\t96\t0.34\t8.9
S9\tC\t8.5\t4.1\t94\t0.29\t9.3
"""

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

    @st.cache_data
    def parse_pasted_data(text):
        if not text.strip():
            return None
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
            except Exception:
                continue
        return None

    def get_numeric_columns(df):
        num_cols = []
        for col in df.columns:
            converted = pd.to_numeric(df[col].astype(str).str.replace("%", ""), errors="coerce")
            if converted.notna().mean() >= 0.7:
                num_cols.append(col)
        return num_cols

    st.title("📊 App 01 - Descriptive Statistics")
    st.markdown("Paste data from Excel, optionally select grouping columns, and get summary statistics.")

    data_input = st.text_area("Data (Paste with headers from Excel)", height=200)

    if data_input:
        df = parse_pasted_data(data_input)
        if df is not None and not df.empty:
            st.success(f"Loaded shape: {df.shape[0]} rows × {df.shape[1]} columns")
            with st.expander("Preview Loaded Data"):
                st.dataframe(df.head(10), use_container_width=True)

            numeric_cols = get_numeric_columns(df)
            all_cols = list(df.columns)

            st.markdown("### Configuration")
            col1, col2, col3 = st.columns(3)
            with col1:
                selected_vars = st.multiselect("Variables", options=numeric_cols, default=numeric_cols)
            with col2:
                group1 = st.selectbox("Group by 1", options=["(None)"] + all_cols)
            with col3:
                group2 = st.selectbox("Group by 2", options=["(None)"] + all_cols)

            decimals = st.slider("Decimals", min_value=1, max_value=8, value=3)

            if st.button("Run Descriptive Statistics", type="primary"):
                if not selected_vars:
                    st.error("Please select at least one numeric variable.")
                else:
                    for v in selected_vars:
                        df[v] = pd.to_numeric(df[v].astype(str).str.replace("%", ""), errors="coerce")

                    active_groups = [g for g in [group1, group2] if g != "(None)"]

                    def calc_stats(x):
                        return pd.Series({
                            "N": x.count(),
                            "Mean": x.mean(),
                            "Std. Dev": x.std(ddof=1),
                            "Min": x.min(),
                            "Median": x.median(),
                            "Max": x.max(),
                            "CV (%)": (x.std(ddof=1) / x.mean() * 100) if x.mean() != 0 else np.nan
                        })

                    st.markdown("### Results")
                    try:
                        if active_groups:
                            results = df.groupby(active_groups)[selected_vars].apply(lambda g: g.apply(calc_stats)).unstack(level=-1)
                            if len(selected_vars) == 1:
                                results.columns = results.columns.droplevel(0)
                        else:
                            results = df[selected_vars].apply(calc_stats).T
                        display_df(results.reset_index() if isinstance(results.index, pd.MultiIndex) else results, decimals)
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.error("Could not parse data.")

# ==========================================
# APP 02: SHELF LIFE ESTIMATOR
# ==========================================
elif app_selection == "02 - Shelf Life Estimator":
    st.title("📈 App 02 - Shelf Life Estimator")
    st.markdown("Paste **Time** and **Response** columns, set the specification, and estimate shelf life from fit/CI/PI crossing.")

    c1, c2 = st.columns([2, 1])
    with c1:
        data_input = st.text_area("Stability data (2 columns: Time and Response)", value=SHELF_SAMPLE_DATA, height=180)
    with c2:
        pred_input = st.text_area("Optional future X values", value=SHELF_SAMPLE_PRED, height=180)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        spec_side = st.selectbox("Specification side", ["lower", "upper"], format_func=lambda x: "Lower spec" if x == "lower" else "Upper spec")
    with c2:
        shelf_basis = st.selectbox("Shelf life based on", ["ci", "pi", "fit"], format_func=lambda x: {"ci": "Confidence bound", "pi": "Prediction bound", "fit": "Fit line"}[x])
    with c3:
        confidence = st.slider("Confidence", 0.80, 0.99, 0.95, 0.01)
    with c4:
        spec_limit = st.number_input("Specification value", value=90.0, step=0.1)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        spec_label = st.text_input("Specification label", value="Spec")
    with c2:
        xlabel = st.text_input("X label", value="Time")
    with c3:
        ylabel = st.text_input("Y label", value="Response")
    with c4:
        y_suffix = st.text_input("Y suffix", value="%")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        point_label = st.text_input("Point label", value="Data")
    with c2:
        show_ci = st.checkbox("Show CI band", value=True)
    with c3:
        show_pi = st.checkbox("Show PI band", value=False)
    with c4:
        x_min_text = st.text_input("X min", value="")
    with c5:
        x_max_text = st.text_input("X max", value="")

    plot_title = st.text_input("Plot title", value="")
    decimals = st.slider("Decimals", 1, 8, 4)

    if st.button("Run Shelf Life Analysis", type="primary"):
        try:
            data_df = sl_parse_xy_data(data_input)
            pred_x = sl_parse_x_values(pred_input)

            x_data_max = float(data_df["x"].max())
            x_future_max = float(np.max(pred_x)) if len(pred_x) > 0 else x_data_max

            x_min = safe_parse_float(x_min_text, default=min(0.0, float(data_df["x"].min())))
            x_max = safe_parse_float(x_max_text, default=max(x_data_max * 3, x_future_max * 1.15, x_data_max + 12))

            if x_max <= x_min:
                raise ValueError("X max must be greater than X min.")

            model = sl_fit_linear(data_df["x"], data_df["y"])
            grid_x = np.linspace(x_min, x_max, 500)
            grid_df = sl_predict(model, grid_x, confidence=confidence, one_sided=True)
            pred_df = sl_predict(model, pred_x, confidence=confidence, one_sided=True) if len(pred_x) > 0 else pd.DataFrame()

            bound_col = sl_get_bound_column(spec_side, shelf_basis)
            shelf_life = sl_find_crossing(grid_df["x"].to_numpy(), grid_df[bound_col].to_numpy(), spec_limit)

            summary_df = pd.DataFrame([{
                "n": len(data_df),
                "Intercept": model["intercept"],
                "Slope": model["slope"],
                "Residual_SD": model["s"],
                "Degrees_of_freedom": model["df"],
                "R_squared": model["r2"],
                "Shelf_life_estimate": shelf_life
            }])

            st.markdown("### Model Summary")
            display_df(summary_df, decimals)

            if shelf_life is None:
                st.warning("No crossing with the specification was found within the plotted X range.")
            else:
                st.success(f"Estimated shelf life: {shelf_life:.{decimals}f}")

            if not pred_df.empty:
                st.markdown("### Predictions at Requested X Values")
                display_df(pred_df, decimals)

            st.markdown("### Plot")
            render_plot(
                sl_plot,
                data_df=data_df,
                grid_df=grid_df,
                spec_side=spec_side,
                spec_limit=spec_limit,
                shelf_basis=shelf_basis,
                show_ci_band=show_ci,
                show_pi_band=show_pi,
                title=plot_title or "Shelf Life Estimator",
                xlabel=xlabel,
                ylabel=ylabel,
                point_label=point_label,
                y_suffix=y_suffix,
                spec_label=spec_label
            )
        except Exception as e:
            st.error(f"Error: {e}")

# ==========================================
# APP 03: DISSOLUTION COMPARISON
# ==========================================
elif app_selection == "03 - Dissolution Comparison (f2)":
    st.title("💊 App 03 - Dissolution Comparison (f2)")
    st.markdown("Compare reference and test dissolution profiles with FDA-style point selection and optional bootstrap confidence intervals.")

    c1, c2 = st.columns(2)
    with c1:
        ref_input = st.text_area("Reference profile", value=DISS_REF_SAMPLE, height=220)
    with c2:
        test_input = st.text_area("Test profile", value=DISS_TEST_SAMPLE, height=220)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        include_zero = st.checkbox("Include time zero", value=False)
    with c2:
        cutoff_mode = st.selectbox("Point selection", ["apply_85", "all"], format_func=lambda x: "FDA-style: stop after first point where both ≥ threshold" if x == "apply_85" else "Use all common timepoints")
    with c3:
        threshold = st.number_input("Threshold", value=85.0, step=0.5)
    with c4:
        decimals = st.slider("Decimals", 1, 6, 2)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        bootstrap = st.checkbox("Bootstrap f2 CI", value=False)
    with c2:
        boot_method = st.selectbox("Bootstrap CI method", ["both", "percentile", "bca"])
    with c3:
        boot_conf = st.slider("Bootstrap confidence", 0.80, 0.99, 0.90, 0.01)
    with c4:
        boot_n = st.number_input("Resamples", min_value=100, value=2000, step=100)
    with c5:
        boot_seed = st.number_input("Seed", value=123, step=1)

    c1, c2, c3 = st.columns(3)
    with c1:
        show_units = st.checkbox("Show unit traces", value=True)
    with c2:
        show_boot_plot = st.checkbox("Show bootstrap plot", value=True)
    with c3:
        ylabel = st.text_input("Y label", value="% Dissolved")
    plot_title = st.text_input("Plot title", value="Dissolution Profiles")

    if st.button("Run Dissolution Comparison", type="primary"):
        try:
            ref_df = dis_parse_profile_table(ref_input)
            test_df = dis_parse_profile_table(test_input)

            ref_summary = dis_profile_summary(ref_df)
            test_summary = dis_profile_summary(test_df)
            merged = dis_merge_profiles(ref_summary, test_summary)
            selected, first_both_ge_idx = dis_select_points(
                merged=merged,
                include_zero=include_zero,
                cutoff_mode=cutoff_mode,
                threshold=threshold
            )

            if len(selected) < 3:
                raise ValueError("At least 3 selected timepoints are required to calculate f2.")

            selected = selected.copy()
            selected["abs_diff"] = (selected["mean_ref"] - selected["mean_test"]).abs()
            selected["sq_diff"] = (selected["mean_ref"] - selected["mean_test"]) ** 2

            f2_value = dis_calc_f2(selected["mean_ref"], selected["mean_test"])
            fda_tbl, fda_detail_tbl, conventional_ok = dis_fda_checks(
                ref_df=ref_df,
                test_df=test_df,
                merged=merged,
                selected=selected,
                threshold=threshold,
                include_zero=include_zero
            )

            st.markdown("### Result")
            result_df = pd.DataFrame([{
                "f2": f2_value,
                "Conventional_similarity_(f2≥50)": "Yes" if f2_value >= 50 else "No",
                "Conventional_FDA_checks_pass": "Yes" if conventional_ok else "No",
                "Selected_timepoints": len(selected)
            }])
            display_df(result_df, decimals)

            t1, t2, t3, t4 = st.tabs(["Selected Points", "Merged Summary", "FDA Checks", "Plots"])
            with t1:
                display_df(selected, decimals)
            with t2:
                display_df(merged, decimals)
            with t3:
                st.markdown("**FDA / conventional checks**")
                display_df(fda_tbl, decimals)
                st.markdown("**Details**")
                display_df(fda_detail_tbl, decimals)
            with t4:
                render_plot(
                    dis_plot_profiles,
                    ref_df=ref_df,
                    test_df=test_df,
                    ref_summary=ref_summary,
                    test_summary=test_summary,
                    selected=selected,
                    show_units=show_units,
                    title=plot_title,
                    ylabel=ylabel
                )

            if bootstrap:
                ref_mat, _ = dis_get_selected_matrix(ref_df, selected["Time"])
                test_mat, _ = dis_get_selected_matrix(test_df, selected["Time"])
                boot_vals = dis_bootstrap_f2(ref_mat, test_mat, n_boot=int(boot_n), seed=int(boot_seed))
                boot_rows = []

                ci_low = ci_high = None
                ci_label = f"{int(round(boot_conf*100))}% CI"

                if boot_method in ["percentile", "both"]:
                    p_low, p_high = dis_percentile_interval(boot_vals, conf=boot_conf)
                    boot_rows.append({"Method": "Percentile", "CI_low": p_low, "CI_high": p_high})
                    if boot_method == "percentile":
                        ci_low, ci_high = p_low, p_high

                if boot_method in ["bca", "both"]:
                    jack_vals = dis_jackknife_f2(ref_mat, test_mat)
                    b_low, b_high, z0, a = dis_bca_interval(f2_value, boot_vals, jack_vals, conf=boot_conf)
                    boot_rows.append({"Method": "BCa", "CI_low": b_low, "CI_high": b_high, "z0": z0, "acceleration": a})
                    if boot_method == "bca":
                        ci_low, ci_high = b_low, b_high

                if boot_method == "both" and boot_rows:
                    ci_low, ci_high = boot_rows[0]["CI_low"], boot_rows[0]["CI_high"]

                st.markdown("### Bootstrap Confidence Intervals")
                display_df(pd.DataFrame(boot_rows), decimals)

                if show_boot_plot:
                    render_plot(
                        dis_plot_bootstrap_f2_distribution,
                        boot_vals=boot_vals,
                        observed_f2=f2_value,
                        ci_low=ci_low,
                        ci_high=ci_high,
                        ci_label=ci_label
                    )
        except Exception as e:
            st.error(f"Error: {e}")

# ==========================================
# APP 04: TWO-SAMPLE TESTS
# ==========================================
elif app_selection == "04 - Two-Sample Tests":
    st.title("⚖️ App 04 - Two-Sample Tests")
    st.markdown("Wide or long input. Automatic routing uses Anderson-Darling normality plus Levene variance checking.")

    c1, c2 = st.columns(2)
    with c1:
        input_format = st.selectbox("Input format", ["wide", "long"], format_func=lambda x: "Wide: one group per column" if x == "wide" else "Long: group + response columns")
    with c2:
        design = st.selectbox("Design", ["independent", "paired"], format_func=lambda x: "Independent groups" if x == "independent" else "Paired data")

    default_ts_data = TS_WIDE_SAMPLE if input_format == "wide" else TS_LONG_SAMPLE
    data_input = st.text_area("Data", value=default_ts_data, height=220)

    try:
        df_raw = ts_parse_raw_table(data_input)
        df_loaded = ts_promote_header_if_needed(df_raw)
        st.markdown("### Preview")
        st.dataframe(df_loaded.head(20), use_container_width=True)
    except Exception as e:
        df_loaded = None
        st.error(f"Could not parse data: {e}")

    group_col = response_col = None
    if df_loaded is not None and input_format == "long":
        guessed_group, guessed_response = ts_guess_long_columns(df_loaded)
        c1, c2 = st.columns(2)
        with c1:
            group_col = st.selectbox("Group column", options=list(df_loaded.columns), index=list(df_loaded.columns).index(guessed_group) if guessed_group in df_loaded.columns else 0)
        with c2:
            response_col = st.selectbox("Response column", options=list(df_loaded.columns), index=list(df_loaded.columns).index(guessed_response) if guessed_response in df_loaded.columns else min(1, len(df_loaded.columns)-1))

    c1, c2, c3 = st.columns(3)
    with c1:
        alpha = st.slider("Alpha", 0.001, 0.20, 0.05, 0.001)
    with c2:
        route_mode = st.selectbox("Route", ["auto", "parametric_equal", "welch", "nonparametric"], format_func=lambda x: {
            "auto": "Auto choose test",
            "parametric_equal": "Force equal-variance parametric",
            "welch": "Force Welch route",
            "nonparametric": "Force nonparametric",
        }[x])
    with c3:
        decimals = st.slider("Decimals", 1, 8, 4)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        plot_title = st.text_input("Plot title", value="Group Comparison")
    with c2:
        ylabel = st.text_input("Y label", value="Response")
    with c3:
        show_plot = st.checkbox("Show plot", value=True)
    with c4:
        show_points = st.checkbox("Show raw points", value=True)

    if st.button("Run Comparison", type="primary"):
        try:
            if df_loaded is None:
                raise ValueError("Please provide a valid table.")

            paired_data = None
            if input_format == "wide":
                if design == "paired":
                    long_df, groups, name1, name2, x1, x2 = ts_prepare_wide_paired(df_loaded)
                    paired_data = (name1, name2, x1, x2)
                else:
                    long_df, groups = ts_prepare_wide(df_loaded)
            else:
                if design == "paired":
                    raise ValueError("Paired mode is supported here only for wide input with exactly two columns.")
                long_df, groups = ts_prepare_long(df_loaded, group_col, response_col)

            summary_tbl = ts_group_summary(long_df)
            selected_test, norm_tbl, lev_stat, lev_p = ts_choose_test(
                groups=groups,
                alpha=alpha,
                route_mode=route_mode,
                design=design,
                paired_data=paired_data
            )
            result = ts_run_test(groups, selected_test, paired_data=paired_data)

            st.markdown("### Test Result")
            result_df = pd.DataFrame([result])
            display_df(result_df, decimals)

            if "pvalue" in result and pd.notna(result["pvalue"]):
                if float(result["pvalue"]) < alpha:
                    st.warning(f"Result is significant at alpha = {alpha:.3f}.")
                else:
                    st.success(f"Result is not significant at alpha = {alpha:.3f}.")

            t1, t2, t3 = st.tabs(["Group Summary", "Assumption Checks", "Plot"])
            with t1:
                display_df(summary_tbl, decimals)
            with t2:
                st.markdown("**Normality**")
                display_df(norm_tbl, decimals)
                if design != "paired":
                    lev_df = pd.DataFrame([{
                        "Levene_stat": lev_stat,
                        "Levene_pvalue": lev_p,
                        "Equal_variance_at_alpha": "Yes" if pd.notna(lev_p) and lev_p >= alpha else "No"
                    }])
                    st.markdown("**Variance check**")
                    display_df(lev_df, decimals)
            with t3:
                if show_plot:
                    render_plot(
                        ts_plot_groups,
                        long_df=long_df,
                        title=plot_title,
                        ylabel=ylabel,
                        show_points=show_points,
                        design=design
                    )
        except Exception as e:
            st.error(f"Error: {e}")

# ==========================================
# APP 05: TWO-WAY ANOVA
# ==========================================
elif app_selection == "05 - Two-Way ANOVA":
    st.title("📐 App 05 - Two-Way ANOVA")
    st.markdown("Paste long-format data, choose one response plus two factors, and get ANOVA, means, counts, and an interaction plot.")

    data_input = st.text_area("Data", value=TWA_SAMPLE, height=220)

    try:
        df_raw = twa_parse_pasted_table(data_input)
        df_loaded = twa_clean_dataframe(df_raw)
        guessed_response, guessed_a, guessed_b = twa_guess_columns(df_loaded)
        st.markdown("### Preview")
        st.dataframe(df_loaded.head(20), use_container_width=True)
    except Exception as e:
        df_loaded = None
        st.error(f"Could not parse data: {e}")

    if df_loaded is not None:
        cols = list(df_loaded.columns)
        c1, c2, c3 = st.columns(3)
        with c1:
            response_col = st.selectbox("Response", cols, index=cols.index(guessed_response) if guessed_response in cols else 0)
        with c2:
            factor_a_col = st.selectbox("Factor A", cols, index=cols.index(guessed_a) if guessed_a in cols else 0)
        with c3:
            factor_b_col = st.selectbox("Factor B", cols, index=cols.index(guessed_b) if guessed_b in cols else min(1, len(cols)-1))
    else:
        response_col = factor_a_col = factor_b_col = None

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        include_interaction = st.checkbox("Include interaction", value=True)
    with c2:
        anova_type = st.selectbox("ANOVA type", [2, 3], format_func=lambda x: f"Type {x} SS")
    with c3:
        show_plot = st.checkbox("Show interaction plot", value=True)
    with c4:
        show_points = st.checkbox("Show raw points", value=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        plot_title = st.text_input("Plot title", value="Interaction Plot")
    with c2:
        ylabel = st.text_input("Y label", value="Response")
    with c3:
        decimals = st.slider("Decimals", 1, 8, 4)

    if st.button("Run Two-Way ANOVA", type="primary"):
        try:
            if df_loaded is None:
                raise ValueError("Please provide a valid table.")

            if len({response_col, factor_a_col, factor_b_col}) < 3:
                raise ValueError("Response, Factor A, and Factor B must be different columns.")

            df_analysis = twa_prepare_analysis_df(df_loaded, response_col, factor_a_col, factor_b_col)
            model, anova_tbl = twa_fit_model(df_analysis, include_interaction=include_interaction, typ=int(anova_type))
            summary_tbl = twa_make_group_summary(df_analysis)
            means_tbl = twa_make_means_table(df_analysis)
            counts_tbl = twa_make_counts_table(df_analysis)

            t1, t2, t3, t4 = st.tabs(["ANOVA", "Group Summary", "Means & Counts", "Plot"])
            with t1:
                display_df(anova_tbl, decimals)
            with t2:
                display_df(summary_tbl, decimals)
            with t3:
                st.markdown("**Means table**")
                display_df(means_tbl, decimals)
                st.markdown("**Counts table**")
                display_df(counts_tbl, decimals)
            with t4:
                if show_plot:
                    render_plot(
                        twa_plot_interaction,
                        df_analysis=df_analysis,
                        ylabel=ylabel,
                        title=plot_title,
                        show_points=show_points
                    )
        except Exception as e:
            st.error(f"Error: {e}")

# ==========================================
# APP 06: TOLERANCE & CONFIDENCE INTERVALS
# ==========================================
elif app_selection == "06 - Tolerance & Confidence Intervals":
    st.title("🎯 App 06 - Tolerance & Confidence Intervals")
    st.markdown("Two groups only. Computes mean confidence intervals, a difference-in-means confidence interval, and normal-based tolerance intervals.")

    input_format = st.selectbox("Input format", ["wide", "long"], format_func=lambda x: "Wide: two columns" if x == "wide" else "Long: group + response")
    default_tc_data = TC_WIDE_SAMPLE if input_format == "wide" else TC_LONG_SAMPLE
    data_input = st.text_area("Data", value=default_tc_data, height=220)

    try:
        df_raw = tc_parse_raw_table(data_input)
        df_loaded = tc_promote_header_if_needed(df_raw)
        st.markdown("### Preview")
        st.dataframe(df_loaded.head(20), use_container_width=True)
    except Exception as e:
        df_loaded = None
        st.error(f"Could not parse data: {e}")

    group_col = response_col = None
    if df_loaded is not None and input_format == "long":
        guessed_group, guessed_response = tc_guess_long_columns(df_loaded)
        c1, c2 = st.columns(2)
        with c1:
            group_col = st.selectbox("Group column", options=list(df_loaded.columns), index=list(df_loaded.columns).index(guessed_group) if guessed_group in df_loaded.columns else 0)
        with c2:
            response_col = st.selectbox("Response column", options=list(df_loaded.columns), index=list(df_loaded.columns).index(guessed_response) if guessed_response in df_loaded.columns else min(1, len(df_loaded.columns)-1))

    c1, c2, c3 = st.columns(3)
    with c1:
        ci_conf = st.slider("CI confidence", 0.80, 0.99, 0.95, 0.01)
    with c2:
        diff_method = st.selectbox("Difference CI method", ["auto", "pooled", "welch"], format_func=lambda x: {
            "auto": "Auto (Levene -> pooled/Welch)",
            "pooled": "Pooled",
            "welch": "Welch"
        }[x])
    with c3:
        alpha_chk = st.slider("Alpha for checks", 0.001, 0.20, 0.05, 0.001)

    c1, c2, c3 = st.columns(3)
    with c1:
        tol_conf = st.slider("Tolerance confidence", 0.80, 0.99, 0.95, 0.01)
    with c2:
        tol_cov = st.slider("Tolerance coverage", 0.50, 0.999, 0.95, 0.005)
    with c3:
        tol_side = st.selectbox("Tolerance side", ["two-sided", "upper", "lower"], format_func=lambda x: {"two-sided": "Two-sided", "upper": "Upper", "lower": "Lower"}[x])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        plot_title = st.text_input("Plot title", value="Confidence and Tolerance Intervals by Group")
    with c2:
        ylabel = st.text_input("Y label", value="Response")
    with c3:
        show_plot = st.checkbox("Show plot", value=True)
    with c4:
        show_points = st.checkbox("Show raw points", value=True)

    decimals = st.slider("Decimals", 1, 8, 4)

    if st.button("Run Intervals", type="primary"):
        try:
            if df_loaded is None:
                raise ValueError("Please provide a valid table.")

            if input_format == "wide":
                long_df, groups = tc_prepare_wide(df_loaded)
            else:
                long_df, groups = tc_prepare_long(df_loaded, group_col, response_col)

            group_names = list(groups.keys())
            diff_res = tc_diff_mean_ci(
                groups[group_names[0]],
                groups[group_names[1]],
                conf=ci_conf,
                method=diff_method,
                alpha_var=alpha_chk
            )
            diff_res["Comparison"] = f"{group_names[0]} - {group_names[1]}"

            ad_tbl = tc_ad_table(groups, alpha_chk)
            summary_tbl = tc_group_summary(groups)
            ci_tbl = tc_ci_table(groups, conf=ci_conf)
            ti_tbl = tc_ti_table(groups, conf=tol_conf, coverage=tol_cov, side=tol_side)

            t1, t2, t3, t4 = st.tabs(["Group Summary", "Normality & Difference CI", "CI & TI Tables", "Plot"])
            with t1:
                display_df(summary_tbl, decimals)
            with t2:
                st.markdown("**Normality**")
                display_df(ad_tbl, decimals)
                st.markdown("**Difference in means CI**")
                display_df(pd.DataFrame([diff_res]), decimals)
            with t3:
                st.markdown("**Mean confidence intervals**")
                display_df(ci_tbl, decimals)
                st.markdown("**Tolerance intervals**")
                display_df(ti_tbl, decimals)
            with t4:
                if show_plot:
                    render_plot(
                        tc_plot_intervals,
                        long_df=long_df,
                        ci_tbl=ci_tbl,
                        ti_tbl=ti_tbl,
                        title=plot_title,
                        ylabel=ylabel,
                        show_points=show_points
                    )
        except Exception as e:
            st.error(f"Error: {e}")

# ==========================================
# APP 07: PCA ANALYSIS
# ==========================================
elif app_selection == "07 - PCA Analysis":
    st.title("🌐 App 07 - PCA Analysis")
    st.markdown("Paste multivariate data, choose variables, and generate scores, loadings, scree, and cumulative variance plots.")

    data_input = st.text_area("Data", value=PCA_SAMPLE, height=240)

    try:
        df_loaded = pca_parse_table(data_input)
        numeric_cols = pca_numeric_candidates(df_loaded)
        all_cols = list(df_loaded.columns)
        st.markdown("### Preview")
        st.dataframe(df_loaded.head(20), use_container_width=True)
    except Exception as e:
        df_loaded = None
        numeric_cols = []
        all_cols = []
        st.error(f"Could not parse data: {e}")

    if df_loaded is not None and numeric_cols:
        c1, c2, c3 = st.columns(3)
        with c1:
            feature_cols = st.multiselect("Variables", options=numeric_cols, default=numeric_cols)
        with c2:
            label_col = st.selectbox("Label column", options=["(None)"] + all_cols)
        with c3:
            group_col = st.selectbox("Group column", options=["(None)"] + all_cols)
    else:
        feature_cols = []
        label_col = "(None)"
        group_col = "(None)"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        prep_mode = st.selectbox("Preprocess", ["none", "center", "autoscale"], format_func=lambda x: {"none": "None", "center": "Mean-center", "autoscale": "Autoscale"}[x])
    with c2:
        max_pcs = max(2, min(10, len(feature_cols) if len(feature_cols) > 0 else 2))
        n_components = st.slider("PCs", 2, max_pcs, min(3, max_pcs))
    with c3:
        ellipse_mode = st.selectbox("Ellipse", ["none", "3sigma", "mahal_95", "mahal_99"], format_func=lambda x: {
            "none": "None", "3sigma": "3σ ellipse", "mahal_95": "95% Mahalanobis", "mahal_99": "99% Mahalanobis"
        }[x])
    with c4:
        ellipse_scope = st.selectbox("Ellipse scope", ["overall", "by_group"], format_func=lambda x: "Overall" if x == "overall" else "By group")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        show_labels = st.checkbox("Show labels", value=True)
    with c2:
        show_scree = st.checkbox("Show scree plot", value=True)
    with c3:
        show_cum = st.checkbox("Show cumulative plot", value=True)
    with c4:
        show_scores = st.checkbox("Show scores plot", value=True)
    with c5:
        show_loadings = st.checkbox("Show loadings plot", value=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        scores_title = st.text_input("Scores title", value="Scores Plot")
    with c2:
        loadings_title = st.text_input("Loadings title", value="Loadings Plot")
    with c3:
        decimals = st.slider("Decimals", 1, 8, 4)

    if st.button("Run PCA", type="primary"):
        try:
            if df_loaded is None:
                raise ValueError("Please provide a valid table.")
            if len(feature_cols) < 2:
                raise ValueError("Choose at least two numeric variables.")

            X, meta = pca_prepare_matrix(df_loaded, feature_cols, label_col=label_col, group_col=group_col)
            Xp, prep_info = pca_preprocess(X, prep_mode)

            n_components_eff = min(int(n_components), Xp.shape[0], Xp.shape[1])
            pca_model, scores_df, loadings_df, explained_df = pca_run(Xp, n_components=n_components_eff)

            scores_out = pd.concat([meta.reset_index(drop=True), scores_df.reset_index(drop=True)], axis=1)
            loadings_out = loadings_df.copy()
            loadings_out.insert(0, "Variable", list(X.columns))

            t1, t2, t3 = st.tabs(["PCA Tables", "Scores & Loadings", "Plots"])
            with t1:
                st.markdown("**Preprocessing summary**")
                display_df(prep_info, decimals)
                st.markdown("**Explained variance**")
                display_df(explained_df, decimals)
            with t2:
                st.markdown("**Scores**")
                display_df(scores_out, decimals)
                st.markdown("**Loadings**")
                display_df(loadings_out, decimals)
            with t3:
                if show_scree:
                    render_plot(pca_plot_scree, explained_df=explained_df, title="Scree Plot")
                if show_cum:
                    render_plot(pca_plot_cumulative, explained_df=explained_df, title="Cumulative Explained Variance")
                if show_scores and scores_df.shape[1] >= 2:
                    render_plot(
                        pca_plot_scores,
                        scores_df=scores_df,
                        meta_df=meta,
                        explained_df=explained_df,
                        show_labels=show_labels,
                        title=scores_title,
                        ellipse_mode=ellipse_mode,
                        ellipse_scope=ellipse_scope
                    )
                if show_loadings and loadings_df.shape[1] >= 2:
                    render_plot(
                        pca_plot_loadings,
                        loadings_df=loadings_df,
                        variable_names=list(X.columns),
                        explained_df=explained_df,
                        title=loadings_title
                    )
        except Exception as e:
            st.error(f"Error: {e}")


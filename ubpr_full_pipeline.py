import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cvxpy as cp
from linearmodels.panel import PanelOLS
from sklearn.preprocessing import StandardScaler

###############################################################################
# Bank Group Utility
###############################################################################
def get_bank_groups(use_placebo=False, seed=42):
    """
    Returns treated and control bank groups - either original or shuffled for placebo tests.

    Parameters:
    -----------
    use_placebo : bool
        If True, returns shuffled bank groups for placebo tests
        If False, returns the original bank groups
    seed : int
        Random seed for reproducibility of shuffling

    Returns:
    --------
    tuple: (treated_group, control_group)
    """
    # Original bank groups
    original_treated = [
        3284070,  # Ally Financial Inc.
        1394676,  # American Express Company
        30810,    # Discover Financial Services
        723112,   # Fifth Third Bancorp
        12311,    # Huntington Bancshares Inc.
        280110,   # KeyCorp
        501105,   # M&T Bank Corporation
        233031    # Regions Financial Corporation
    ]

    original_control = [
        480228,   # Bank of America
        112837,   # Capital One
        476810,   # Citigroup
        852218,   # JPMorgan Chase
        210434,   # Northern Trust
        541101,   # Bank of New York Mellon
        2182786,  # Goldman Sachs
        817824,   # PNC
        504713,   # U.S. Bancorp
        451965    # Wells Fargo
    ]

    if not use_placebo:
        return original_treated, original_control

    # For placebo test, create combined list and randomly reassign
    random.seed(seed)
    all_banks = original_treated + original_control
    random.shuffle(all_banks)

    # Keep same group sizes but randomly assigned
    new_treated = all_banks[:len(original_treated)]
    new_control = all_banks[len(original_treated):]

    return new_treated, new_control

###############################################################################
# Data Utility
###############################################################################

def load_full_feature_panel(data_folder, series_list):
    """
    Loads all relevant UBPR series into a single wide panel dataset.
    Returns a DataFrame with (bank_id, date) as index and all numeric columns merged.
    """
    full_dfs = []
    
    for code, name, ffiec_form, df_col_name in series_list:
        data_file = os.path.join(data_folder, f"{ffiec_form} Combined.csv")
        if not os.path.exists(data_file):
            print(f"[Warning] File not found: {data_file}")
            continue

        df = pd.read_csv(data_file, index_col=0, header=[0, 1])
        df.index = pd.to_datetime(df.index, format="%m/%d/%Y %I:%M:%S %p")
        df.columns = df.columns.droplevel(1)
        df = df[['ID RSSD', code]].rename(columns={
            'ID RSSD': 'bank_id',
            code: df_col_name
        })
        df['date'] = df.index
        full_dfs.append(df)

    # Merge all dataframes on index = (date, bank_id)
    wide_df = full_dfs[0]
    for other in full_dfs[1:]:
        wide_df = pd.merge(wide_df, other, on=['date', 'bank_id'], how='outer')

    wide_df.set_index(['bank_id', 'date'], inplace=True)
    return wide_df

def prepare_ml_sample(df, outcome_col, treatment_col, covariate_list):
    """
    Prepares cleaned and standardized ML sample for Causal Forest and DML.
    Ensures outcome, treatment, and covariates are aligned with no duplicates.

    Returns:
        df_clean: standardized dataframe
        covariates_final: list of covariate names
        outcome_scaler: StandardScaler fitted on outcome_col
    """

    print("\n[Prepare ML Sample]")

    protected_cols = ['treated', 'post', 'treated_post']
    covariates_to_standardize = [col for col in covariate_list if col not in protected_cols]

    # Full list of required columns
    required_cols = [outcome_col, treatment_col] + covariate_list

    # Check missing columns
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in df: {missing}")

    # Subset and deduplicate
    df_subset = df[required_cols].copy()
    df_subset = df_subset.loc[:, ~df_subset.columns.duplicated()]
    print(f"  → Columns after deduplication: {df_subset.columns.tolist()}")

    # Drop missing
    df_clean = df_subset.dropna()
    print(f"  → Rows after dropping NAs: {df_clean.shape[0]}")

    # Standardize covariates (exclude protected treatment indicators)
    if covariates_to_standardize:
        cov_scaler = StandardScaler()
        df_clean[covariates_to_standardize] = cov_scaler.fit_transform(df_clean[covariates_to_standardize])
        print(f"  → Standardized covariates: {covariates_to_standardize}")
    else:
        print(f"  → No covariates standardized.")

    # Standardize outcome separately
    outcome_scaler = StandardScaler()
    df_clean[outcome_col] = outcome_scaler.fit_transform(df_clean[[outcome_col]])
    print(f"  → Outcome {outcome_col} standardized separately.")

    return df_clean, covariate_list, outcome_scaler



###############################################################################
# Model Utility
###############################################################################

from causalml.inference.tree import CausalTreeRegressor
from doubleml import DoubleMLData, DoubleMLPLR
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

def estimate_causal_forest_effect(df, outcome_col, treatment_col, covariate_cols, outcome_scaler,
                                  n_trees=100, outdir=None, df_col_name=None, test_type=None):
    """
    Estimates ATE using a causal forest built from causalml's CausalTreeRegressor.

    The causal forest methodology follows the Wager and Athey (2018) paper
    "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests"

    Parameters:
        df: DataFrame with outcome, treatment, and covariates
        outcome_col: str
        treatment_col: str
        covariate_cols: list of str
        n_trees: int
        outdir: str, results directory
        df_col_name: str, short variable name
        test_type: str, 'real' or 'placebo'

    Returns:
        float: Average treatment effect (ATE)
    """

    print("\n[Estimate Causal Forest Effect]")

    # Extract variables safely
    y = df[[outcome_col]].values.ravel()
    t = df[[treatment_col]].values.ravel()
    X = df[covariate_cols].values

    print(f"  → Shapes: X={X.shape}, y={y.shape}, t={t.shape}")

    ates = []
    cate_matrix = np.zeros((len(y), n_trees))

    for i in range(n_trees):
        X_boot, y_boot, t_boot = resample(X, y, t, replace=True, random_state=i)

        if len(np.unique(t_boot)) < 2:
            print(f"  Skipping tree {i}: only one treatment group present.")
            continue

        tree = CausalTreeRegressor(random_state=i)
        tree.fit(X=X_boot, treatment=t_boot, y=y_boot)

        ate_i, _, _ = tree.estimate_ate(X=X, treatment=t, y=y)
        ates.append(ate_i)
        cate_matrix[:, i] = tree.predict(X=X)

    # Average ATE and CATEs (still standardized)
    avg_ate_std = np.mean(ates)
    avg_cates_std = np.mean(cate_matrix, axis=1)

    # Rescale to real-world units
    avg_ate_real = outcome_scaler.inverse_transform([[avg_ate_std]])[0][0]
    avg_cates_real = outcome_scaler.inverse_transform(avg_cates_std.reshape(-1, 1)).flatten()

    print(f"  → Causal Forest ATE (real-world units): {avg_ate_real:.4f}")

    # Save results
    if outdir and df_col_name and test_type:
        hte_df = df[covariate_cols].copy()
        hte_df['CATE'] = avg_cates_real
        output_path = os.path.join(outdir, f"cf_cate_{df_col_name}_{test_type}.csv")
        hte_df.to_csv(output_path, index=False)
        print(f"  → Saved CATE estimates to {output_path}")

        # Generate visualizations
        generate_cate_visualizations(hte_df, outdir, df_col_name, test_type, model_tag="cf")

    return avg_ate_real


def estimate_dml_effect(df, outcome_col, treatment_col, covariate_cols, outcome_scaler,
                        outdir=None, df_col_name=None, test_type=None):
    """
    Estimates ATE and using DoubleML Partial Linear Regression. Nuisance functions are
    estimated with the random forest algorithm and the 'outcome' with a final linear stage.

    The double machine learning methodology follows the Chernozhukov (2018)
    paper "Double/debiased machine learning for treatment and structural parameters"

    Returns:
        float: estimated ATE
    """

    print("\n[Estimate DML Effect]")

    # Extract variables safely
    y = df[[outcome_col]].values.ravel()
    t = df[[treatment_col]].values.ravel()
    X = df[covariate_cols].values

    print(f"  → Shapes: X={X.shape}, y={y.shape}, t={t.shape}")

    # Prepare DoubleMLData
    dml_data = DoubleMLData.from_arrays(x=X, y=y, d=t)

    # Flexible ML models
    ml_l = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    ml_m = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    ml_g = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)

    # DML with Partial Linear Regression
    dml_model = DoubleMLPLR(
        obj_dml_data=dml_data,
        ml_l=ml_l,
        ml_m=ml_m,
        ml_g=ml_g,
        n_folds=5,
        score='IV-type'
    )

    dml_model.fit()

    # Extract ATE (standardized)
    ate_std = dml_model.coef[0]
    se_std = dml_model.se[0]
    print(f"  → DML ATE (standardized units): {ate_std:.4f} (SE: {se_std:.4f})")

    # Pseudo-CATEs (y-residuals / t-residuals)
    try:
        ml_l_hat = ml_l.fit(X, y).predict(X)
        ml_m_hat = ml_m.fit(X, t).predict(X)

        y_res = y - ml_l_hat
        t_res = t - ml_m_hat

        eps = 1e-8  # To avoid division by zero
        cate_pseudo_std = y_res / (t_res + eps)

        # Rescale ATE and CATEs to real-world units
        ate_real = outcome_scaler.inverse_transform([[ate_std]])[0][0]
        cate_pseudo_real = outcome_scaler.inverse_transform(cate_pseudo_std.reshape(-1, 1)).flatten()

        print(f"  → DML ATE (real-world units): {ate_real:.4f}")

        # Save results
        if outdir and df_col_name and test_type:
            hte_df = df[covariate_cols].copy()
            hte_df['CATE'] = cate_pseudo_real
            output_path = os.path.join(outdir, f"dml_cate_{df_col_name}_{test_type}.csv")
            hte_df.to_csv(output_path, index=False)
            print(f"  → Saved CATE estimates to {output_path}")

            # Generate visualizations
            generate_cate_visualizations(hte_df, outdir, df_col_name, test_type, model_tag="dml")

    except Exception as e:
        print(f"  Failed to compute/save DML pseudo-CATEs: {e}")

    return ate_real


###############################################################################
# Visualization Utility
###############################################################################

def generate_cate_visualizations(cate_df, outdir, df_col_name, test_type, model_tag="model"):
    """
    Generate diagnostic and publication-quality visuals of heterogeneous treatment effects.

    Parameters:
    - cate_df: pandas.DataFrame containing CATEs and covariates
    - outdir: str, directory to save the plots
    - df_col_name: str, name of the outcome variable
    - test_type: str, 'real' or 'placebo'
    - model_tag: str, short label to prefix files with (e.g., 'cf' for Causal Forest, 'dml' for Double ML)
    """

    prefix = f"{model_tag}_{df_col_name}_{test_type}"
    plot_prefix = os.path.join(outdir, prefix)

    # Histogram of CATEs
    plt.figure(figsize=(7, 4))
    sns.histplot(cate_df['CATE'], bins=30, kde=True, color="skyblue")
    plt.title("Distribution of CATEs")
    plt.xlabel("Estimated CATE")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{plot_prefix}_cate_histogram.png", dpi=300)
    plt.close()

    # Identify covariates to plot (exclude dummy/time indicators)
    exclude_cols = {'CATE', 'post', 'treated', 'treated_post', 'eff_q'}
    candidate_covs = [col for col in cate_df.columns if col not in exclude_cols]

    # Rank by absolute correlation
    top_covariates = sorted(
        candidate_covs,
        key=lambda c: abs(cate_df['CATE'].corr(cate_df[c].astype(float))),
        reverse=True
    )[:5]  # top 5 most informative

    # CATE vs. important covariates
    for col in top_covariates:
        plt.figure(figsize=(7, 4))
        sns.scatterplot(x=cate_df[col], y=cate_df['CATE'], alpha=0.6)
        sns.regplot(x=cate_df[col], y=cate_df['CATE'], scatter=False, color='red', ci=None)
        plt.title(f"CATE vs. {col}")
        plt.xlabel(col)
        plt.ylabel("Estimated CATE")
        plt.tight_layout()
        plt.savefig(f"{plot_prefix}_cate_vs_{col}.png", dpi=300)
        plt.close()

    # Boxplot by Efficiency Ratio Quartiles
    if 'efficiency_ratio' in cate_df.columns:
        cate_df['eff_q'] = pd.qcut(cate_df['efficiency_ratio'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        plt.figure(figsize=(7, 4))
        sns.boxplot(x='eff_q', y='CATE', data=cate_df, palette='pastel')
        plt.title("CATEs by Efficiency Ratio Quartile")
        plt.xlabel("Efficiency Ratio Quartile")
        plt.ylabel("Estimated CATE")
        plt.tight_layout()
        plt.savefig(f"{plot_prefix}_cate_by_efficiency_quartile.png", dpi=300)
        plt.close()

    print(f"  CATE plots saved with prefix: {prefix}")

    # LaTeX summary table of CATE stats
    summary_stats = {
        'Mean CATE': [cate_df['CATE'].mean()],
        'Std Dev CATE': [cate_df['CATE'].std()],
        'Min CATE': [cate_df['CATE'].min()],
        'Max CATE': [cate_df['CATE'].max()]
    }

    stats_df = pd.DataFrame(summary_stats).T.rename(columns={0: 'Value'})

    # Correlations
    corr_rows = []
    for col in candidate_covs:
        try:
            corr = cate_df['CATE'].corr(cate_df[col].astype(float))
            corr_rows.append((col, corr))
        except:
            continue
    corr_df = pd.DataFrame(corr_rows, columns=['Covariate', 'Corr. with CATE']).set_index('Covariate')
    corr_df = corr_df.loc[corr_df['Corr. with CATE'].abs().sort_values(ascending=False).index]

    # Combine both sections
    latex_df = pd.concat([stats_df, corr_df])

    latex_path = os.path.join(outdir, f"{prefix}_cate_summary.tex")
    with open(latex_path, "w") as f:
        f.write(latex_df.to_latex(float_format="%.4f", bold_rows=True))

    print(f"  LaTeX summary of CATE statistics saved to: {latex_path}")

###############################################################################
# Main Analysis Function
###############################################################################
def run_analysis(
    code, 
    series_name, 
    ffiec_form, 
    df_col_name,
    full_panel, 
    use_placebo=False,          # real or placebo?
    data_folder='data', 
    results_folder='results',
    start_date=pd.Timestamp('2012-09-30'),
    rollback_date=pd.Timestamp('2018-05-24'),
    dfast_2020=pd.Timestamp('2019-12-31')
):
    """
    Runs the DiD and Synthetic DiD analyses for a single UBPR series, 
    saving outputs to a dedicated subdirectory under `results_folder` -> <df_col_name> -> 'real' or 'placebo'.

    Parameters
    ----------
    code         : str, e.g. 'UBPRE630'
    series_name  : str, e.g. 'Return on Equity'
    ffiec_form   : str, e.g. 'FFIEC CDR UBPR Ratios Capital Analysis-a'
    df_col_name  : str, e.g. 'return_on_equity' (the name to use in the DataFrame)
    use_placebo  : bool, if True, treat vs. control are shuffled.
    data_folder  : str, folder where the CSV data files are stored
    results_folder: str, root folder to which results will be written
    start_date, rollback_date, dfast_2020 : Timestamps bounding the analysis

    Returns
    -------
    None (saves figures and regression outputs to disk)
    """

    # Output directory (separate for real vs. placebo)
    test_type = "placebo" if use_placebo else "real"
    outdir = os.path.join(results_folder, df_col_name, test_type)
    os.makedirs(outdir, exist_ok=True)

    print(f"\n=== {series_name} ({df_col_name}) | Test: {test_type.upper()} ===")

    #---------------------------------------------------------------------------
    # 1. LOAD THE DATA
    #---------------------------------------------------------------------------
    data_file = os.path.join(data_folder, f"{ffiec_form} Combined.csv")
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")

    print(f"Reading data from: {data_file}")
    cnip_df = pd.read_csv(
        data_file,
        index_col=0,
        header=[0,1]
    )

    # Convert the string index to datetime
    cnip_df.index = pd.to_datetime(
        cnip_df.index,
        format="%m/%d/%Y %I:%M:%S %p"
    )
    # Drop the second level in columns
    cnip_df.columns = cnip_df.columns.droplevel(1)

    #---------------------------------------------------------------------------
    # 2. GET TREATED vs CONTROL
    #---------------------------------------------------------------------------
    treated_banks, control_banks = get_bank_groups(use_placebo=use_placebo)

    # Subset
    subset_df = cnip_df[
        (cnip_df['ID RSSD'].isin(treated_banks + control_banks)) 
        & (cnip_df.index >= start_date)
        & (cnip_df.index <= dfast_2020)
    ].copy()

    # Rename columns
    subset_df.rename(columns={
        'ID RSSD': 'bank_id',
        code: df_col_name
    }, inplace=True)

    # Build MultiIndex (bank_id, date)
    subset_df['date'] = subset_df.index
    subset_df.set_index(['bank_id', 'date'], inplace=True)

    # Mark treated
    subset_df['treated'] = np.where(
        subset_df.index.get_level_values('bank_id').isin(treated_banks),
        1, 0
    )

    # Mark post
    subset_df['post'] = (
        subset_df.index.get_level_values('date') >= rollback_date
    ).astype(int)

    # Interaction
    subset_df['treated_post'] = subset_df['treated'] * subset_df['post']

    #---------------------------------------------------------------------------
    # 3. Quick Plot: Mean over Time for TREATED vs. CONTROL
    #---------------------------------------------------------------------------
    df_plot = subset_df.reset_index()  # bank_id, date, ...
    trend_df = df_plot.groupby(['date','treated'], as_index=False)[df_col_name].mean()

    plt.figure(figsize=(8,5))
    for key, grp in trend_df.groupby('treated'):
        label = 'Treated' if key == 1 else 'Control'
        plt.plot(grp['date'], grp[df_col_name], label=label)

    plt.axvline(rollback_date, color='red', linestyle='--', label='2018 Rollback')
    plt.title(f"Average {series_name}: Treated vs. Control ({test_type.capitalize()})")
    plt.xlabel("Date")
    plt.ylabel(series_name)
    plt.legend()

    # Save figure
    plot_did_path = os.path.join(outdir, f"did_plot_{df_col_name}_{test_type}.png")
    plt.savefig(plot_did_path, dpi=150, bbox_inches='tight')
    plt.close()

    #---------------------------------------------------------------------------
    # 4. Difference-in-Differences Regression
    #---------------------------------------------------------------------------
    did_model = PanelOLS.from_formula(
        formula=f'{df_col_name} ~ treated_post + EntityEffects + TimeEffects',
        data=subset_df
    )
    did_results = did_model.fit(cov_type='clustered', cluster_entity=True)

    # Save DiD summary
    did_summary_path = os.path.join(outdir, f"did_regression_summary_{df_col_name}_{test_type}.txt")
    with open(did_summary_path, 'w') as f:
        f.write(str(did_results.summary))

    #---------------------------------------------------------------------------
    # 5. Synthetic DiD
    #---------------------------------------------------------------------------
    # Split into pre and post
    df_pre = subset_df[subset_df['post'] == 0].copy()
    df_post = subset_df[subset_df['post'] == 1].copy()

    # TREATED Pre: average outcome
    treated_pre = df_pre[df_pre['treated'] == 1]
    treated_pre_mean = treated_pre.groupby('date')[df_col_name].mean()

    # CONTROL Pre: pivot to (time x bank)
    control_pre = df_pre[df_pre['treated'] == 0].reset_index()
    control_panel_pre = control_pre.pivot_table(
        index='date',
        columns='bank_id',
        values=df_col_name
    )
    # Align time index
    control_panel_pre = control_panel_pre.reindex(treated_pre_mean.index).dropna()

    # Solve QP to match treated pre-trend
    n_controls = control_panel_pre.shape[1]
    w = cp.Variable(n_controls, nonneg=True)
    objective = cp.Minimize(
        cp.sum_squares(control_panel_pre.values @ w - treated_pre_mean[control_panel_pre.index].values)
    )
    constraints = [cp.sum(w) == 1]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    w_opt = w.value

    # Synthetic pre
    syn_pre_vals = control_panel_pre.values @ w_opt
    syn_pre_series = pd.Series(syn_pre_vals, index=control_panel_pre.index, name='synthetic_pre')

    # Post
    treated_post_mean = df_post[df_post['treated'] == 1].groupby('date')[df_col_name].mean()
    control_post = df_post[df_post['treated'] == 0].reset_index()
    control_panel_post = control_post.pivot_table(
        index='date',
        columns='bank_id',
        values=df_col_name
    ).reindex(treated_post_mean.index).dropna()

    syn_post_vals = control_panel_post.values @ w_opt
    syn_post_series = pd.Series(syn_post_vals, index=control_panel_post.index, name='synthetic_post')

    # Plot TREATED vs. SYNTHETIC
    plt.figure(figsize=(8,5))
    # Pre
    plt.plot(treated_pre_mean.index, treated_pre_mean, label='Treated (Pre)', color='orange')
    plt.plot(syn_pre_series.index, syn_pre_series, label='Synthetic (Pre)', color='blue')
    # Post
    plt.plot(treated_post_mean.index, treated_post_mean, label='Treated (Post)', color='orange', linestyle='--')
    plt.plot(syn_post_series.index, syn_post_series, label='Synthetic (Post)', color='blue', linestyle='--')
    plt.axvline(rollback_date, color='red', linestyle='--', label='Rollback Date')

    plt.title(f"Synthetic DiD: {series_name} (Treated vs. Synthetic) ({test_type.capitalize()})")
    plt.ylabel(series_name)
    plt.legend()

    # Save Synthetic DiD plot
    plot_sdid_path = os.path.join(outdir, f"sdid_plot_{df_col_name}_{test_type}.png")
    plt.savefig(plot_sdid_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Build panel for regression
    treated_series = pd.concat([treated_pre_mean, treated_post_mean], axis=0).sort_index()
    synthetic_series = pd.concat([syn_pre_series, syn_post_series], axis=0).sort_index()

    df_sdid_list = []

    # Treated group
    df_treated = pd.DataFrame({
        'group_id': 1,
        'date': treated_series.index,
        'outcome': treated_series.values
    })
    df_sdid_list.append(df_treated)

    # Synthetic group
    df_synthetic = pd.DataFrame({
        'group_id': 0,
        'date': synthetic_series.index,
        'outcome': synthetic_series.values
    })
    df_sdid_list.append(df_synthetic)

    df_sdid = pd.concat(df_sdid_list, ignore_index=True)
    df_sdid = df_sdid.dropna(subset=['outcome']).sort_values(['group_id','date'])

    df_sdid['treated'] = df_sdid['group_id']
    df_sdid['post'] = (df_sdid['date'] >= rollback_date).astype(int)
    df_sdid['treated_post'] = df_sdid['treated'] * df_sdid['post']
    df_sdid.set_index(['group_id','date'], inplace=True)

    sdid_model = PanelOLS.from_formula(
        "outcome ~ treated_post + EntityEffects + TimeEffects",
        data=df_sdid
    )
    sdid_results = sdid_model.fit(cov_type='clustered', cluster_entity=True)

    # Save Synthetic DiD summary
    sdid_summary_path = os.path.join(outdir, f"sdid_regression_summary_{df_col_name}_{test_type}.txt")
    with open(sdid_summary_path, 'w') as f:
        f.write(str(sdid_results.summary))

    #---------------------------------------------------------------------------
    # 6. Causal Forest and Double ML Estimation
    #---------------------------------------------------------------------------

    # Full panel for HTE
    subset_reset = subset_df.reset_index()
    print(f" → Available columns in subset_reset: {subset_reset.columns.tolist()}")

    # Merge in covariates from full_panel
    df_flat = subset_reset.merge(
        full_panel.reset_index(), on=["bank_id", "date"], how="left")

    print(f"  → Merged df_flat shape: {df_flat.shape}")
    print(f"  → Available columns for ML: {df_flat.columns.tolist()}")

    # After merge, fix duplicate columns if they exist
    if f"{df_col_name}_x" in df_flat.columns and f"{df_col_name}_y" in df_flat.columns:
        print(f"  [Info] Resolving duplicate columns for {df_col_name}")
        df_flat[df_col_name] = df_flat[f"{df_col_name}_x"]
        df_flat.drop(columns=[f"{df_col_name}_x", f"{df_col_name}_y"], inplace=True)

    # Final check
    if df_col_name not in df_flat.columns:
        raise ValueError(f"[Error] Outcome column '{df_col_name}' missing after merge with full_panel.")

    # List of covariates you specified
    full_covariates = ['treated', 'post', 'treated_post', 'return_on_equity',
    'tier_1_leverage_capital_ratio', 'tier_1_risk_based_capital_ratio', 'total_risk_based_capital_ratio',
    'brokered_deposits_to_total_deposits', 'net_non_core_funding_dependence', 'core_deposits_as_of_total_assets',
    'short_term_investments_as_of_total_assets', '1_4_family_residential_loans', 'loans_to_individuals',
    'commercial_industrial_loans', 'non_interest_expense_average_assets', 'efficiency_ratio',
    'return_on_average_assets', 'net_interest_margin', 'noninterest_income_average_assets']

    # Prepare standardized dataset
    df_clean, covariate_cols, outcome_scaler = prepare_ml_sample(
        df=df_flat,
        outcome_col=df_col_name,
        treatment_col='treated',
        covariate_list=full_covariates)

    # Causal Forest
    cf_ate = estimate_causal_forest_effect(
        df_clean, outcome_col=df_col_name, treatment_col='treated', covariate_cols=covariate_cols,
        outcome_scaler=outcome_scaler, outdir=outdir, df_col_name=df_col_name, test_type=test_type)

    # DML (Partially-Linear Model)
    dml_ate = estimate_dml_effect(
        df_clean, outcome_col=df_col_name, treatment_col='treated', covariate_cols=covariate_cols,
        outcome_scaler=outcome_scaler, outdir=outdir, df_col_name=df_col_name, test_type=test_type)


    # Writing ates
    with open(os.path.join(outdir, f"ml_ate_estimates_{df_col_name}_{test_type}.txt"), 'w') as f:
        f.write(f"Causal Forest ATE: {cf_ate}\n")
        f.write(f"Double ML ATE: {dml_ate}\n")

    print(f"Done -> {series_name} ({df_col_name}), test={test_type} | results in {outdir}")


###############################################################################
# Main Driver
###############################################################################
def main():
    """
    For each UBPR series, run two analyses:
      1) Real (non-shuffled) analysis
      2) Placebo (shuffled) analysis
    Stores results in:
      results/<df_col_name>/real/
      results/<df_col_name>/placebo/
    """

    # You can also load the following from a CSV (e.g., `pd.read_csv('UBPR_Series.csv')`)
    all_series = [
        ("UBPRE630", "Return on Equity", 
         "FFIEC CDR UBPR Ratios Capital Analysis-a", "return_on_equity"),
        ("UBPR7204", "Tier 1 Leverage Capital Ratio", 
         "FFIEC CDR UBPR Ratios Concept Not In Presentation", "tier_1_leverage_capital_ratio"),
        ("UBPR7206", "Tier 1 Risk-Based Capital Ratio", 
         "FFIEC CDR UBPR Ratios Concept Not In Presentation", "tier_1_risk_based_capital_ratio"),
        ("UBPR7205", "Total Risk-Based Capital Ratio", 
         "FFIEC CDR UBPR Ratios Concept Not In Presentation", "total_risk_based_capital_ratio"),
        ("UBPRE595", "Brokered Deposits to Total Deposits", 
         "FFIEC CDR UBPR Ratios Liquidity and Funding", "brokered_deposits_to_total_deposits"),
        ("UBPRK447", "Net Non-Core Funding Dependence", 
         "FFIEC CDR UBPR Ratios Liquidity and Funding", "net_non_core_funding_dependence"),
        ("UBPRE591", "Core Deposits as % of Total Assets", 
         "FFIEC CDR UBPR Ratios Liquidity and Funding", "core_deposits_as_of_total_assets"),
        ("UBPRE589", "Short Term Investments as % of Total Assets", 
         "FFIEC CDR UBPR Ratios Liquidity and Inv Portfolio", "short_term_investments_as_of_total_assets"),
        ("UBPRE415", "1-4 Family Residential Loans", 
         "FFIEC CDR UBPR Ratios Allowance and Loan Mix-b", "1_4_family_residential_loans"),
        ("UBPRE424", "Loans to Individuals", 
         "FFIEC CDR UBPR Ratios Allowance and Loan Mix-b", "loans_to_individuals"),
        ("UBPRE423", "Commercial & Industrial Loans", 
         "FFIEC CDR UBPR Ratios Allowance and Loan Mix-b", "commercial_industrial_loans"),
        ("UBPRE005", "Non-Interest Expense / Average Assets", 
         "FFIEC CDR UBPR Ratios Executive Summary Report", "non_interest_expense_average_assets"),
        ("UBPRE088", "Efficiency Ratio", 
         "FFIEC CDR UBPR Ratios Noninterest Income and Expenses", "efficiency_ratio"),
        ("UBPRE013", "Return on Average Assets", 
         "FFIEC CDR UBPR Ratios Executive Summary Report", "return_on_average_assets"),
        ("UBPRE018", "Net Interest Margin", 
         "FFIEC CDR UBPR Ratios Summary Ratios", "net_interest_margin"),
        ("UBPRE004", "Noninterest Income / Average Assets", 
         "FFIEC CDR UBPR Ratios Executive Summary Report", "noninterest_income_average_assets"),
    ]

    # Wide panel for HTE estimation
    full_panel = load_full_feature_panel(data_folder='data', series_list=all_series)

    # Run real test, then placebo test for each series
    for code, name, ffiec_form, df_col_name in all_series:
        # 1) Real analysis
        run_analysis(code, name, ffiec_form, df_col_name, full_panel, use_placebo=False)
        # 2) Placebo analysis
        run_analysis(code, name, ffiec_form, df_col_name, full_panel, use_placebo=True)

if __name__ == "__main__":
    main()

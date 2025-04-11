import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp
from linearmodels.panel import PanelOLS

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
# Main Analysis Function
###############################################################################
def run_analysis(
    code, 
    series_name, 
    ffiec_form, 
    df_col_name, 
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

    # Run real test, then placebo test for each series
    for code, name, ffiec_form, df_col_name in all_series:
        # 1) Real analysis
        run_analysis(code, name, ffiec_form, df_col_name, use_placebo=False)
        # 2) Placebo analysis
        run_analysis(code, name, ffiec_form, df_col_name, use_placebo=True)

if __name__ == "__main__":
    main()

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium", auto_download=["ipynb", "html"])


@app.cell
def _(mo):
    mo.md(r"""
    **Background**

    In the experiment simulated by the data, an advertising promotion was tested to see if it would bring more customers to purchase a specific product priced at $10. Since it costs the company $0.15 to send out each promotion, it would be best to limit that promotion only to those that are most receptive to the promotion.

    **Task Part 1: Experiment Analysis**

    Analyze the results of the experiment and identify the effect of the Treatment on product purchase and Net Incremental Revenue; Demonstrate an end to end experiment analysis pipeline. Key question addressed:

    **1) Was the experiment valid?**
    **2) What did the experiment find? ATE and Confidence Intervals**

    **Data Description**

    A randomized experiment was conducted and the results are in Training.csv:

        Treatment – Indicates if the customer was part of treatment or control
        Purchase – Indicates if the customer purchased the product
        ID – Customer ID
        V1 to V7 – features of the customer

    **Cost of sending a Promotion: $0.15 Revenue from purchase of product: $10 (There is only one product)**

    Metrics:

    Incremental Response Rate (IRR)

    IRR depicts how many more customers purchased the product with the promotion, as compared to if they didn't receive the promotion. Mathematically, it's the ratio of the number of purchasers in the promotion group to the total number of customers in the purchasers group (treatment) minus the ratio of the number of purchasers in the non-promotional group to the total number of customers in the non-promotional group (control).


    Net Incremental Revenue (NIR)

    NIR depicts how much is made (or lost) by sending out the promotion. Mathematically, this is 10 times the total number of purchasers that received the promotion minus 0.15 times the number of promotions sent out, minus 10 times the number of purchasers who were not given the promotion.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Table of Contents:
    1) Executive Summary
    2) Config and Set Up
    3) SECTION 1a: Data Validity & Understanding
    4) SECTION 1b: Experiment Analysis
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Executive Summary

    Part 1a: The experiment was validated by running multiple checks that are a part of standard experimentation pipelines at scale. The checks involved included:
    1. Randomization Integrity Check - Passed. No user was assigned to more than one experimental group, confirming mutually exclusive assignment. This check supports internal validity of experiments.
    2. Baseline Balance Test - Passed. Using both visualization and actual statistical tests, I validate that the distribution of pre-treatment characteristics is similar across treatment arms. If this were not the case, this would be considered a randomization failure (bug) and affect the interpretation of tests. For e.g., what we see in the result of the A/B test may no longer only be dependent on the experience change but also pre-existing differences between control and challenger.
    4. Sample Ratio Mismatch Check - Passed. No evidence of sample ratio mismatch was detected. SRM is usually a deal breaker and considered a signal to stop further analysis. In practice, many practitioners don't trust the results of tests with evidence of sample ratio mismatch.
    5. MDL Check - A minimum detectable lift check was implemented to understand what is the minimum effect size the test was powered to detect. In an ideal case, we would plan to have sufficient sample size and estimate the no. of days the test would run prior to the test start. Since, we don't have access to this information, I have used the MDL checks to estimate whether I can derive meaningful conclusions from the lifts observed in the test.

    Part 1b: A scalable experiment pipeline was created to understand the impact of the promotion. Here are the key takeaways:
    1. The promotion drove a stastistically significant lift in conversion rate or net incremental response rate.
    2. However, since a small subset of users actually ended up purchasing more, the overall cost of promotion more than offset the goodness in conversion resulting a decline in net revenue.
    3. **This indicates that there is a need to tailor the promotion to a subset of users who may most benefit from the promotion which will also ensure that costs of promotion are optimized to maximize net incremental revenue.**
    """)
    return


@app.cell
def _():

    ## baseline standard library imports 
    import marimo as mo
    import pandas as pd 
    import numpy as np 
    import math 

    ## for visualization 
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick         
    import seaborn as sns

    ## for modeling tasks 
    from scipy import stats 
    import statsmodels.formula.api as smf

    ## for statistical tests 
    from scipy.stats import ks_2samp, chi2_contingency, chisquare, norm, bootstrap, ttest_ind
    from statsmodels.stats.proportion import proportions_ztest

    ## for pre-processing continuous and categorical data 
    from sklearn.preprocessing import StandardScaler, OneHotEncoder

    ## for model selection
    from sklearn.model_selection import (
        train_test_split,      # hold-out validation set
        StratifiedKFold,       # cross-fitting for DR-Learner; stratified preserves purchase rate
        cross_val_predict,     # OOF predictions in one call
    )

    ## for calibration 
    from sklearn.calibration import calibration_curve   # propensity model check

    # base models 
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.ensemble import (
        GradientBoostingClassifier,   
        GradientBoostingRegressor,  
        RandomForestClassifier,       
    )

    # for eval metrics 
    from sklearn.metrics import roc_auc_score, brier_score_loss

    return (
        bootstrap,
        chi2_contingency,
        chisquare,
        ks_2samp,
        math,
        mo,
        norm,
        np,
        pd,
        plt,
        proportions_ztest,
        ttest_ind,
    )


@app.cell
def _():
    # config parameters 
    revenue_per_purchase = 10 
    promotion_cost_treatment = 0.15
    promotion_cost_control = 0.0 
    random_state = 42 


    train_path    = "datasets/Training.csv"
    test_path    = "datasets/Test.csv"
    output_path   = "datasets/targeted_customers.csv"

    treatment_col = "Promotion"     
    outcome_col = "purchase"       
    id_col = "ID"
    return (
        id_col,
        outcome_col,
        promotion_cost_control,
        promotion_cost_treatment,
        revenue_per_purchase,
        treatment_col,
    )


@app.cell
def _(pd):
    # load the data 
    df_train = pd.read_csv("datasets/Training.csv")
    # df_test = pd.read_csv("datasets/Test.csv")
    return (df_train,)


@app.cell
def _(df_train):
    ## data validation: quick check on data types 
    df_train.info() 
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For the sake of simplicity, we assume that length of df_train is equal to the population on which this experience will be scaled. In real world settings, this may not be true and depends on the rate of experiment exposure.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    This shows that we have 84.5K observations for all columns. There are no missing values so missingness doesn't need to be handled but in production, this should always be checked.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### SECTION 1: Data Validity Checks & Understanding
    1. check_overlap(): do we see users as having multiple treatments?
    2. check_id_duplication(): do we see users with multiple ids?
    3. check_value_distribution() function to understand the spread of values across different data types
    4. check_covariate_distribution() function to understand if at a high level, covariates are balanced between control and treatment.It addresses questions such as are treatment and control distributions overlapping, with similar shape and similar proportions?
    5. check_covariate_balance() function to use KS test, SMD, TVD and chisquare (test of independence) and variance ratio tests.
    6. check_sample_ratio_mistmatch() function to implement chisquare (goodness of fit) and ensure that we can trust the results of the test and the test split is reliable
    7. check_minimum_detectable_lift() function to understand what lift the test is powered to detect. I am using this because I don't have any upfront information on what assumptions went into how long this promotional AB test was supposed to be run using standard sample size calculator.
    """)
    return


@app.cell
def _(pd):
    def check_overlap(
        df: pd.DataFrame,
        id_col: int,
        treatment_col: str
    ) -> dict:
        """
        Check if any user appears in both control and treatment group

        Returns dict
        """
        if df.empty:
            raise ValueError("Dataframe is empty.")

        if df[treatment_col].nunique() != 2:
            raise ValueError(f"Treatment column must have exactly 2 groups, got {df[treatment_col].nunique()}.") 


        treatments_per_user = df.groupby(id_col)[treatment_col].nunique()
        overlapping_users = treatments_per_user[treatments_per_user > 1]

        return {
            "n_overlapping_users": len(overlapping_users),
            "overlap_detected": len(overlapping_users) > 0,
            "verdict": "PASS" if len(overlapping_users) == 0 else "FAIL: Users assigned to multiple treatments",
            "offending_users": df[df[id_col].isin(overlapping_users.index)]
        }

    return (check_overlap,)


@app.cell
def _(check_overlap, df_train, id_col, treatment_col):
    check_overlap(df_train, id_col = id_col, treatment_col = treatment_col)
    return


@app.cell
def _(pd):
    def check_id_duplication(
        df: pd.DataFrame,
        id_col: int 
    ) -> dict:
        """
        Check if any user ID appears more than once in the dataframe.

        Returns dict
        """
        if df.empty:
            raise ValueError("Dataframe is empty.")

        duplicated_ids = df[df.duplicated(subset=[id_col], keep=False)]

        return {
            "n_duplicated_ids": duplicated_ids[id_col].nunique(),
            "duplication_detected": len(duplicated_ids) > 0,
            "verdict": "PASS" if len(duplicated_ids) == 0 else "FAIL: Duplicate user IDs detected",
            "offending_users": duplicated_ids
        }

    return (check_id_duplication,)


@app.cell
def _(check_id_duplication, df_train, id_col):
    check_id_duplication(df_train, id_col = id_col)
    return


@app.cell
def _(mo):
    mo.md(r"""
    These two checks show that:
    1) Spillover between test and control group is not a challenge.
    2) There are no duplicate ids so the grain of analysis is distinct at the user level. This is critical for AB tests since the results of this test will be used for user level targeting.
    """)
    return


@app.cell
def _(pd):
    def check_value_distribution(df: pd.DataFrame, id_var: int, cat_unique_threshold: int = 10) -> dict:
        """
        Compute distribution statistics for numeric and categorical columns,
        with missing values handled separately.
        """
        if df.empty:
            raise ValueError("Input dataframe is empty.")
        if id_var not in df.columns:
            raise ValueError(f"{id_var} not found in columns.")

        df = df.drop(columns=[id_var])

        # Missing values table
        missing_df = df.isna().sum().to_frame("missing_count")
        missing_df["missing_fraction"] = missing_df["missing_count"] / len(df)

        # Infer column types
        cat_cols, num_cols = [], []
        for col in df.columns:
            if df[col].dtype in ["object", "category"] or df[col].nunique() <= cat_unique_threshold:
                cat_cols.append(col)
            else:
                num_cols.append(col)

        # Categorical: value counts (long-format)
        cat_records = []
        for col in cat_cols:
            vc = df[col].value_counts(dropna=True)  # don't count missing here 
            for val, count in vc.items():
                cat_records.append((col, str(val), count / len(df), count))
        cat_df = pd.DataFrame(cat_records, columns=["column", "value", "frequency", "count"])

        # Numeric: describe() stats
        num_df = df[num_cols].describe().T if num_cols else pd.DataFrame()

        return {
            "categorical": cat_df,
            "numeric": num_df,
            "missing": missing_df
        }

    return (check_value_distribution,)


@app.cell
def _(check_value_distribution, df_train, id_col):
    check_value_distribution(df_train, id_var=id_col)
    return


@app.cell
def _(mo):
    mo.md(r"""
    1. The promotion column shows a 50.1% and 49.8% split. This should be tested further for sample ratio mismatch before interpreting results.
    2. The purchase column shows high class imbalance with ~98.7% of the values indicating there was no purchase. Ultimately, this means that only a very small subset of users ended up making a purchase. One hypothesis at this point is that the promotion may or may not benefit all user segments.
    4. V1 has four categories 0,1,2,3 with top two categories being 1 and 2; V4 has only two categories with 2 being the top category in volume terms; V5 has 4 categories, V6 has 4 categories and V7 has two categories. In general, all these categorical variables have decent spread of values.
    5. Numerical variables v2 and v3 have very different scales and will likely need to be normalized for modeling purposes. V3 also has negative values ranging from -1.69 to +1.69 indicating that it is likely some scaled score (e.g. Z score)
    """)
    return


@app.cell
def _(math, pd, plt):
    def check_covariate_distribution(
        df: pd.DataFrame,
        treatment_col: str,
        id_col: str,
        outcome_col: str,
        cat_threshold: int = 5,
        bins: int = 20
    ) -> None:
        """
        Plot distributions of all covariates split by treatment vs control.
        """
        if df.empty:
            raise ValueError("Dataframe is empty.")
        control, treat = (grp for _, grp in df.groupby(treatment_col))

        cols = df.drop(columns=[id_col, treatment_col, outcome_col]).columns
        cat_cols = [c for c in cols if df[c].nunique() <= cat_threshold]
        num_cols = [c for c in cols if c not in cat_cols]
        plot_cols = num_cols + cat_cols

        ncols = 3
        nrows = math.ceil(len(plot_cols) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
        axes = axes.flatten()

        for i, col in enumerate(plot_cols):
            ax = axes[i]
            if col in num_cols:
                ax.hist(control[col], bins=bins, alpha=0.5, label="Control", density=True)
                ax.hist(treat[col], bins=bins, alpha=0.5, label="Treatment", density=True)
                ax.set_ylabel("Density")
            else:
                plot_df = pd.DataFrame({
                    "Control": control[col].value_counts(normalize=True),
                    "Treatment": treat[col].value_counts(normalize=True)
                }).fillna(0)
                plot_df.plot(kind="bar", ax=ax)
                ax.set_ylabel("Proportion")
            ax.set_title(col)
            ax.legend()

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # plt.tight_layout()
        plt.show()

    return (check_covariate_distribution,)


@app.cell
def _(
    check_covariate_distribution,
    df_train,
    id_col,
    outcome_col,
    treatment_col,
):
    check_covariate_distribution(
        df_train,
        treatment_col=treatment_col,
        id_col=id_col,
        outcome_col=outcome_col 
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    Overall covariate distributions seem balanced. V2 and V3 show some differences in shape which can be further investigated by statistical tests.
    """)
    return


@app.cell
def _(chi2_contingency, ks_2samp, np, pd):
    ## check class imbalance with statistical tests 

    def check_covariate_balance(
        df: pd.DataFrame,
        treatment_col: str,
        id_col: str,
        outcome_col: str,
        cat_threshold: int = 5
    ) -> pd.DataFrame:
        """
        Evaluate balance between treatment and control groups for all covariates.

        Raises ValueError If dataframe is empty 
        """
        if df.empty:
            raise ValueError("Dataframe is empty.")

        control, treat = (grp for _, grp in df.groupby(treatment_col))

        cols = df.drop(columns=[id_col, treatment_col, outcome_col]).columns
        records = []

        for col in cols:
            is_cat = df[col].dtype == "object" or df[col].nunique() <= cat_threshold

            if is_cat:
                # effect size: total variation distance
                treat_dist = treat[col].value_counts(normalize=True)
                ctrl_dist = control[col].value_counts(normalize=True)
                tvd = (treat_dist - ctrl_dist).abs().sum() / 2  

                # statistical test: chi-square on counts
                contingency = pd.crosstab(df[col], df[treatment_col])
                chi2_p = chi2_contingency(contingency).pvalue

                records.append({
                    "variable": col,
                    "type": "categorical",
                    "smd": np.nan,
                    "ks_pvalue": np.nan,
                    "variance_ratio": np.nan,
                    "tvd": tvd,
                    "chi2_pvalue": chi2_p,
                    "imbalanced": (tvd > 0.1) or (chi2_p < 0.05)
                })

            else:
                t = treat[col].dropna()
                c = control[col].dropna()
                smd = (t.mean() - c.mean()) / np.sqrt((t.var() + c.var()) / 2)
                ks_p = ks_2samp(t, c).pvalue
                var_ratio = t.var() / c.var() if c.var() != 0 else np.nan

                records.append({
                    "variable": col,
                    "type": "numeric",
                    "smd": smd,
                    "ks_pvalue": ks_p,
                    "variance_ratio": var_ratio,
                    "tvd": np.nan,
                    "chi2_pvalue": np.nan,
                    "imbalanced": abs(smd) > 0.1 or ks_p < 0.05 or (not np.isnan(var_ratio) and not (0.5 <= var_ratio <= 2.0))
                                   })

        return pd.DataFrame(records).sort_values("imbalanced", ascending=False)

    return (check_covariate_balance,)


@app.cell
def _(check_covariate_balance, df_train, id_col, outcome_col, treatment_col):
    check_covariate_balance(df_train, treatment_col = treatment_col, id_col = id_col, outcome_col = outcome_col)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The learnings show the following:
    1. All variables except V3 are balanced
    2. V3 is not of much concern since absolute standardized mean difference is quite small (benchmark is < 0.1). Small KS p value is possible due to slight shape differences.
    3. Note on assumptions: ChiSquare Test of Proportions is fine to use because most categories have enough volume to avoid extreme outliers (e.g. cell count <5).
    """)
    return


@app.cell
def _(chisquare, pd):
    def check_sample_ratio_mismatch(
        df: pd.DataFrame,
        treatment_col: str,
        expected_ratio: float = 0.50,
        alpha: float = 0.01 
    ) -> dict:
        """
        Chi-square goodness-of-fit test for Sample Ratio Mismatch (SRM).

        Raises ValueError
        If dataframe is empty or treatment column does not have exactly 2 groups.
        """
        if df.empty:
            raise ValueError("Dataframe is empty.")
        if df[treatment_col].nunique() != 2:
            raise ValueError(f"Treatment column must have exactly 2 groups, got {df[treatment_col].nunique()}.") ## this is good to have and can be added consistently across all functions if analyzing a 50/50 AB test 

        groups = dict(tuple(df.groupby(treatment_col)))
        (control_label, control), (treatment_label, treat) = groups.items()

        n_control = len(control)
        n_treatment = len(treat)
        n_total = n_control + n_treatment

        observed = [n_treatment, n_control]
        expected = [n_total * expected_ratio, n_total * (1 - expected_ratio)]

        chi2_stat, p_value = chisquare(f_obs=observed, f_exp=expected)
        observed_ratio = n_treatment / n_total

        verdict = "PASS: No SRM detected" if p_value > alpha else "FAIL: SRM detected — experiment is invalid"

        return {
            "control_group": control_label,
            "treatment_group": treatment_label,
            "chi2": chi2_stat,
            "p_value": p_value,
            "observed_ratio": round(observed_ratio, 4),
            "expected_ratio": expected_ratio,
            "verdict": verdict
        }

    return (check_sample_ratio_mismatch,)


@app.cell
def _(check_sample_ratio_mismatch, df_train, treatment_col):
    check_sample_ratio_mismatch(df_train, treatment_col = treatment_col)
    return


@app.cell
def _(norm, pd):
    # MDL given the sample size

    def check_minimum_detectable_lift(
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        alpha: float = 0.05,
        power: float = 0.80
    ) -> dict[str, float]:
        """
        Compute the minimum detectable lift (MDL) given observed sample sizes
        and baseline conversion rate. Control group CVR is used as a proxy for baseline CVR 

        Parameters
        alpha : Significance level (0.05) 
        power : Desired statistical power (0.80)

        Returns dict

        Raises ValueError if dataframe is empty or treatment column has more than 2 values 
        """
        if df.empty:
            raise ValueError("Dataframe is empty.")
        if df[treatment_col].nunique() != 2:
            raise ValueError(f"Expected binary treatment column, got {df[treatment_col].nunique()} unique values.")

        groups = dict(tuple(df.groupby(treatment_col)))
        (control_label, control), (treatment_label, treat) = groups.items()

        n_control = len(control)
        n_treatment = len(treat)

        # baseline conversion rate from control group
        p_baseline = control[outcome_col].mean()

        # z-scores for alpha (two-tailed) and power
        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta  = norm.ppf(power)

        # minimum detectable effect (absolute)
        mde_absolute = (z_alpha + z_beta) * (
            (p_baseline * (1 - p_baseline) * (1 / n_control + 1 / n_treatment)) ** 0.5
        )

        # relative lift over baseline
        mde_relative = mde_absolute / p_baseline

        return {
            "control_group": control_label,
            "treatment_group": treatment_label,
            "n_control": n_control,
            "n_treatment": n_treatment,
            "baseline_rate": round(p_baseline, 4),
            "mde_absolute": round(mde_absolute, 4),
            "mde_relative_pct": round(mde_relative * 100, 4),
            "alpha": alpha,
            "power": power
        }

    return (check_minimum_detectable_lift,)


@app.cell
def _(check_minimum_detectable_lift, df_train, outcome_col, treatment_col):
    check_minimum_detectable_lift(df_train, treatment_col = treatment_col, outcome_col = outcome_col) 
    return


@app.cell
def _(mo):
    mo.md(r"""
    This shows that we can reliably estimate a 22% relative increase or a 0.17pp absolute increase. If the test lift is larger than this, than I can confidently proceed to using these estimates in further downstream analysis.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Section 1B: The experiment data is valid and we can now proceed to the analysis phase. In the analysis phase, the goal is to understand the impact of treatment on key metrics (incremental response rate and net incremental revenue) and statistically evaluate significance.

    Design Approach:
    1. Baseline metric computation functions for conversion rate and revenue per user.
    3. A 'MetricDefinition' to ensure metadata is built in case we want to scale multiple metrics eventually. It also includes the functions to calculate the metrics for control and challenger. Note that in this question, there is a difference in how revenue will be calculated for control vs challenger given different promotion costs.
    4. An 'AnalyzeExperiment' class to compute ATE, p-values, bootstrap confidence intervals and projected business impact based on test learnings.
    """)
    return


@app.function
## Metric Computation Functions: Conversion Rate as a precursor to Net Incremental Response Rate 

def conversion_rate(df, outcome_col: str) -> float:
    """
    Ratio of purchasers to total customers.
    Returns decimal 
    """
    if df.empty:
        raise ValueError("Empty dataframe")
    return df[outcome_col].mean()


@app.cell
def _(revenue_per_purchase):
    ## Metric Computation Functions: Revenue as a precursor to Net Incremental Revenue

    def net_revenue_per_user(df, outcome_col: str, promotion_cost: float) -> float:
        """
        Net revenue per user = (purchases × $10 - promotion_cost × n) / n
        """
        if df.empty:
            raise ValueError("Empty dataframe")
        total_revenue = (df[outcome_col] * revenue_per_purchase).sum()
        total_cost    = promotion_cost * len(df)
        return (total_revenue - total_cost) / len(df)

    return (net_revenue_per_user,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Metric Class:
    A design choice to help plan for scale and ensure that each function has it's own metadata with it.
    """)
    return


@app.class_definition
class MetricDefinition:
    """
    Defines a metric for experiment evaluation.
    unit dependent computations: 
        "rate": decimal (0-1), analytical CI, no cost needed
        "usd": dollar value, bootstrap CI, cost needed
    """
    def __init__(
        self,
        name: str,
        func,
        metric_type: str,
        unit: str,
        higher_is_better: bool = True,
        control_cost: float = None,
        treatment_cost: float = None
    ):
        if unit not in ("rate", "usd"):
            raise ValueError(f"unit must be 'rate' or 'usd', got '{unit}'")
        if unit == "usd" and (control_cost is None or treatment_cost is None):
            raise ValueError(f"USD metric '{name}' requires control_cost and treatment_cost")

        self.name             = name
        self.func             = func
        self.metric_type      = metric_type
        self.unit             = unit
        self.higher_is_better = higher_is_better
        self.control_cost     = control_cost
        self.treatment_cost   = treatment_cost

    def compute_control(self, df, outcome_col: str) -> float:
        """Compute metric for control group."""
        if self.unit == "usd":
            return self.func(df, outcome_col, self.control_cost)
        return self.func(df, outcome_col)

    def compute_treatment(self, df, outcome_col: str) -> float:
        """Compute metric for treatment group."""
        if self.unit == "usd":
            return self.func(df, outcome_col, self.treatment_cost)
        return self.func(df, outcome_col)


@app.cell
def _(net_revenue_per_user, promotion_cost_control, promotion_cost_treatment):
    metrics = [
        MetricDefinition(
            name="conversion_rate",
            func=conversion_rate,
            metric_type="primary",
            unit="rate",
            higher_is_better=True
            # no costs — conversion_rate doesn't need them
        ),
        MetricDefinition(
            name="net_revenue_per_user",
            func=net_revenue_per_user,
            metric_type="primary",
            unit="usd",
            higher_is_better=True,
            control_cost= promotion_cost_control,       # 0.0
            treatment_cost= promotion_cost_treatment    # 0.15
        ),
    ]
    return (metrics,)


@app.cell
def _(metrics):
    # check if this works as expected: can be removed from the production pipeline 
    # first metric (conversion_rate)
    print(metrics[0])                 
    print(metrics[0].name)             
    print(metrics[0].func)            
    print(metrics[0].metric_type)  
    print(metrics[0].unit)
    print(metrics[0].control_cost)    
    print(metrics[0].higher_is_better) 
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Analyze Experiment Class
    This class is built to ensure that all metrics are computed for control and challenger, ATE is estimated, p values and confidence intervals defined and finally business impact is projected.
    """)
    return


@app.cell
def _(
    bootstrap,
    np,
    pd,
    promotion_cost_control,
    promotion_cost_treatment,
    proportions_ztest,
    revenue_per_purchase,
    ttest_ind,
):
    class AnalyzeExperiment:
        """
        Evaluates the experiment.

        Methods:
            compute_metrics()                    → point estimates
            compute_statistics()                 → CI + p-values
            compute_nir()                        → actual dollar impact in experiment
            project_impact(statistics, pop_size) → estimated impact at future scale
        """

        def __init__(
            self,
            df,
            treatment_col: str,
            outcome_col: str,
            metrics: list,
            control_value,
            treatment_value
        ):
            # --- validation ---
            if df.empty:
                raise ValueError("Dataframe is empty")
            if treatment_col not in df.columns:
                raise ValueError(f"Treatment column '{treatment_col}' not found")
            if outcome_col not in df.columns:
                raise ValueError(f"Outcome column '{outcome_col}' not found")

            unique_values = set(df[treatment_col].unique())
            if control_value not in unique_values or treatment_value not in unique_values:
                raise ValueError(f"Control/treatment values not found in {unique_values}")

            if df[outcome_col].nunique() > 2:
                print(f"Warning: '{outcome_col}' has {df[outcome_col].nunique()} unique values — expected binary (0/1)")

            # --- store ---
            self.df = df
            self.treatment_col = treatment_col
            self.outcome_col = outcome_col
            self.metrics = metrics
            self.control_value = control_value
            self.treatment_value = treatment_value

            # --- split groups ---
            self.control = df[df[treatment_col] == control_value]
            self.treatment = df[df[treatment_col] == treatment_value]

            if len(self.control) == 0 or len(self.treatment) == 0:
                print("Warning: One of the groups has zero rows. Tests may be NaN.")

        # --------------------------------------------------------
        # POINT ESTIMATES
        # --------------------------------------------------------
        def compute_metrics(self) -> pd.DataFrame:
            records = []

            for metric in self.metrics:
                ctrl_val = metric.compute_control(self.control, self.outcome_col)
                trt_val = metric.compute_treatment(self.treatment, self.outcome_col)
                ate = trt_val - ctrl_val
                rel_lift = (ate / ctrl_val * 100) if ctrl_val != 0 else np.nan

                records.append({
                    "metric": metric.name,
                    "type": metric.metric_type,
                    "unit": metric.unit,
                    "control_group": self.control_value,
                    "treatment_group": self.treatment_value,
                    "control": round(ctrl_val, 4),
                    "treatment": round(trt_val, 4),
                    "ate": round(ate, 4),
                    "relative_lift_pct": round(rel_lift, 2) if not np.isnan(rel_lift) else np.nan
                })

            return pd.DataFrame(records)

        # --------------------------------------------------------
        # STATISTICS WITH TESTS + BOOTSTRAP
        # --------------------------------------------------------
        def compute_statistics(self, alpha: float = 0.05, n_bootstrap: int = 500) -> pd.DataFrame:
            records = []

            for metric in self.metrics:
                # --- per-user arrays ---
                if metric.unit == "rate":
                    ctrl_values = self.control[self.outcome_col].values
                    trt_values = self.treatment[self.outcome_col].values
                else:
                    ctrl_values = (self.control[self.outcome_col] * revenue_per_purchase - promotion_cost_control).values
                    trt_values = (self.treatment[self.outcome_col] * revenue_per_purchase - promotion_cost_treatment).values

                # --- point estimate ---
                ctrl_mean = ctrl_values.mean() if len(ctrl_values) > 0 else np.nan
                trt_mean = trt_values.mean() if len(trt_values) > 0 else np.nan
                ate = trt_mean - ctrl_mean

                # --- parametric test ---
                if metric.unit == "rate":
                    n_ctrl_success = int(ctrl_values.sum()) if len(ctrl_values) > 0 else 0
                    n_trt_success = int(trt_values.sum()) if len(trt_values) > 0 else 0
                    n_ctrl = len(ctrl_values)
                    n_trt = len(trt_values)

                    if n_ctrl > 0 and n_trt > 0:
                        stat, p_value = proportions_ztest(
                            count=[n_trt_success, n_ctrl_success],
                            nobs=[n_trt, n_ctrl]
                        )
                    else:
                        stat, p_value = np.nan, np.nan
                else:
                    if len(ctrl_values) > 1 and len(trt_values) > 1:
                        stat, p_value = ttest_ind(trt_values, ctrl_values, equal_var=False)
                    else:
                        stat, p_value = np.nan, np.nan

                # --- bootstrap CI ---
                def ate_statistic(ctrl_sample, trt_sample):
                    return trt_sample.mean() - ctrl_sample.mean()

                try:
                    result = bootstrap(
                        data=(ctrl_values, trt_values),
                        statistic=ate_statistic,
                        n_resamples=n_bootstrap,
                        method="percentile",
                        confidence_level=1 - alpha,
                        paired=False,
                        random_state=42
                    )
                    ci_lower = result.confidence_interval.low
                    ci_upper = result.confidence_interval.high
                except Exception:
                    ci_lower, ci_upper = np.nan, np.nan

                records.append({
                    "metric": metric.name,
                    "unit": metric.unit,
                    "ate": round(ate, 4) if not np.isnan(ate) else np.nan,
                    "ci_lower": round(ci_lower, 4) if ci_lower is not None else np.nan,
                    "ci_upper": round(ci_upper, 4) if ci_upper is not None else np.nan,
                    "p_value": max(round(p_value, 4), 1e-300) if p_value is not None else np.nan,
                    "method": f"{'z-test' if metric.unit=='rate' else 't-test'} + bootstrap CI"
                })

            return pd.DataFrame(records)

        # --------------------------------------------------------
        # BUSINESS IMPACT
        # --------------------------------------------------------
        def compute_nir(self) -> pd.DataFrame:
            purchases_treatment = self.treatment[self.outcome_col].sum()
            purchases_control = self.control[self.outcome_col].sum()
            n_treatment = len(self.treatment)

            treatment_revenue = (revenue_per_purchase * purchases_treatment) - (promotion_cost_treatment * n_treatment)
            control_revenue = revenue_per_purchase * purchases_control
            nir = treatment_revenue - control_revenue

            return pd.DataFrame([{
                "purchases_treatment": int(purchases_treatment),
                "purchases_control": int(purchases_control),
                "n_treatment": n_treatment,
                "treatment_revenue": round(treatment_revenue, 2),
                "control_revenue": round(control_revenue, 2),
                "nir": round(nir, 2)
            }])

        # --------------------------------------------------------
        # PROJECTION
        # --------------------------------------------------------
        def project_impact(self, statistics: pd.DataFrame, population_size: int) -> pd.DataFrame:
            """
            Estimate dollar impact if promotion rolled out to N users.
            Uses the 'ate' from compute_metrics as the single source of truth.
            """
            if population_size <= 0:
                raise ValueError("Population size must be positive")

            # Copy statistics to avoid modifying original
            results = statistics.copy()

            # Ensure 'ate' exists; if not, merge from compute_metrics
            if "ate" not in results.columns:
                metric_results = self.compute_metrics()[["metric", "ate"]]
                results = results.merge(metric_results, on="metric", suffixes=("", "_metric"))
                if "ate_metric" in results.columns:
                    results["ate"] = results["ate_metric"]

            # Only USD metrics are projected
            is_usd = results["unit"] == "usd"

            # Scale ate and CI by population size
            results["population_size"] = population_size
            results["projected_impact"] = np.where(is_usd, results["ate"] * population_size, np.nan)
            results["projected_ci_lower"] = np.where(is_usd, results["ci_lower"] * population_size, np.nan)
            results["projected_ci_upper"] = np.where(is_usd, results["ci_upper"] * population_size, np.nan)

            return results[results["unit"] == "usd"]

    return (AnalyzeExperiment,)


@app.cell
def _(AnalyzeExperiment, df_train, metrics):
    # ============================================================
    # 7. EXECUTION
    # ============================================================

    evaluator = AnalyzeExperiment(
        df=df_train,
        treatment_col="Promotion",
        outcome_col="purchase",
        metrics=metrics,
        control_value="No",
        treatment_value="Yes"
    )
    return (evaluator,)


@app.cell
def _(evaluator):
    # point estimates
    metric_results = evaluator.compute_metrics()
    metric_results 
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    1. The data shows a statistically significant lift (125% in conversion rate). Note that the MDL from this test was ~22% so the test was powered to detect a lift of this size.
    2. Revenue per user declines by 73% indicating that the promotion cost offset the goodness in conversion resulting in a decline in revenue per user.

    Note that these results only make sense because the bootstrap confidence intervals indicate that these results are meaningful. Due to massive class imbalance (1% purchase rate) and large n-size, the p-values are likely to show 0 or near 0 and the best practice is to use bootstrap to infer results.
    """)
    return


@app.cell
def _(evaluator):

    statistics = evaluator.compute_statistics()
    return (statistics,)


@app.cell
def _(statistics):
    statistics
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    1. For conversion rate, since the CI doesn't cross 0 we can confirm that the treatment has a positive effect on conversion rate. Since ATE is a binary metric (purchase or no purchase), we have learned that the treatment increased the probability of the outcome by 0.95 percentage points (pp)

    2. If the experiment was repeated a 100 times, 95% of the time the ATE would be positive and will range between 0.8 pp to 1.08 pp

    3. Similarly, for net revenue per user, we can confirm that the treatment has a negative effect on net revenue per user. The ATE of -0.0555 means that, on average, the treatment caused a loss of $0.0555 per user compared to the control group.

    Example: If you scaled this treatment to 1M users, you would expect a total revenue loss of approximately $55,500.

    4. The Bootstrap CI of [-0.0695, -0.0417] provides the range of likely values for this revenue drop:
        Lower Bound: In the worst-case scenario, the treatment reduces revenue by $0.0695 per user.
        Upper Bound: In the best-case scenario, the treatment still reduces revenue by $0.0417 per user.
        Statistical Significance: Because the entire interval is below zero, this drives a statistically significant revenue loss.
    """)
    return


@app.cell
def _(evaluator):

    # actual NIR from experiment
    nir = evaluator.compute_nir()
    return (nir,)


@app.cell
def _(nir):
    nir
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Based on the assumptions of the experiment, the actual revenue loss for the test audience is ~$2.3K
    """)
    return


@app.cell
def _(df_train, evaluator, statistics):
    # future rollout projection - to be updated 
    future_population = len(df_train)  # df = the full population you want to scale to
    impact = evaluator.project_impact(statistics=statistics, population_size=future_population)
    return (impact,)


@app.cell
def _(impact):
    impact 
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Assuming that the population is equal to sum of control and treatment group (i.e. 100% of the population was a part of this experiment), the projected revenue loss at scale is $4.7K. If we repeated the experiment a 100 times, the range of loss is likely to be between $5.9K (worst case scenario) to $3.5K (best case scenario)
    """)
    return


if __name__ == "__main__":
    app.run()

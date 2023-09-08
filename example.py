
from scipy.stats import norm, t, ttest_ind
from scipy.special import stdtr
import numpy as np
import math


def generate_data(
        sample_size,
        avg_daily_conversion_rate_A,  # avg daily conversion rate for group A
        avg_daily_conversion_rate_B,  # avg daily conversion rate for group B
        std_dev = 0.05  # standard deviation (must be non-negative)
    ):
    """Generate fake data to perform a two-sample t-test"""

    # Set a random seed for reproducibility
    np.random.seed(42)

    # Generate data for group A and B
    group_A = np.random.normal(avg_daily_conversion_rate_A, std_dev, sample_size)
    group_B = np.random.normal(avg_daily_conversion_rate_B, std_dev, sample_size)

    return group_A, group_B


def get_min_sample_size(
        std_dev,  # standard deviation
        mde,  # minimum detectable effect
        alpha = 0.05,  # significance level
        power = 0.8  # statistical power
    ):
    """
    Estimate minimum sample size for t-test
    Assumptions:
        Sample sizes will be the same for both groups
        Both groups have the same standard deviation
    """
    
    # Find Z_beta from desired power
    Z_beta = norm.ppf(power)

    # Find Z_alpha
    Z_alpha = norm.ppf(1 - alpha / 2)

    # Return minimum sample size
    return math.ceil(2 * std_dev**2 * (Z_beta + Z_alpha)**2 / mde**2)


def main() -> None:
    # Define parameters
    std_dev = 0.05  # historical standard deviation
    alpha = 0.05  # i.e. 95% CL
    power = 0.8  # i.e. beta = 0.2
    mde = 0.03  # 15% effect on an avg_daily_conversion_rate of 0.2

    # Estimate minimum sample size
    min_sample_size = get_min_sample_size(std_dev = std_dev, mde = mde, alpha = alpha, power = power)
    print(f'Minimum sample size for alpha = {alpha}, power = {power} and mde = {mde}: {min_sample_size}')

    # Test #1 (rejecting the null hypothesis)
    print('\n>>> Test #1: rejecting the null hypothesis <<<')

    # Generate fake data
    group_A, group_B = generate_data(
        sample_size = min_sample_size,
        avg_daily_conversion_rate_A = 0.2,
        avg_daily_conversion_rate_B = 0.23,
        std_dev = std_dev
    )

    # Perform the t-test
    result = ttest_ind(group_A, group_B)
    pvalue = result.pvalue
    if pvalue < alpha:
       print(f"Decision: There is a significant difference between the groups (p-value = {pvalue}).")
    else:
       print(f"Decision: There is no significant difference between the groups (p-value = {pvalue}).")

    tstat = result.statistic
    print(f't-statistic = {round(tstat, 2)}')

    # Let's calculate the t-statistic by hand
    avgA = np.mean(group_A)
    avgB = np.mean(group_B)
    varA = np.var(group_A, ddof = 1)
    varB = np.var(group_B, ddof = 1)
    n = min_sample_size
    my_tstat = (avgA - avgB) / math.sqrt((varA + varB)/ n)
    print(f't-statistic calculated by hand = {round(my_tstat, 2)}')

    # Let's calculate the critical t-statistic value
    df = 2 * n - 2  # degrees of freedom
    critical_t_stat = round(t.ppf(1 - alpha / 2, df), 2)  # ppf = percent point function (inverse of the cumulative distribution function)
    print(f'critical t-statistic = {critical_t_stat}')
    # note: we fail to reject the null hypothesis if the absolute value of our t-statistic is below this critical t-statistic value

    # note:
    #   if t-statistic <= 0:
    #       p-value = 2 × (area to the left of the t distribution)
    #   if t-statistic > 0:
    #       p-value = 2 x (area to the right of the t distribution)
    my_pvalue = 2 * stdtr(df, -np.abs(my_tstat))  # stdtr = Student t distribution cumulative distribution function
    print(f'p-value calculated by hand = {my_pvalue}')
    
    # Test #2 (failing to reject the null hypothesis)
    print('\n>>> Test #2: failing to reject the null hypothesis <<<')

    # Generate fake data
    group_A, group_B = generate_data(
        sample_size = min_sample_size,
        avg_daily_conversion_rate_A = 0.2,
        avg_daily_conversion_rate_B = 0.201,
        std_dev = std_dev
    )

    # Let's again perform a t-test
    result = ttest_ind(group_A, group_B)
    pvalue = result.pvalue
    if pvalue < alpha:
       print(f"Decision: There is a significant difference between the groups (p-value = {pvalue}).")
    else:
       print(f"Decision: There is no significant difference between the groups (p-value = {pvalue}).")


if __name__ == '__main__':
    main()
# Statistical Testing in Insurance Claim Analysis

This guide covers various statistical methods used in analyzing insurance claim data, including t-tests, F-tests, Chi-Square tests, correlation tests, and more. Each method is explained with its hypothesis, use case, and Python implementation.

## Table of Contents
- [Comparing Means (t-tests)](#comparing-means-t-tests)
  - Independent t-test
  - Paired t-test
- [Comparing Variances (F-tests)](#comparing-variances-f-tests)
- [Testing for Normality](#testing-for-normality)
  - Shapiro-Wilk Test
  - Kolmogorov-Smirnov Test
- [Comparing Proportions (Chi-Square Tests)](#comparing-proportions-chi-square-tests)
- [Correlation Testing](#correlation-testing)
- [ANOVA (Analysis of Variance)](#anova-analysis-of-variance)
- [Regression Coefficients Testing](#regression-coefficients-testing)
- [Non-Parametric Tests](#non-parametric-tests)
- [Time Series Analysis](#time-series-analysis)
- [Feature Selection and Validation](#feature-selection-and-validation)

---

## Comparing Means (t-tests)

### 1. Independent t-test
Used to compare the average claim amounts between two different regions (e.g., urban vs. rural).

- **Hypothesis**:
  - H0: The mean claim amount is the same for both regions.
  - Ha: The mean claim amount is different for both regions.

- **Code**:
    ```python
    from scipy import stats

    # Example claim amounts for urban and rural regions
    urban_claims = [300, 450, 200, 400, 500]
    rural_claims = [320, 460, 210, 410, 480]

    # Independent t-test
    t_stat, p_val = stats.ttest_ind(urban_claims, rural_claims)

    print(f"t-statistic: {t_stat}, p-value: {p_val}")
    ```

### 2. Paired t-test
Compares the claim amounts before and after implementing a new fraud detection algorithm for the same policies.

- **Hypothesis**:
  - H0: There is no difference in claim amounts before and after the new algorithm.
  - Ha: There is a significant difference in claim amounts before and after the new algorithm.

- **Code**:
    ```python
    # Example claim amounts before and after fraud detection algorithm
    before_algorithm = [400, 420, 450, 500]
    after_algorithm = [380, 400, 430, 480]

    # Paired t-test
    t_stat, p_val = stats.ttest_rel(before_algorithm, after_algorithm)

    print(f"t-statistic: {t_stat}, p-value: {p_val}")
    ```

---

## Comparing Variances (F-tests)

### 3. F-test for comparing variances between two age groups

- **Hypothesis**:
  - H0: The variances of claim amounts are the same for both age groups.
  - Ha: The variances of claim amounts are different for both age groups.

- **Code**:
    ```python
    # Example claim amounts for two age groups
    age_group_1 = [320, 450, 600, 480, 510]
    age_group_2 = [300, 400, 550, 470, 495]

    # F-test to compare variances
    f_stat, p_val = stats.levene(age_group_1, age_group_2)

    print(f"F-statistic: {f_stat}, p-value: {p_val}")
    ```

---

## Testing for Normality

### 4. Shapiro-Wilk Test

Used to test if the claim amount distribution follows a normal distribution.

- **Hypothesis**:
  - H0: The claim amounts are normally distributed.
  - Ha: The claim amounts are not normally distributed.

- **Code**:
    ```python
    # Example claim amounts
    claims = [320, 400, 450, 470, 500]

    # Shapiro-Wilk test for normality
    stat, p_val = stats.shapiro(claims)

    print(f"Shapiro-Wilk test statistic: {stat}, p-value: {p_val}")
    ```

### 5. Kolmogorov-Smirnov Test

Used to test if the residuals of a regression model predicting claim amounts are normally distributed.

- **Hypothesis**:
  - H0: The residuals are normally distributed.
  - Ha: The residuals are not normally distributed.

- **Code**:
    ```python
    from scipy.stats import kstest

    # Example residuals from a regression model
    residuals = [0.5, -0.3, 0.2, -0.4, 0.1]

    # Kolmogorov-Smirnov test
    stat, p_val = kstest(residuals, 'norm')

    print(f"K-S test statistic: {stat}, p-value: {p_val}")
    ```

---

## Comparing Proportions (Chi-Square Tests)

### 6. Chi-Square Test of Independence

Used to test if there is a significant association between gender and claim approval status.

- **Hypothesis**:
  - H0: Gender and claim approval status are independent.
  - Ha: Gender and claim approval status are not independent.

- **Code**:
    ```python
    from scipy.stats import chi2_contingency

    # Example contingency table: Gender vs. Claim Approval
    contingency_table = [[100, 200],  # Male
                         [150, 250]]  # Female

    # Chi-square test
    chi2, p_val, dof, expected = chi2_contingency(contingency_table)

    print(f"Chi-square statistic: {chi2}, p-value: {p_val}")
    ```

### 7. Chi-Square Goodness-of-Fit Test

Used to test if the distribution of claim types matches the expected distribution.

- **Hypothesis**:
  - H0: The observed distribution of claim types matches the expected distribution.
  - Ha: The observed distribution of claim types does not match the expected distribution.

- **Code**:
    ```python
    # Observed and expected frequencies of claim types
    observed = [50, 30, 20]  # Accident, Theft, Fire
    expected = [45, 35, 20]  # Expected distribution

    # Chi-square goodness-of-fit test
    chi2, p_val = stats.chisquare(observed, f_exp=expected)

    print(f"Chi-square statistic: {chi2}, p-value: {p_val}")
    ```

---

## Correlation Testing

### 8. Pearson Correlation Test

Tests the linear relationship between policyholder's age and claim amount.

- **Hypothesis**:
  - H0: There is no linear relationship between age and claim amount.
  - Ha: There is a linear relationship between age and claim amount.

- **Code**:
    ```python
    # Example ages and claim amounts
    ages = [25, 35, 45, 55, 65]
    claim_amounts = [200, 400, 350, 450, 500]

    # Pearson correlation test
    corr, p_val = stats.pearsonr(ages, claim_amounts)

    print(f"Pearson correlation: {corr}, p-value: {p_val}")
    ```

---

## ANOVA (Analysis of Variance)

### 9. One-way ANOVA

Used to compare the mean claim amounts across different types of insurance policies (e.g., health, auto, home).

- **Hypothesis**:
  - H0: The mean claim amounts are the same across all policy types.
  - Ha: At least one policy type has a different mean claim amount.

- **Code**:
    ```python
    # Example claim amounts for different policy types
    health = [300, 320, 350, 370, 390]
    auto = [200, 220, 240, 260, 280]
    home = [400, 420, 440, 460, 480]

    # One-way ANOVA
    f_stat, p_val = stats.f_oneway(health, auto, home)

    print(f"F-statistic: {f_stat}, p-value: {p_val}")
    ```

---

## Regression Coefficients Testing

### 10. t-tests for Regression Coefficients

Tests if the coefficients in a regression model are significantly different from zero.

- **Hypothesis**:
  - H0: The coefficient of the predictor is zero (no effect).
  - Ha: The coefficient of the predictor is not zero (significant effect).

- **Code**:
    ```python
    from sklearn.linear_model import LinearRegression
    import numpy as np

    # Example data
    X = np.array([[25, 1], [35, 0], [45, 1], [55, 0], [65, 1]])
    y = np.array([200, 400, 350, 450, 500])

    # Fit linear regression model
    model = LinearRegression().fit(X, y)

    # Get coefficients
    coef = model.coef_

    print(f"Regression coefficients: {coef}")
    ```

---

## Non-Parametric Tests

### 11.

 Mann-Whitney U Test

Used to compare claim amounts between two regions when data is not normally distributed.

- **Hypothesis**:
  - H0: The distributions of claim amounts are the same for both regions.
  - Ha: The distributions of claim amounts are different for both regions.

- **Code**:
    ```python
    # Example claim amounts for two regions
    region_1 = [300, 450, 200, 400, 500]
    region_2 = [320, 460, 210, 410, 480]

    # Mann-Whitney U test
    u_stat, p_val = stats.mannwhitneyu(region_1, region_2)

    print(f"U-statistic: {u_stat}, p-value: {p_val}")
    ```

### 12. Wilcoxon Signed-Rank Test

Compares claim amounts before and after a new policy for the same set of customers.

- **Hypothesis**:
  - H0: There is no difference in claim amounts before and after the new policy.
  - Ha: There is a significant difference in claim amounts before and after the new policy.

- **Code**:
    ```python
    # Example claim amounts before and after new policy
    before_policy = [400, 420, 450, 500]
    after_policy = [380, 400, 430, 480]

    # Wilcoxon signed-rank test
    w_stat, p_val = stats.wilcoxon(before_policy, after_policy)

    print(f"Wilcoxon statistic: {w_stat}, p-value: {p_val}")
    ```

---

## Time Series Analysis

### 13. Augmented Dickey-Fuller Test

Used to test for stationarity in the time series data of monthly claim amounts.

- **Hypothesis**:
  - H0: The time series has a unit root (is non-stationary).
  - Ha: The time series does not have a unit root (is stationary).

- **Code**:
    ```python
    from statsmodels.tsa.stattools import adfuller

    # Example time series data of monthly claim amounts
    monthly_claims = [200, 220, 240, 230, 250, 270, 260]

    # Augmented Dickey-Fuller test
    adf_stat, p_val, _, _, _, _ = adfuller(monthly_claims)

    print(f"ADF statistic: {adf_stat}, p-value: {p_val}")
    ```

---

## Feature Selection and Validation

### 14. Permutation Tests for Feature Significance

Validates the significance of features like age, income, and policy type in predicting claim amounts.

- **Hypothesis**:
  - H0: The feature importance is the same as in permuted data.
  - Ha: The feature importance is greater than in permuted data.

- **Code**:
    ```python
    from sklearn.inspection import permutation_importance

    # Example data
    X = np.array([[25, 50000], [35, 60000], [45, 70000], [55, 80000], [65, 90000]])
    y = np.array([200, 400, 350, 450, 500])

    # Fit a simple regression model
    model = LinearRegression().fit(X, y)

    # Perform permutation importance
    result = permutation_importance(model, X, y, n_repeats=10, random_state=0)

    print(f"Feature importance: {result.importances_mean}")
    ```

---

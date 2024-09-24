
# Outlier Detection Techniques

This repository provides an overview of common outlier detection techniques used in data preprocessing. Outliers are data points that deviate significantly from the majority of data and can impact machine learning model performance. This document covers both statistical and machine learning-based outlier detection techniques.

## Techniques Overview

1. **Z-Score Method:**
   - **Description:** This method calculates the z-score for each data point and identifies outliers based on a threshold (e.g., z-score > 3 or < -3).
   - **Formula:**  
     \[
     Z = \frac{X - \mu}{\sigma}
     \]
   - **Example:**
     ```
     Data: [10, 15, 20, 100, 25, 30]
     Mean: 30
     Standard Deviation: 30.276
     Z-Scores: [-0.66, -0.49, -0.33, 2.97, -0.16, 0]
     Outlier: 100 (z-score = 2.97)
     ```

2. **Interquartile Range (IQR) Method:**
   - **Description:** Defines outliers as data points that fall below \(Q1 - 1.5 \times IQR\) or above \(Q3 + 1.5 \times IQR\), where \(Q1\) is the first quartile, \(Q3\) is the third quartile, and IQR is the interquartile range.
   - **Example:**
     ```
     Data: [10, 15, 20, 100, 25, 30]
     Q1: 15
     Q3: 30
     IQR: 15
     Outlier Bounds: [15 - 1.5*15, 30 + 1.5*15] = [-7.5, 52.5]
     Outlier: 100
     ```

3. **Modified Z-Score Method:**
   - **Description:** Similar to the z-score method, but uses the median and the median absolute deviation (MAD) instead of the mean and standard deviation.
   - **Formula:**  
     \[
     MZ = \frac{0.6745 \times (X - \text{median})}{\text{MAD}}
     \]
   - **Example:**
     ```
     Data: [10, 15, 20, 100, 25, 30]
     Median: 22.5
     MAD: 7.5
     Modified Z-Scores: [-1, -0.67, -0.33, 7.33, 0, 0.33]
     Outlier: 100 (modified z-score = 7.33)
     ```

4. **Isolation Forest:**
   - **Description:** Builds an ensemble of decision trees and identifies outliers as data points that have shorter average path lengths in the trees.
   - **Algorithm:** Uses random forest isolation-based outlier detection.

5. **DBSCAN (Density-Based Spatial Clustering of Applications with Noise):**
   - **Description:** Clusters data points based on density and identifies outliers as points that do not belong to any cluster or belong to very small clusters.

6. **Local Outlier Factor (LOF):**
   - **Description:** Computes the local density deviation of a data point with respect to its neighbors and identifies outliers based on significant deviations in local density.

## Code Implementation

### Outlier Detection Techniques

Here is Python code implementing various outlier detection techniques described above. The code uses `numpy` and `scikit-learn` libraries.

```python
# Required Libraries
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import zscore

# Example data
data = np.array([10, 20, 15, 30, 100, 25, 40, 50])

# 1. Z-Score Method
mean = np.mean(data)
std_dev = np.std(data)
z_scores = zscore(data)
outliers_z_score = np.abs(z_scores) > 3  # Identifying outliers with z-score > 3
print("Z-Score Outliers:", data[outliers_z_score])

# 2. Interquartile Range (IQR) Method
q1 = np.percentile(data, 25)
q3 = np.percentile(data, 75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers_iqr = (data < lower_bound) | (data > upper_bound)
print("IQR Outliers:", data[outliers_iqr])

# 3. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
dbscan = DBSCAN(eps=15, min_samples=2)
labels = dbscan.fit_predict(data.reshape(-1, 1))
outliers_dbscan = labels == -1
print("DBSCAN Outliers:", data[outliers_dbscan])

# 4. Isolation Forest
isolation_forest = IsolationForest(contamination=0.1)
outliers_isolation_forest = isolation_forest.fit_predict(data.reshape(-1, 1)) == -1
print("Isolation Forest Outliers:", data[outliers_isolation_forest])

# 5. Local Outlier Factor (LOF)
lof = LocalOutlierFactor(n_neighbors=2)
outliers_lof = lof.fit_predict(data.reshape(-1, 1)) == -1
print("Local Outlier Factor Outliers:", data[outliers_lof])
```

### Explanation of Code

1. **Z-Score Method:**
   - Calculates the z-scores for each data point and identifies outliers as those with an absolute z-score greater than 3.
   ```python
   z_scores = zscore(data)
   outliers_z_score = np.abs(z_scores) > 3
   ```

2. **Interquartile Range (IQR) Method:**
   - Computes the 1st and 3rd quartiles and determines the outlier bounds using the IQR. Outliers are points outside this range.
   ```python
   q1 = np.percentile(data, 25)
   q3 = np.percentile(data, 75)
   iqr = q3 - q1
   lower_bound = q1 - 1.5 * iqr
   upper_bound = q3 + 1.5 * iqr
   ```

3. **DBSCAN:**
   - Uses a density-based clustering algorithm where points in low-density regions are labeled as outliers (`label = -1`).
   ```python
   dbscan = DBSCAN(eps=15, min_samples=2)
   labels = dbscan.fit_predict(data.reshape(-1, 1))
   outliers_dbscan = labels == -1
   ```

4. **Isolation Forest:**
   - Identifies outliers by isolating data points with fewer splits required in a random forest model. Points with shorter paths are considered outliers.
   ```python
   isolation_forest = IsolationForest(contamination=0.1)
   outliers_isolation_forest = isolation_forest.fit_predict(data.reshape(-1, 1)) == -1
   ```

5. **Local Outlier Factor (LOF):**
   - Measures the local density deviation of a point with respect to its neighbors. Points with significantly lower density are outliers.
   ```python
   lof = LocalOutlierFactor(n_neighbors=2)
   outliers_lof = lof.fit_predict(data.reshape(-1, 1)) == -1
   ```

### Example Output

The resulting output would look like this:

```bash
Z-Score Outliers: []
IQR Outliers: [100]
DBSCAN Outliers: [100]
Isolation Forest Outliers: [100]
Local Outlier Factor Outliers: [100]
```

## Considerations

- **Z-Score Method:** Works well for normally distributed data but may fail for skewed distributions.
- **IQR Method:** Robust to outliers but may fail for small datasets or those with uneven distribution.
- **DBSCAN:** Effective in detecting outliers in spatial datasets, but the choice of parameters `eps` and `min_samples` is crucial.
- **Isolation Forest:** Works well for datasets with many features but may not be suitable for all types of data.
- **Local Outlier Factor:** Detects outliers based on local density, which makes it more effective for high-dimensional datasets.

---

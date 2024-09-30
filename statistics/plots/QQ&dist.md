### QQ-Plot (Quantile-Quantile Plot)

A **QQ-plot** is used to compare the distribution of a dataset against a theoretical distribution (commonly the normal distribution). It helps to visually assess whether the data follows a specific distribution. Here's how it works:

- **X-axis**: The theoretical quantiles of the distribution you're comparing against (e.g., normal distribution).
- **Y-axis**: The quantiles of the actual data.
- If the data matches the theoretical distribution, the points in the QQ-plot will lie roughly along a 45-degree line.

#### Uses:
- To check **normality** (whether data is normally distributed).
- To assess how well a dataset fits a certain distribution (e.g., exponential, uniform).

#### Interpretation:
- If the points deviate significantly from the 45-degree line, the data likely doesn't follow the assumed distribution.
  - **Upward curve**: Indicates the distribution is more right-skewed than normal.
  - **Downward curve**: Indicates the distribution is more left-skewed than normal.

#### Example Code for QQ-Plot

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Example: Generate a sample dataset (Normal distribution)
data = np.random.normal(loc=0, scale=1, size=1000)

# Function to create QQ-Plot
def plot_qq(data, distribution='norm'):
    plt.figure(figsize=(6, 6))
    stats.probplot(data, dist=distribution, plot=plt)
    plt.title(f'QQ-Plot against {distribution} distribution')
    plt.show()

# Usage
plot_qq(data)  # QQ-Plot against Normal Distribution
```

### Explanation of Code:
- `stats.probplot(data, dist='norm', plot=plt)`:
  - This function generates the QQ-plot.
  - The `dist='norm'` argument specifies that we are comparing against a normal distribution.
  - `plot=plt` ensures the plot is rendered using Matplotlib.

---

### Distplot (Distribution Plot)

A **distplot** combines a **histogram** and **KDE (Kernel Density Estimate)** that shows the distribution of data. It's useful for visualizing the overall distribution and the probability density of a dataset.

- **Histogram**: The bars represent the count of data points within bins.
- **KDE curve**: A smooth curve that estimates the probability density function of the data.

#### Uses:
- To visualize the distribution of a dataset.
- To see both the counts of data in bins and an estimate of the continuous distribution.

#### Interpretation:
- The **histogram** shows the frequency of data points in specific ranges.
- The **KDE curve** gives an idea of the underlying distribution's shape.
- Helps in checking for skewness, modality (number of peaks), and general distribution shape.

#### Example Code for Distplot

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Example: Generate a sample dataset (Normal distribution)
data = np.random.normal(loc=0, scale=1, size=1000)

# Function to create Distplot
def plot_dist(data, bins=30, kde=True):
    plt.figure(figsize=(8, 6))
    sns.histplot(data, bins=bins, kde=kde)
    plt.title('Distribution Plot (Histogram + KDE)')
    plt.xlabel('Data values')
    plt.ylabel('Frequency / Density')
    plt.show()

# Usage
plot_dist(data)  # Distplot with Histogram + KDE
```

### Explanation of Code:
- `sns.histplot(data, bins=30, kde=True)`:
  - `data`: The dataset to plot.
  - `bins=30`: The number of bins for the histogram.
  - `kde=True`: Adds a Kernel Density Estimate (KDE) curve to show the probability density function.

### Customization:
- **QQ-Plot**: You can change the `dist` argument in `stats.probplot()` to compare the data against other distributions (e.g., `expon`, `uniform`).
- **Distplot**: You can set `kde=False` if you don't want the KDE curve and just need the histogram.

---

### Summary:
- **QQ-Plot**: Used to compare a dataset to a theoretical distribution (e.g., normality check).
- **Distplot**: Combines a histogram and KDE for visualizing the data's distribution and probability density.

These tools are useful for assessing the distribution of your data before applying statistical tests or machine learning models.
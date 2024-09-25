
# Sampling Techniques

Sampling techniques are methods used to select a subset of data from a larger population, allowing for efficient analysis and inference while saving time and computational resources. This guide covers various common sampling techniques along with their Python implementations.

## Sampling Techniques Overview

### 1. Simple Random Sampling
- **Description:** Every member of the population has an equal chance of being selected.
- **Use Case:** This method is useful when each data point is equally important.
- **Example:**
    ```python
    import random

    # Define the population
    population = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # Set the sample size
    sample_size = 5
    
    # Draw a random sample
    sample = random.sample(population, sample_size)
    
    print("Simple Random Sample:", sample)
    ```

### 2. Stratified Sampling
- **Description:** Divides the population into homogeneous subgroups (strata) and selects a random sample from each stratum to ensure representation.
- **Use Case:** Useful when the population has distinct subgroups (e.g., different classes).
- **Example:**
    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris

    # Load dataset (e.g., Iris dataset)
    data = load_iris(as_frame=True).frame
    X = data.drop(columns=['target'])
    y = data['target']

    # Stratify sampling to maintain the proportion of classes in training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    print("Stratified Sampling - Train/Test Split Done")
    ```

### 3. Systematic Sampling
- **Description:** Selects every k-th item from the population after a random start.
- **Use Case:** Suitable for ordered data where selection at regular intervals is preferred.
- **Example:**
    ```python
    import random

    # Population and k value
    population = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    k = 2

    # Random start
    start = random.randint(0, k - 1)
    
    # Select every k-th element starting from the random start
    systematic_sample = population[start::k]
    
    print("Systematic Sample:", systematic_sample)
    ```

### 4. Cluster Sampling
- **Description:** Divides the population into clusters, randomly selects a few clusters, and samples all members within the selected clusters.
- **Use Case:** Useful when the population is large and geographically dispersed, making other sampling methods difficult.
- **Example:**
    ```python
    from sklearn.cluster import KMeans
    import pandas as pd
    import random

    # Simulate a dataset
    X = pd.DataFrame({
        'feature_1': random.sample(range(1, 100), 50),
        'feature_2': random.sample(range(1, 100), 50)
    })

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=5)
    X['cluster'] = kmeans.fit_predict(X[['feature_1', 'feature_2']])
    
    # Randomly select two clusters
    selected_clusters = random.sample(list(X['cluster'].unique()), 2)

    # Sample all data points from the selected clusters
    cluster_sample = X[X['cluster'].isin(selected_clusters)]
    
    print("Cluster Sample:\n", cluster_sample)
    ```

### 5. Sequential Sampling
- **Description:** Involves selecting data points one at a time, typically in a sequential manner, until the desired sample size is reached.
- **Use Case:** Common in scenarios where data points arrive one after another, such as in real-time systems.
- **Example:**
    ```python
    # Example sequential sampling process
    sequential_sample = []
    population = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    sample_size = 5

    # Select next data point in a sequence
    for i in range(sample_size):
        next_point = population[i]  # Here we're selecting points sequentially
        sequential_sample.append(next_point)
    
    print("Sequential Sample:", sequential_sample)
    ```

### 6. Reservoir Sampling
- **Description:** Used when the total number of elements in the population is not known in advance. It ensures each element has an equal probability of being selected.
- **Use Case:** Best for streaming data or unknown data sizes, ensuring a fair representation.
- **Example:**
    ```python
    import itertools
    import random

    # Reservoir sampling function
    def reservoir_sampling(stream, k):
        reservoir = list(itertools.islice(stream, k))  # Initial reservoir
        n = k
        for item in stream:
            n += 1
            i = random.randint(0, n-1)
            if i < k:
                reservoir[i] = item  # Replace a random element
        return reservoir

    # Simulating a data stream
    stream = iter(range(1, 101))  # Infinite stream of data

    # Sample size
    sample_size = 5

    # Perform reservoir sampling
    sample = reservoir_sampling(stream, sample_size)

    print("Reservoir Sample:", sample)
    ```

## Code Explanation

1. **Simple Random Sampling:**
   - Randomly selects a subset from the population where every element has an equal chance of being picked.
   ```python
   sample = random.sample(population, sample_size)
   ```

2. **Stratified Sampling:**
   - Ensures that different subgroups (strata) are proportionally represented in the sample.
   ```python
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
   ```

3. **Systematic Sampling:**
   - Selects every k-th element from the population after a random start.
   ```python
   systematic_sample = population[start::k]
   ```

4. **Cluster Sampling:**
   - Divides data into clusters, then samples all members from selected clusters.
   ```python
   selected_clusters = random.sample(list(X['cluster'].unique()), 2)
   ```

5. **Sequential Sampling:**
   - Sequentially selects data points until a desired sample size is reached.
   ```python
   next_point = population[i]  # Sequential sampling
   ```

6. **Reservoir Sampling:**
   - Ensures each element has an equal probability of being selected from a stream of unknown size.
   ```python
   sample = reservoir_sampling(stream, sample_size)
   ```

## Considerations

- **Simple Random Sampling:** Works best when all elements in the population are accessible and there's no underlying structure.
- **Stratified Sampling:** Ensures each subgroup is represented but requires knowledge of the structure of the population.
- **Systematic Sampling:** Useful when you need a simple and regular selection but may introduce bias if there's a periodic trend in the data.
- **Cluster Sampling:** Efficient for large populations but may introduce bias if clusters are not homogeneous.
- **Sequential Sampling:** Common in real-time or streaming data scenarios but can introduce selection bias if not handled carefully.
- **Reservoir Sampling:** Ideal for unknown population sizes, especially in data streams, as it ensures each element is sampled fairly.

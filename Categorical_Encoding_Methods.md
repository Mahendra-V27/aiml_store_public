# Categorical Encoding Methods

This repository provides an overview of common categorical encoding methods used to transform categorical variables into a numerical format that can be used in machine learning algorithms. These encodings help machine learning models process categorical data efficiently. The code examples are written in Python using `pandas` and `category_encoders` libraries.

## Categorical Encoding Techniques

1. **One-Hot Encoding**
   - **Description:** Transforms each category into a binary vector where one bit is set to `1` and the rest are `0`.
   - **Example:**
     ```
     Category:   A    B    C
     Encoding:  [1, 0, 0]
                [0, 1, 0]
                [0, 0, 1]
     ```
   - **Use Case:** Best suited for nominal categories where there is no ordinal relationship between them.

2. **Label Encoding**
   - **Description:** Maps each category to a unique integer.
   - **Example:**
     ```
     Category:   A    B    C
     Encoding:    0    1    2
     ```
   - **Use Case:** Works well when there is no ordinal relationship, but can cause issues for models like linear regression that may interpret the values as having a natural order.

3. **Ordinal Encoding**
   - **Description:** Similar to label encoding, but categories are mapped based on a defined order or ranking.
   - **Example:**
     ```
     Category:   Low    Medium    High
     Encoding:    0        1         2
     ```
   - **Use Case:** Used when categories have a natural order, such as low, medium, and high.

4. **Frequency Encoding**
   - **Description:** Encodes categories based on their frequency or occurrence in the dataset.
   - **Example:**
     ```
     Category:  A    B    A    C    B
     Encoding:  2    2    2    1    2
     ```
   - **Use Case:** Useful when the frequency of occurrence is significant for the model.

5. **Target Encoding**
   - **Description:** Categories are encoded based on the mean of the target variable for each category.
   - **Example:**
     ```
     Category:  A    B    A    C    B
     Target:    1    0    1    1    0
     Encoding:  0.67 0.33 0.67 1    0.33
     ```
   - **Use Case:** Works well for high-cardinality features in classification and regression problems, but it requires careful validation to prevent data leakage.

## Usage

The provided Python script implements the five encoding techniques on a sample dataset. To run the code:

1. Load the dataset.
2. Apply the encoding techniques as demonstrated in the script.

### Example:
```python
# Example dataset with 'Category' and 'Target' columns
data = {
    'Category': ['A', 'B', 'A', 'C', 'B'],
    'Target': [1, 0, 1, 1, 0]
}

# Convert to pandas DataFrame
df = pd.DataFrame(data)

# Apply different encoding techniques (see code in script for details)
```

## Results

After running the script, you will see various encoding results applied to the `Category` column. Each encoding method is suitable for different scenarios, so it's important to select the appropriate method for your use case.

## Considerations

- **One-Hot Encoding** is ideal when categories are nominal and unordered, but it may increase the dimensionality of the dataset for high-cardinality columns.
- **Label Encoding** is simple but can introduce ordinal relationships that may confuse models like linear regression.
- **Ordinal Encoding** works best for categories with a meaningful order, such as ratings.
- **Frequency and Target Encoding** are useful for reducing dimensionality but may require careful validation to avoid overfitting or data leakage.

Here's a Python implementation of different categorical encoding methods using libraries like `pandas` and `category_encoders`. I've included inline comments for clarity.

```python
# Required libraries
import pandas as pd
import category_encoders as ce  # For advanced encoding techniques like Target and Frequency Encoding

# Example dataset with a categorical column 'Category' and a 'Target' column for target encoding
data = {
    'Category': ['A', 'B', 'A', 'C', 'B'],
    'Target': [1, 0, 1, 1, 0]
}

# Convert to a pandas DataFrame
df = pd.DataFrame(data)

# 1. One-Hot Encoding
# Create binary columns for each category
df_one_hot = pd.get_dummies(df['Category'], prefix='Category')
# Concatenating the one-hot encoded columns to the original dataframe
df_one_hot = pd.concat([df, df_one_hot], axis=1)

# 2. Label Encoding
# Label encoding assigns an integer to each category
df['Category_LabelEncoded'] = df['Category'].astype('category').cat.codes

# 3. Ordinal Encoding
# Custom ordinal encoding based on a logical ranking
ordinal_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
df['Category_OrdinalEncoded'] = df['Category'].map({'A': 0, 'B': 1, 'C': 2})  # Example of ordinal mapping

# 4. Frequency Encoding
# Frequency encoding assigns the count of each category to that category
frequency_encoding = df['Category'].value_counts(normalize=False).to_dict()
df['Category_FrequencyEncoded'] = df['Category'].map(frequency_encoding)

# 5. Target Encoding
# Target encoding encodes categories based on the target variable (e.g., mean of 'Target')
target_encoder = ce.TargetEncoder(cols=['Category'])
df['Category_TargetEncoded'] = target_encoder.fit_transform(df['Category'], df['Target'])

# Display the resulting dataframe
print(df)
```

### Explanation:
1. **One-Hot Encoding:**
   - Converts each unique category into binary vectors (0 or 1).
   - Utilizes `pandas.get_dummies()` to create one column per category.
   
2. **Label Encoding:**
   - Assigns a unique integer to each category.
   - Uses the `cat.codes` method in pandas to map categories to integers.
   
3. **Ordinal Encoding:**
   - Used when categories have a meaningful order.
   - Manually maps categories to a specific order (in this case, A, B, and C are mapped in a custom order).

4. **Frequency Encoding:**
   - Assigns the count of each category as its encoding. More frequent categories get higher numbers.
   - Achieved by counting category frequencies using `value_counts()` and mapping these values back to the dataframe.

5. **Target Encoding:**
   - Encodes categories based on the mean of the target variable.
   - Uses `category_encoders.TargetEncoder` to calculate the mean of the target for each category.

Each of these methods serves a different purpose based on the context of the data and the machine learning algorithm used. Let me know if you'd like any further customization or explanations!

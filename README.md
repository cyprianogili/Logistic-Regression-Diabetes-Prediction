# Logistic-Regression-Diabetes-Prediction
A supervised machine learning project that uses Logistic Regression to predict whether a patient has diabetes based on medical features.




## What is Logistic Regression?

Logistic Regression is a supervised machine learning algorithm used to predict probabilities and classify data into binary outcomes (like `0` or `1`, Yes or No).

**Key difference from other algorithms:**

| Algorithm | Predicts | Example Output |
|-----------|---------|----------------|
| Linear Regression | A number | House price: ₦45,000,000 |
| Logistic Regression | Yes or No | Has diabetes: Yes (1) or No (0) |
| Decision Tree | Yes or No (via flowchart) | Branching yes/no questions |

**How it works:**

1. Looks at input features (clues)
2. Calculates a weighted sum
3. Passes it through a **sigmoid function** to get a probability (0 to 1)
4. Applies a cutoff (usually 0.5) to make a final Yes/No decision

> The sigmoid function squeezes any number into the range 0–1, making it suitable for probability estimation.

---

## Dataset

The dataset used is the **Pima Indians Diabetes Dataset**, containing medical records of 768 patients.

| Column Name | Description |
|-------------|-------------|
| **Pregnancies** | Number of times the patient has been pregnant |
| **Glucose** | Blood glucose level measured after a glucose tolerance test |
| **BloodPressure** | Diastolic blood pressure reading |
| **SkinThickness** | Thickness of the skin fold on the triceps, used to estimate body fat |
| **Insulin** | Insulin level in the blood measured after 2 hours |
| **BMI** | Body Mass Index, a measure of body fat based on height and weight |
| **DiabetesPedigreeFunction** | Score representing genetic risk of diabetes based on family history |
| **Age** | Age of the patient in years |
| **Outcome** | Final diagnosis: **0 = No diabetes**, **1 = Diabetes** |

---

## Project Workflow

### 1. Load and Inspect the Data

```python
df_log.head()
df_log.info()
df_log.describe()
```

**Key observations from `.info()`:**
- 768 rows, 9 columns
- All columns are numeric (`int64` or `float64`)
- No missing values at first glance

**Key observations from `.describe()`:**
- Several columns show `0` as their minimum value
- A person cannot have 0 glucose, 0 blood pressure, or 0 BMI and be alive
- These zeros are **placeholders for missing data**

---

### 2. Check for Missing Values

```python
df_log.isnull().sum()
```

Initially returns all zeros — because the missing data is disguised as `0`.

```python
(df_log == 0).sum()
```

**Results:**

| Column | Zero Count | Valid? |
|--------|-----------|--------|
| Pregnancies | 111 | ✅ Yes — can have 0 pregnancies |
| Glucose | 5 | ❌ Fake — impossible to have 0 blood sugar |
| BloodPressure | 35 | ❌ Fake — cannot be alive with 0 blood pressure |
| SkinThickness | 227 | ❌ Fake — everyone has skin |
| Insulin | 374 | ❌ Fake — nearly half the dataset |
| BMI | 11 | ❌ Fake — everyone has body mass |
| DiabetesPedigreeFunction | 0 | ✅ No zeros |
| Age | 0 | ✅ No zeros |
| Outcome | 500 | ✅ Valid — 500 patients simply have no diabetes |

---

### 3. Replace Fake Zeros with NaN

**Why do this?**
- Python treats `0` as a real value — it will distort statistics and mislead the model
- `NaN` (Not a Number) is Python's official way of saying "this value is missing"
- Once marked as `NaN`, pandas tools like `.fillna()` can handle them properly

```python
# Columns where 0 is physiologically invalid
invalid_zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Replace fake zeros with NaN
df_log[invalid_zero_cols] = df_log[invalid_zero_cols].replace(0, np.nan)
```

Verify the replacement:

```python
df_log.isnull().sum()
```

**Results after replacement:**

| Column | NaN Count |
|--------|----------|
| Glucose | 5 |
| BloodPressure | 35 |
| SkinThickness | 227 |
| Insulin | 374 |
| BMI | 11 |

---

### 4. Fill Missing Values with Median

**Why median and not mean?**

| Strategy | Best for |
|----------|---------|
| Mean | Normally distributed columns |
| Median | Skewed columns — not affected by extreme values |
| Mode | Categorical columns |

Since most affected columns are skewed (especially Insulin), **median** is the safer choice.

```python
cols_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Fill NaN with the median of each column
df_log[cols_to_impute] = df_log[cols_to_impute].fillna(df_log[cols_to_impute].median())
```

Verify no missing values remain:

```python
df_log.isnull().sum()
# All columns should now return 0
```

---

### 5. Exploratory Data Analysis (EDA)

#### Distribution of each feature

```python
plt.figure(figsize=(10, 5))
sns.histplot(data=df_log, x='Pregnancies', kde=True)
plt.title('Distribution of Pregnancies')
plt.xlabel('Pregnancies')
plt.ylabel('Frequency')
plt.show()
```

**Distribution summary:**

| Column | Shape | Key Observation |
|--------|-------|----------------|
| Pregnancies | Right skewed | Most patients have 0–1 pregnancies |
| Glucose | Nearly normal | Most values centered around 100–140 |
| BloodPressure | Bell shaped | Most readings between 70–90 mmHg |
| SkinThickness | Right skewed | Most values around 20–30 mm |
| Insulin | Strongly right skewed | Most values low, few extremely high |
| BMI | Nearly normal | Slight right skew, most in healthy-overweight range |
| DiabetesPedigreeFunction | Strongly right skewed | Most have low genetic risk |
| Age | Right skewed | Most patients are in their 20s–30s |

#### Missing Values Heatmap

```python
plt.figure(figsize=(10, 5))
sns.heatmap(df_log.isnull(), cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()
```

> Yellow = missing value | Dark purple = value present

#### Class Distribution

```python
outcome_counts = df_log['Outcome'].value_counts()
print(outcome_counts)

plt.figure(figsize=(6, 4))
sns.countplot(data=df_log, x='Outcome')
plt.title('Distribution of Outcome Variable')
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['No Diabetes', 'Diabetes'])
plt.show()
```

**Results:**

| Outcome | Count | Percentage |
|---------|-------|-----------|
| No Diabetes (0) | 500 | 65% |
| Diabetes (1) | 268 | 35% |

> ⚠️ **Class Imbalance detected.** The model may become biased toward predicting "No Diabetes" since it dominates the dataset.

---

### 6. Prepare Data for Modelling

```python
# Separate features and target
X = df_log[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = df_log['Outcome']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

**Why `stratify=y`?**
> Ensures both training and test sets maintain the same 65/35 class ratio as the original dataset.

**Split results:**

| Set | Size | Purpose |
|-----|------|---------|
| X_train / y_train | 80% (614 patients) | Model learns from this |
| X_test / y_test | 20% (154 patients) | Model is tested on this |

---

### 7. Train the Model

```python
model_log = LogisticRegression(max_iter=1000)
model_log.fit(X_train, y_train)
```

> `max_iter=1000` gives the model up to 1000 attempts to find the best weights during training.

---

### 8. Feature Importance

```python
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model_log.coef_[0]
})
feature_importance = feature_importance.sort_values('Coefficient', ascending=False)
display(feature_importance)
```

> A **positive coefficient** means the feature pushes toward Diabetes (1).
> A **negative coefficient** means the feature pushes toward No Diabetes (0).
> A **larger absolute value** means stronger influence.

---

### 9. Evaluate the Model

#### Accuracy

```python
# Training accuracy
y_train_pred = model_log.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy:.4f}")

# Testing accuracy
y_test_pred = model_log.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Testing Accuracy: {test_accuracy:.4f}")
```

**Results:**

| Set | Accuracy |
|-----|---------|
| Training | 79.32% |
| Testing | 70.13% |

> The ~9% gap indicates slight overfitting — the model learned training data a little too well.

> ⚠️ **Accuracy can be misleading** with imbalanced datasets. A model that predicts "No Diabetes" for everyone would still achieve 65% accuracy.

#### Confusion Matrix

```python
print(f"\nConfusion Matrix (Test Set):")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
```

**Results:**

```
[[81  19]
 [27  27]]
```

| | Predicted No Diabetes | Predicted Diabetes |
|---|---|---|
| **Actually No Diabetes** | 81 (TN) ✅ | 19 (FP) ❌ |
| **Actually Has Diabetes** | 27 (FN) 🚨 | 27 (TP) ✅ |

**Confusion Matrix Terms:**

| Term | Meaning | Example |
|------|---------|---------|
| True Positive (TP) | Model said Diabetes — correct | 27 diabetic patients correctly identified |
| True Negative (TN) | Model said No Diabetes — correct | 81 healthy patients correctly identified |
| False Positive (FP) | Model said Diabetes — patient was healthy | 19 healthy patients wrongly flagged |
| False Negative (FN) | Model said No Diabetes — patient was diabetic | 27 sick patients missed 🚨 |

> The **27 False Negatives** are the biggest concern — these are real diabetic patients the model failed to detect (50% of all diabetic test cases missed).

---

## Key Takeaways

| Topic | Summary |
|-------|---------|
| Fake zeros | Columns like Glucose and Insulin used 0 to represent missing data |
| NaN replacement | Fake zeros replaced with NaN to mark them as officially missing |
| Imputation | NaN values filled with column median due to skewed distributions |
| Class imbalance | Dataset is 65% No Diabetes vs 35% Diabetes — model may be biased |
| Training accuracy | 79.32% — model learned well from training data |
| Testing accuracy | 70.13% — decent but room for improvement |
| Biggest weakness | 27 diabetic patients (50% of test diabetics) were not detected |

---

## Libraries Used

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
```

---

## Possible Improvements

- Handle class imbalance using oversampling (SMOTE) or class weights
- Apply log transformation to heavily skewed columns like Insulin
- Try other algorithms such as Decision Tree or Random Forest
- Tune the decision boundary threshold below 0.5 to improve recall on diabetic patients

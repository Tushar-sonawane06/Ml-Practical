# Machine Learning Practicals

This README contains the Python code for all 11 Machine Learning
practicals, in order. Outputs have been intentionally omitted.

## Practical 1 -- Feature Engineering

``` python
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Create sample dataset
data = {
    'Age': [20, 21, None, 23, 22],
    'Marks': [75, None, 85, 90, 80],
    'Gender': ['Male', 'Female', 'Male', None, 'Female']
}
df = pd.DataFrame(data)
print("Original Dataset:")
print(df)

# Separate numerical and categorical columns
num_cols = ['Age', 'Marks']
cat_cols = ['Gender']

# Handle missing numerical values
num_imputer = SimpleImputer(strategy='mean')
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# Handle missing categorical values
cat_imputer = SimpleImputer(strategy='most_frequent')
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

# Encode categorical variables
df = pd.get_dummies(df, columns=cat_cols, dtype=int)

# Scale numerical attributes
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

print("\nCleaned and Model-Ready Dataset:")
print(df.round(2))
print("\nMissing Values:")
print(df.isnull().sum())
```

## Practical 2 -- PCA and LDA

``` python
import time
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = []

# Original features
start = time.perf_counter()
model = LogisticRegression(max_iter=5000)
model.fit(X_train_scaled, y_train)
train_time = time.perf_counter() - start
pred = model.predict(X_test_scaled)
results.append([
    "Original Features",
    X_train_scaled.shape[1],
    accuracy_score(y_test, pred),
    train_time
])

# PCA dimensionality reduction
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

start = time.perf_counter()
model_pca = LogisticRegression(max_iter=5000)
model_pca.fit(X_train_pca, y_train)
train_time = time.perf_counter() - start
pred_pca = model_pca.predict(X_test_pca)
results.append([
    "PCA",
    X_train_pca.shape[1],
    accuracy_score(y_test, pred_pca),
    train_time
])

# LDA dimensionality reduction
lda = LinearDiscriminantAnalysis(n_components=1)
X_train_lda = lda.fit_transform(X_train_scaled, y_train)
X_test_lda = lda.transform(X_test_scaled)

start = time.perf_counter()
model_lda = LogisticRegression(max_iter=5000)
model_lda.fit(X_train_lda, y_train)
train_time = time.perf_counter() - start
pred_lda = model_lda.predict(X_test_lda)
results.append([
    "LDA",
    X_train_lda.shape[1],
    accuracy_score(y_test, pred_lda),
    train_time
])

# Display comparison
result_df = pd.DataFrame(
    results,
    columns=["Method", "Features", "Accuracy", "Training Time (s)"]
)
print(result_df.round(4))
print("\nPCA explained variance ratio:",
      round(pca.explained_variance_ratio_.sum(), 4))
```

## Practical 3 -- Multiple Linear Regression

``` python
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the open-access dataset
data = fetch_california_housing()
X = data.data
y = data.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Multiple Linear Regression Results")
print("MSE:", round(mse, 4))
print("RMSE:", round(rmse, 4))
print("R2 Score:", round(r2, 4))
```

## Practical 4 -- Polynomial Regression

``` python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# Load the open-access dataset
data = load_diabetes()
X = data.data[:, [2]]  # BMI feature
y = data.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

degrees = [1, 2, 3, 5, 10]
results = []

# Train models with different degrees
for degree in degrees:
    model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        LinearRegression()
    )
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    results.append([degree, train_mse, test_mse])

# Display results
result_df = pd.DataFrame(
    results,
    columns=["Polynomial Degree", "Training MSE", "Testing MSE"]
)
print(result_df.round(2))

# Plot training and testing errors
plt.plot(degrees, result_df["Training MSE"], marker="o", label="Training MSE")
plt.plot(degrees, result_df["Testing MSE"], marker="s", label="Testing MSE")
plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error")
plt.title("Polynomial Degree vs Prediction Error")
plt.legend()
plt.grid(True)
plt.show()
```

## Practical 5 -- Ridge and Lasso

``` python
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Load the open-access dataset
data = load_diabetes()
X = data.data
y = data.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define regression models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=1.0, max_iter=10000)
}

results = []

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    results.append([
        name,
        mean_squared_error(y_train, train_pred),
        mean_squared_error(y_test, test_pred),
        r2_score(y_test, test_pred),
        np.sum(model.coef_ == 0)
    ])

# Display results
df = pd.DataFrame(results, columns=[
    "Model", "Training MSE", "Testing MSE",
    "Testing R2", "Zero Coefficients"
])
print(df.round(3))
```

## Practical 6 -- Logistic Regression Metrics

``` python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# Create dataset
data = {
    'Study_Hours': [1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10],
    'Attendance': [50, 55, 60, 65, 68, 72, 78, 80, 85, 90, 92, 95],
    'Pass': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)

# Separate features and target
X = df[['Study_Hours', 'Attendance']]
y = df['Pass']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict test results
y_pred = model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nPrecision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))
```

## Practical 7 -- SVM (Linear and RBF)

``` python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Create dataset
data = {
    'Study_Hours': [1, 2, 2, 3, 3, 4, 4, 5, 6, 7, 8, 9],
    'Attendance': [50, 55, 60, 62, 68, 70, 75, 78, 82, 88, 92, 95],
    'Pass': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)

# Separate features and target
X = df[['Study_Hours', 'Attendance']]
y = df['Pass']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create classifiers
linear = SVC(kernel='linear')
rbf = SVC(kernel='rbf')

# Train models
linear.fit(X_train, y_train)
rbf.fit(X_train, y_train)

# Evaluate models
print("Linear SVM Accuracy:", accuracy_score(y_test, linear.predict(X_test)))
print("RBF SVM Accuracy:", accuracy_score(y_test, rbf.predict(X_test)))
```

## Practical 8 -- K-Means and Elbow Method

``` python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Create dataset
data = {
    'Study_Hours': [1, 2, 2, 3, 3, 4, 4, 5, 6, 7, 8, 9],
    'Marks': [35, 40, 42, 48, 50, 55, 60, 65, 72, 80, 88, 92]
}
df = pd.DataFrame(data)

# Select features
X = df[['Study_Hours', 'Marks']]

# Calculate inertia for different K values
inertia = []
for k in range(1, 6):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X)
    inertia.append(round(model.inertia_, 2))

print("K values:", [1, 2, 3, 4, 5])
print("Inertia:", inertia)

# Plot the Elbow Method graph
plt.plot(range(1, 6), inertia, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

# Apply K-Means with K = 3
final_model = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = final_model.fit_predict(X)
print("\nStudent Clusters:")
print(df)
```

## Practical 9 -- DBSCAN

``` python
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Create dataset
data = {
    'Study_Hours': [2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 20],
    'Marks': [40, 42, 45, 48, 52, 55, 58, 60, 65, 68, 72, 10]
}
df = pd.DataFrame(data)

# Standardize features
X = StandardScaler().fit_transform(df[['Study_Hours', 'Marks']])

# Apply DBSCAN
model = DBSCAN(eps=0.7, min_samples=2)
df['Cluster'] = model.fit_predict(X)

# Display clusters
print(df)

# Display noise points
print("\nNoise Students:", list(df[df['Cluster'] == -1].index))
```

## Practical 10 -- Ensemble Learning

``` python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.metrics import accuracy_score

# Create dataset
data = {
    'Study_Hours': [1, 2, 2, 3, 3, 4, 4, 5, 6, 7, 8, 9, 10, 11],
    'Attendance': [50, 55, 60, 65, 68, 72, 75, 78, 82, 88, 92, 95, 96, 98],
    'Pass': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)

# Separate features and target
X = df[['Study_Hours', 'Attendance']]
y = df['Pass']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Initialize models
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(
        n_estimators=50, random_state=42
    ),
    'Boosting': GradientBoostingClassifier(random_state=42)
}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"{name} Accuracy:", accuracy_score(y_test, pred))
```

## Practical 11 -- Q-Learning

``` python
import numpy as np
import random

random.seed(42)  # Fixed seed so output is repeatable

# Initialize Q-table
Q = np.zeros((4, 4))

# Parameters
alpha = 0.1
gamma = 0.9
epsilon = 0.2

# Training
for episode in range(500):
    state = 0
    for step in range(20):
        # Exploration or exploitation
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            action = np.argmax(Q[state])

        # Move towards the goal for action 1
        next_state = min(state + 1, 3) if action == 1 else state

        # Reward function
        reward = 10 if next_state == 3 else -1

        # Update Q-value
        Q[state, action] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )
        state = next_state

# Display Q-table
print("Q-Table:")
print(np.round(Q, 2))

# Optimal policy
print("\nOptimal Policy:")
for s in range(4):
    print(f"State {s} -> Action {np.argmax(Q[s])}")
```

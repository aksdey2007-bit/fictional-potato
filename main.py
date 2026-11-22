import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load the dataset
data = pd.read_csv("crowd_data.csv")

# Explore the dataset
print("First five rows:")
print(data.head())
print("Rows and columns:", data.shape)
print("\nData types:")
print(data.dtypes)
print("\nMissing values:")
print(data.isna().sum())

# Calculate queue statistics
queue = data["queue_count"].to_numpy()
print("\nAverage queue:", np.mean(queue))
print("Median queue:", np.median(queue))

# Convert queue counts into crowd categories
def get_crowd_level(queue_count):
    if queue_count <= 7:
        return "Low"
    elif queue_count < 18:
        return "Medium"
    else:
        return "High"

data["crowd_level"] = data["queue_count"].apply(get_crowd_level)
print("\nQueue count and crowd level:")
print(data[["queue_count", "crowd_level"]].head())

# Select input features and target
X = data[["hour", "day", "event"]]
y = data["crowd_level"]

print("\nInputs (X):")
print(X.head())
print("\nAnswers (y):")
print(y.head())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# Create and train the Decision Tree model
model = DecisionTreeClassifier(
    max_depth=4,
    min_samples_leaf=2,
    random_state=42
)

model.fit(X_train, y_train)

# Predict crowd levels for the test data
predictions = model.predict(X_test)

print("\nPredicted:", predictions)
print("Actual:   ", y_test.to_numpy())

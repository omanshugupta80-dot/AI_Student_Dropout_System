import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

# Load Dataset
df = pd.read_csv(
    "dataset/raw/student_dropout.csv",
    sep=";"
)

# Remove Enrolled students
df = df[df["Target"] != "Enrolled"]

# Select only required columns
df = df[
    [
        "Gender",
        "Age at enrollment",
        "Tuition fees up to date",
        "Scholarship holder",
        "Admission grade",
        "Curricular units 1st sem (approved)",
        "Curricular units 2nd sem (approved)",
        "Target"
    ]
]

# Convert Target
df["Target"] = df["Target"].map({
    "Graduate": 0,
    "Dropout": 1
})

# Features and Target
X = df.drop("Target", axis=1)
y = df["Target"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = GradientBoostingClassifier(
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

print(
    classification_report(
        y_test,
        y_pred
    )
)

# Save Model
joblib.dump(
    model,
    "model/trained_model.pkl"
)

print("New model saved successfully!")
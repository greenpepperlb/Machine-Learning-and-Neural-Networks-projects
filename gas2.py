#Author Laurent Béguelin
#Date January 26 2026
#Gas Code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from catboost import Pool, CatBoostClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from codecarbon import EmissionsTracker

######### Loading data
df = pd.read_csv(
    "/Users/laurent/PycharmProjects/ML Project/data/gas.csv",
    sep=";",
    encoding="utf-8-sig",
    thousands=",",
    low_memory=False
)
# Correction of types
df["AVERAGE HOUSESIZE"] = pd.to_numeric(df["AVERAGE HOUSESIZE"], errors="coerce")
df["THERM JANUARY 2010"] = pd.to_numeric(df["THERM JANUARY 2010"], errors="coerce")
print(df["TOTAL THERMS"].dtype)
#Cleaning
df.dropna(inplace=True)
df = df[df["TOTAL UNITS"] > 0]
df.reset_index(drop=True, inplace=True)

print(f"Original:        67051 rows")
print(f"After cleaning:  {len(df)} rows")
print(f"Dropped:         {67051 - len(df)} rows")
#Building the classes
#### Classes for gas
# Class 1: Gas consumption <= 5000
# Class 2: 5000 < Gas consumption <= 10000
# Class 3: 10000 < Gas consumption <= 15000
# Class 4: 15000 < Gas consumption <= 20000
# Class 5: Gas consumption > 20000
GAS_BINS = [-np.inf, 5000, 10000, 15000, 20000, np.inf]
CLASS_LABELS = [1, 2, 3, 4, 5]
def build_classes(series):
	return pd.cut(series, bins=GAS_BINS, labels=CLASS_LABELS).astype(int)
# Usage
df["TOTAL THERMS"] = build_classes(df["TOTAL THERMS"])
print(df["TOTAL THERMS"].value_counts().sort_index())

#ONE HOT ENCODING
df = pd.get_dummies(df, columns=["COMMUNITY AREA NAME", "BUILDING TYPE"])
print(f"\nDataFrame shape after one-hot encoding: {df.shape}")
print(df.columns.tolist())  # see all 88 column names

############### Model building
X = df. drop (columns=["TOTAL THERMS" ])
Y = df["TOTAL THERMS" ]

##### Definition train, validation and test split
# Step 1: 70% train, 30% temp
X_train, X_temp, Y_train, Y_temp = train_test_split(
    X, Y, test_size=0.30, random_state=42, stratify=Y
)

# Step 2: split 30% temp into 15% val + 15% test
X_val, X_test, Y_val, Y_test = train_test_split(
    X_temp, Y_temp, test_size=0.50, random_state=42, stratify=Y_temp
)

print(f"Train:      {len(X_train)} rows ({len(X_train)/len(X)*100:.1f}%)")
print(f"Validation: {len(X_val)} rows ({len(X_val)/len(X)*100:.1f}%)")
print(f"Test:       {len(X_test)} rows ({len(X_test)/len(X)*100:.1f}%)")
##### Model Training

model = CatBoostClassifier(
    iterations=1193,
    depth=6,
    learning_rate=0.09537086016868294,
    l2_leaf_reg=2.15834389136596,
    colsample_bylevel=0.5404380524747109,
    min_data_in_leaf=45,
    loss_function="MultiClass",
    eval_metric="Accuracy",
    auto_class_weights="Balanced",
    random_seed=42,
    verbose=100
)

#Running the model
tracker = EmissionsTracker()
tracker.start()
model.fit(X_train, Y_train, eval_set=(X_val, Y_val))
# Stop tracking
emissions = tracker.stop()
print(f"Estimated CO2 emissions: {emissions} kg")
##### Model Predictions
Y_pred_val = model.predict(X_val)
Y_pred_test = model.predict(X_test)
#Checking features importance :
feature_importance = pd.Series(
    model.get_feature_importance(data=Pool(X_train, Y_train)),  # ← add this
    index=X_train.columns
).sort_values(ascending=False)
print(feature_importance.head(10))

# Use the confusion matrix in sklearn.metrics
cm = confusion_matrix(Y_test, Y_pred_test, labels=[1, 2, 3, 4, 5])
# ConfusionMatrixDisplay wraps the matrix with axis labels for clean plotting.
disp = ConfusionMatrixDisplay(
confusion_matrix=cm,
display_labels=["Class 1", "Class 2", "Class 3", "Class 4", "Class 5"])
# Create the figure and draw the matrix.
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, colorbar=True, cmap="Blues")
# Add a descriptive title.
ax. set_title("Gas Model - Confusion Matrix (Test Set)")
plt. tight_layout()
plt. savefig("confusion_matrix_gasfinal.png", dpi=150)
plt. show()

##### Visualization Results

print("\n─── Validation Metrics ───")
print(f"Accuracy:  {accuracy_score(Y_val, Y_pred_val)*100:.2f}%")
print(f"F1 Score:  {f1_score(Y_val, Y_pred_val, average='weighted')*100:.4f}%")
print(f"Precision: {precision_score(Y_val, Y_pred_val, average='weighted')*100:.4f}%")
print(f"Recall:    {recall_score(Y_val, Y_pred_val, average='weighted')*100:.4f}%")

print("\n─── Test Metrics ───")
print(f"Accuracy:  {accuracy_score(Y_test, Y_pred_test)*100:.2f}%")
print(f"F1 Score:  {f1_score(Y_test, Y_pred_test, average='weighted')*100:.4f}%")
print(f"Precision: {precision_score(Y_test, Y_pred_test, average='weighted')*100:.4f}%")
print(f"Recall:    {recall_score(Y_test, Y_pred_test, average='weighted')*100:.4f}%")

##### Per-class metrics bar chart
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

classes = [1, 2, 3, 4, 5]
f1_per_class        = f1_score(Y_test, Y_pred_test, average=None, labels=classes)
precision_per_class = precision_score(Y_test, Y_pred_test, average=None, labels=classes)
recall_per_class    = recall_score(Y_test, Y_pred_test, average=None, labels=classes)

# F1 per class
axes[0].bar([f"Class {c}" for c in classes], f1_per_class, color="steelblue")
axes[0].set_title("F1 Score per Class")
axes[0].set_ylim(0, 1)
axes[0].set_ylabel("Score")

# Precision per class
axes[1].bar([f"Class {c}" for c in classes], precision_per_class, color="darkorange")
axes[1].set_title("Precision per Class")
axes[1].set_ylim(0, 1)

# Recall per class
axes[2].bar([f"Class {c}" for c in classes], recall_per_class, color="seagreen")
axes[2].set_title("Recall per Class")
axes[2].set_ylim(0, 1)

plt.suptitle("Gas Model - Metrics per Class (Test Set)", fontsize=13)
plt.tight_layout()
plt.savefig("metrics_per_class_gasfinal.png", dpi=150)
plt.show()

##### Feature importance graph
feature_importance.head(15).sort_values().plot(
    kind="barh", figsize=(10, 6), color="steelblue"
)
plt.title("Top 15 Most Important Features")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance_gasfinal.png", dpi=150)
plt.show()



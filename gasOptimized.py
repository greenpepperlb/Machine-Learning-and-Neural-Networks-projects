# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from catboost import Pool, CatBoostClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


############# Loading data
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
# =============================================================================
# STEP 3 — FEATURE ENGINEERING
# =============================================================================

# Log transform on dominant feature — same reasoning as electricity
df['log_therm_jan']       = np.log1p(df['THERM JANUARY 2010'])

# Therms per unit — intensity rather than raw volume
df['therm_jan_per_unit']  = df['THERM JANUARY 2010'] / (df['TOTAL UNITS'] + 1)

# Total floor space
df['total_surface']       = df['AVERAGE HOUSESIZE'] * df['TOTAL UNITS']

# Surface per person — larger space per person = more heating needed
df['surface_per_person']  = df['total_surface'] / (df['TOTAL POPULATION'] + 1)

# Vacancy rate — empty units still need some heating (pipes etc.)
df['vacancy_rate']        = 1 - df['OCCUPIED UNITS PERCENTAGE']

# Old tall buildings = bad insulation + more heating floors
df['age_x_stories']       = df['AVERAGE BUILDING AGE'] * df['AVERAGE STORIES']

# Active heated floor space
df['surface_x_occupancy'] = df['total_surface'] * df['OCCUPIED UNITS PERCENTAGE']

# Log transforms
df['log_total_surface']   = np.log1p(df['total_surface'])
df['log_population']      = np.log1p(df['TOTAL POPULATION'])

#Building the classes

# Building age era
df['age_era'] = pd.cut(
    df['AVERAGE BUILDING AGE'],
    bins=[0, 30, 60, 80, 200],
    labels=['modern', 'postwar', 'prewar', 'historic']
).astype(str)
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
df = pd.get_dummies(df, columns=["COMMUNITY AREA NAME", "BUILDING TYPE","age_era"])
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
# =============================================================================
# STEP 8 — FEATURE SELECTION
# =============================================================================
selector = CatBoostClassifier(
    iterations=300, depth=6,
    auto_class_weights="Balanced",
    random_seed=42, verbose=0
)
selector.fit(X_train, Y_train)

importances = pd.Series(
    selector.get_feature_importance(),
    index=X_train.columns
).sort_values(ascending=False)

print("\n=== Top 20 features ===")
print(importances.head(20))

# Visualize with engineered features highlighted
top_20_sorted = importances.head(20).sort_values()
engineered = [
    'log_therm_jan', 'therm_jan_per_unit', 'total_surface',
    'surface_per_person', 'vacancy_rate', 'age_x_stories',
    'surface_x_occupancy', 'log_total_surface', 'log_population'
]
colors = ['darkorange' if f in engineered else 'steelblue'
          for f in top_20_sorted.index]

plt.figure(figsize=(10, 7))
plt.barh(top_20_sorted.index, top_20_sorted.values, color=colors, edgecolor="white")
plt.title("Top 20 Features — Blue = Original | Orange = Engineered", fontsize=13)
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance_gas.png", dpi=150)
plt.show()

TOP_N        = 20
top_features = importances.head(TOP_N).index.tolist()
X_train      = X_train[top_features]
X_val        = X_val[top_features]
X_test       = X_test[top_features]

model = CatBoostClassifier(
    loss_function="MultiClass",       # explicit — 5 classes
    eval_metric="Accuracy",           # as required by instructions
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    auto_class_weights="Balanced",    # handles class imbalance
    random_seed=42,                   # reproducibility
    verbose=100                       # prints progress every 100 iterations
)

#Running the model
model.fit(X_train, Y_train, eval_set=(X_val, Y_val))
##### Model Predictions
Y_pred_val = model.predict(X_val)
val_accuracy = accuracy_score(Y_val, Y_pred_val) * 100
print(f"\nValidation Accuracy: {val_accuracy: .2f}%")
#Checking features importance :
feature_importance = pd.Series(
    model.get_feature_importance(data=Pool(X_train, Y_train)),  # ← add this
    index=X_train.columns
).sort_values(ascending=False)
print(feature_importance.head(10))
# evaluate on test sample
Y_pred_test = model.predict(X_test)
test_accuracy = accuracy_score(Y_test, Y_pred_test) * 100
print(f"Test Accuracy: {test_accuracy: .2f}%")

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
plt. savefig("confusion_matrix_gasOptimized.png", dpi=150)
plt. show()

##### Visualization Results

##### Additional Metrics
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
plt.savefig("metrics_per_class_gasOptimized.png", dpi=150)
plt.show()

##### Feature importance graph
feature_importance.head(15).sort_values().plot(
    kind="barh", figsize=(10, 6), color="steelblue"
)
plt.title("Top 15 Most Important Features")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance_gasOptimized.png", dpi=150)
plt.show()

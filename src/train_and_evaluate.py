import pandas as pd
import os

from sklearn.preprocessing import StandardScaler
import numpy as np

from sklearn.model_selection import train_test_split

import shap
import matplotlib.pyplot as plt

import xgboost as xgb
import time

from sklearn.linear_model import LogisticRegression
import time

from imblearn.over_sampling import SMOTE
import pandas as pd

import joblib

file_path = 'data\processed\df_application_train_final.csv'
df = pd.read_csv(file_path)

print("DataFrame loaded successfully. First 5 rows:")
print(df.head())


y = df['TARGET']
X = df.drop(['TARGET', 'SK_ID_CURR'], axis=1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_val: {X_val.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_val: {y_val.shape}")

print("\nTarget distribution in training set:")
print(y_train.value_counts(normalize=True))

print("\nTarget distribution in validation set:")
print(y_val.value_counts(normalize=True))


all_numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

continuous_numerical_cols = []
binary_numerical_cols = []
constant_cols = []

for col in all_numeric_cols:
    unique_vals = X_train[col].dropna().unique()

    if len(unique_vals) == 1:
        constant_cols.append(col)
    elif len(unique_vals) == 2 and 0 in unique_vals and 1 in unique_vals:
        binary_numerical_cols.append(col)
    else:
        continuous_numerical_cols.append(col)

print(f"Identified continuous numerical columns to be scaled: {len(continuous_numerical_cols)} columns")
print(f"Identified binary/one-hot encoded columns (not scaled): {len(binary_numerical_cols)} columns")
print(f"Identified constant columns (not scaled): {len(constant_cols)} columns")

scaler = StandardScaler()

X_train_processed = X_train.copy()
X_val_processed = X_val.copy()

if continuous_numerical_cols:
    X_train_processed[continuous_numerical_cols] = scaler.fit_transform(X_train[continuous_numerical_cols])
    X_val_processed[continuous_numerical_cols] = scaler.transform(X_val[continuous_numerical_cols])
else:
    print("No continuous numerical columns found to scale.")

print("\nNumerical variables (excluding binary/one-hot encoded and constant) scaled successfully.")
print(f"X_train_processed head (scaled continuous numerical features):\n{X_train_processed[continuous_numerical_cols].head() if continuous_numerical_cols else 'No continuous numerical columns to display.'}")
print(f"X_val_processed head (scaled continuous numerical features):\n{X_val_processed[continuous_numerical_cols].head() if continuous_numerical_cols else 'No continuous numerical columns to display.'}")

X_train = X_train_processed
X_val = X_val_processed

# import numpy as np

# corr_matrix = X_train_selected.corr().abs()

# upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.90)]

# print(f"Identified {len(to_drop)} highly correlated features to remove:")
# print(to_drop)

# X_train_selected = X_train_selected.drop(columns=to_drop)
# X_val_selected = X_val_selected.drop(columns=to_drop)

# print(f"\nShape of X_train_selected after multicollinearity check: {X_train_selected.shape}")
# print(f"Shape of X_val_selected after multicollinearity check: {X_val_selected.shape}")

# X_train = X_train_selected
# X_val = X_val_selected

# print("\nFinal selected features after multicollinearity check:")
# print(X_train_selected.columns.tolist())


print("Applying SMOTE to the training data...")

sm = SMOTE(random_state=42)

X_train_smote, y_train_smote = sm.fit_resample(X_train, y_train)

print("SMOTE application complete.")
print(f"Original X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"SMOTE-resampled X_train_smote shape: {X_train_smote.shape}, y_train_smote shape: {y_train_smote.shape}")

print("\nClass distribution after SMOTE:")
print(y_train_smote.value_counts(normalize=True))


print("Training Logistic Regression Model...")
start_time = time.time()

logistic_model = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000, class_weight='balanced')
logistic_model.fit(X_train, y_train)

end_time = time.time()
training_time_lr = end_time - start_time

print(f"Logistic Regression Model trained in {training_time_lr:.2f} seconds.")
print("Logistic Regression Model training complete.")

from sklearn.linear_model import LogisticRegression
import time

print("Training Logistic Regression Model on SMOTE balanced data...")
start_time = time.time()

logistic_model_smote = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000, class_weight='balanced')
logistic_model_smote.fit(X_train_smote, y_train_smote)

end_time = time.time()
training_time_lr_smote = end_time - start_time

print(f"Logistic Regression Model (SMOTE) trained in {training_time_lr_smote:.2f} seconds.")
print("Logistic Regression Model (SMOTE) training complete.")

from sklearn.ensemble import ExtraTreesClassifier
import time

print("Training Extra Trees Classifier Model...")
start_time = time.time()

extra_trees_model = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced') # n_jobs=-1 uses all available cores
extra_trees_model.fit(X_train, y_train)

end_time = time.time()
training_time_et = end_time - start_time

print(f"Extra Trees Classifier Model trained in {training_time_et:.2f} seconds.")
print("Extra Trees Classifier Model training complete.")

from sklearn.ensemble import ExtraTreesClassifier
import time

print("Training Extra Trees Classifier Model on SMOTE balanced data...")
start_time = time.time()

extra_trees_model_smote = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
extra_trees_model_smote.fit(X_train_smote, y_train_smote)

end_time = time.time()
training_time_et_smote = end_time - start_time

print(f"Extra Trees Classifier Model (SMOTE) trained in {training_time_et_smote:.2f} seconds.")
print("Extra Trees Classifier Model (SMOTE) training complete.")


print("Training XGBoost Classifier Model...")
start_time = time.time()

neg_count = y_train.value_counts()[0]
pos_count = y_train.value_counts()[1]
scale_pos_weight_value = neg_count / pos_count

xgboost_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1,
    scale_pos_weight=scale_pos_weight_value
)
xgboost_model.fit(X_train, y_train)

end_time = time.time()
training_time_xgb = end_time - start_time

print(f"XGBoost Classifier Model trained in {training_time_xgb:.2f} seconds.")
print("XGBoost Classifier Model training complete.")

import xgboost as xgb
import time

print("Training XGBoost Classifier Model on SMOTE balanced data...")
start_time = time.time()

xgboost_model_smote = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1,
    scale_pos_weight=1
)
xgboost_model_smote.fit(X_train_smote, y_train_smote)

end_time = time.time()
training_time_xgb_smote = end_time - start_time

print(f"XGBoost Classifier Model (SMOTE) trained in {training_time_xgb_smote:.2f} seconds.")
print("XGBoost Classifier Model (SMOTE) training complete.")

from sklearn.ensemble import RandomForestClassifier
import time

print("Training Random Forest Classifier Model...")
start_time = time.time()

random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
random_forest_model.fit(X_train, y_train)

end_time = time.time()
training_time_rf = end_time - start_time

print(f"Random Forest Classifier Model trained in {training_time_rf:.2f} seconds.")
print("Random Forest Classifier Model training complete.")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_val, y_val, model_name):
    print(f"\n--- Evaluating {model_name} ---")
    
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)

    roc_auc = roc_auc_score(y_val, y_pred_proba)
    pr_auc = average_precision_score(y_val, y_pred_proba)
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred)
    cr = classification_report(y_val, y_pred)

    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    print("\nConfusion Matrix:")
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    print("\nClassification Report:")
    print(cr)
    return {
        'model_name': model_name,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

all_model_metrics = []

metrics_lr = evaluate_model(logistic_model, X_val, y_val, "Logistic Regression (Class Weight)")
all_model_metrics.append(metrics_lr)

metrics_lr_smote = evaluate_model(logistic_model_smote, X_val, y_val, "Logistic Regression (SMOTE)")
all_model_metrics.append(metrics_lr_smote)

metrics_et = evaluate_model(extra_trees_model, X_val, y_val, "Extra Trees (Class Weight)")
all_model_metrics.append(metrics_et)

metrics_et_smote = evaluate_model(extra_trees_model_smote, X_val, y_val, "Extra Trees (SMOTE)")
all_model_metrics.append(metrics_et_smote)

metrics_xgb = evaluate_model(xgboost_model, X_val, y_val, "XGBoost (Class Weight)")
all_model_metrics.append(metrics_xgb)

metrics_xgb_smote = evaluate_model(xgboost_model_smote, X_val, y_val, "XGBoost (SMOTE)")
all_model_metrics.append(metrics_xgb_smote)

metrics_rf = evaluate_model(random_forest_model, X_val, y_val, "Random Forest (Class Weight)")
all_model_metrics.append(metrics_rf)

metrics_df = pd.DataFrame(all_model_metrics)
metrics_df = metrics_df.set_index('model_name')

print("\n--- Model Performance Summary (Updated) ---")
print(metrics_df.round(4))

plt.figure(figsize=(14, 7))
metrics_df[['roc_auc', 'pr_auc', 'accuracy', 'precision', 'recall', 'f1_score']].plot(kind='bar', figsize=(14, 7))
plt.title('Model Performance Comparison Across All Metrics (Updated)')
plt.ylabel('Score')
plt.xticks(rotation=45, ha='right')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

metrics_df = pd.DataFrame(all_model_metrics)
metrics_df = metrics_df.set_index('model_name')

print("\n--- Model Performance Summary ---")
print(metrics_df.round(4))

plt.figure(figsize=(14, 7))
metrics_df[['roc_auc', 'pr_auc', 'accuracy', 'precision', 'recall', 'f1_score']].plot(kind='bar', figsize=(14, 7))
plt.title('Model Performance Comparison Across All Metrics')
plt.ylabel('Score')
plt.xticks(rotation=45, ha='right')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_threshold(model, X_val, y_val, threshold):
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred_threshold = (y_pred_proba >= threshold).astype(int)

    precision = precision_score(y_val, y_pred_threshold)
    recall = recall_score(y_val, y_pred_threshold)
    f1 = f1_score(y_val, y_pred_threshold)

    risky_clients_predicted = np.sum(y_pred_threshold == 1)
    truly_risky_clients_found = np.sum((y_val == 1) & (y_pred_threshold == 1))

    print(f"\n--- Threshold: {threshold:.2f} ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Number of clients identified as risky: {risky_clients_predicted}")
    print(f"Number of truly risky clients found by model: {truly_risky_clients_found}")
    
    return { 'threshold': threshold, 'precision': precision, 'recall': recall, 'f1_score': f1, 'risky_clients_predicted': risky_clients_predicted, 'truly_risky_clients_found': truly_risky_clients_found }

thresholds = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
threshold_results = []

for t in thresholds:
    results = evaluate_threshold(xgboost_model, X_val, y_val, t)
    threshold_results.append(results)


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

threshold_df = pd.DataFrame(threshold_results)

threshold_melted = threshold_df.melt(id_vars=['threshold'], 
                                     value_vars=['precision', 'recall', 'f1_score', 'risky_clients_predicted', 'truly_risky_clients_found'],
                                     var_name='Metric', value_name='Score')

plt.figure(figsize=(20, 10))
sns.barplot(x='threshold', y='Score', hue='Metric', data=threshold_melted, palette='viridis')

plt.title('Performance Metrics and Identified Clients vs. Threshold (Bar Plot)')
plt.xlabel('Threshold')
plt.ylabel('Score / Count')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


shap.initjs()

sample_size = 5000  
X_val_sample = X_val.sample(n=min(len(X_val), sample_size), random_state=42)

print("Explaining model predictions with SHAP...")

explainer = shap.TreeExplainer(xgboost_model)

shap_values = explainer.shap_values(X_val_sample)

print("SHAP values calculated. Generating summary plot...")

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_val_sample, plot_type="bar", show=False)
plt.title('Global Feature Importance (Mean Absolute SHAP Value)')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_val_sample, show=False)
plt.title('SHAP Summary Plot (Feature Impact and Direction)')
plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np

shap_importance = pd.DataFrame({
    'feature': X_val_sample.columns,
    'mean_abs_shap_value': np.abs(shap_values).mean(axis=0),
    'mean_shap_value': shap_values.mean(axis=0)
})
shap_importance = shap_importance.sort_values(by='mean_abs_shap_value', ascending=False).reset_index(drop=True)

print("\nTop 10 Most Important Features based on SHAP (with direction):")
print(shap_importance.head(10))

choice = input("Želite li spremiti naybolji model (XGBoost - class weight) - y/n")
if choice == 'y':
    drive_path = 'model/'
    model_filename = 'xgboost_best_model.joblib'
    full_model_path = os.path.join(drive_path, model_filename)
    os.makedirs(drive_path, exist_ok=True)

    joblib.dump(xgboost_model, full_model_path)
    print(f"Model '{model_filename}' successfully saved to: {full_model_path}")

import pandas as pd
import os

import matplotlib.pyplot as plt
import seaborn as sns

directory = 'data/raw/'
files_to_keep = [
    'application_train.csv',
    'bureau.csv',
    'previous_application.csv',
    'installments_payments.csv'
]

print("Shape of each table:")
for filename in files_to_keep:
    file_path = os.path.join(directory, filename)
    try:
        df = pd.read_csv(file_path)
        print(f"{filename}: {df.shape}")
    except Exception as e:
        print(f"Error reading {filename}: {e}")

file_path = os.path.join(directory, 'application_train.csv')

try:
    df_application_train = pd.read_csv(file_path)
except Exception as e:
    print(f"Error reading application_train.csv: {e}")

file_path = os.path.join(directory, 'bureau.csv')

try:
    df_bureau = pd.read_csv(file_path)
except Exception as e:
    print(f"Error reading bureau.csv: {e}")

file_path = os.path.join(directory, 'previous_application.csv')

try:
    df_previous_application = pd.read_csv(file_path)
except Exception as e:
    print(f"Error reading previous_application.csv: {e}")

file_path = os.path.join(directory, 'installments_payments.csv')

try:
    df_installments_payments = pd.read_csv(file_path)
except Exception as e:
    print(f"Error reading installments_payments.csv: {e}")

df_application_train.info()

selected_columns = [
    'SK_ID_CURR',
    'TARGET',
    'AMT_INCOME_TOTAL',
    'AMT_CREDIT',
    'AMT_ANNUITY',
    'NAME_CONTRACT_TYPE',
    'CODE_GENDER',
    'FLAG_OWN_CAR',
    'FLAG_OWN_REALTY',
    'CNT_CHILDREN',
    'CNT_FAM_MEMBERS',
    'NAME_EDUCATION_TYPE',
    'NAME_FAMILY_STATUS',
    'REGION_POPULATION_RELATIVE',
    'DAYS_BIRTH',
    'DAYS_EMPLOYED',
    'NAME_INCOME_TYPE',
    'NAME_HOUSING_TYPE',
    'OCCUPATION_TYPE',
    'AMT_GOODS_PRICE'
]

df_application_train_filtered = df_application_train[selected_columns].copy()

print("--- Info for filtered application_train.csv ---")
df_application_train_filtered.info()
df_application_train_filtered.head()

df_application_train = df_application_train[selected_columns].copy()

df_application_train['FLAG_OWN_CAR'] = df_application_train['FLAG_OWN_CAR'].map({'Y': 1, 'N': 0})
df_application_train['FLAG_OWN_REALTY'] = df_application_train['FLAG_OWN_REALTY'].map({'Y': 1, 'N': 0})

categorical_cols = df_application_train.select_dtypes(include='object').columns

df_application_train = pd.get_dummies(df_application_train, columns=categorical_cols, dummy_na=False)

print("--- Info for df_application_train after filtering and encoding ---")
df_application_train.info()
print("\n--- Head of df_application_train after filtering and encoding ---")
print(df_application_train.head())

categorical_cols_bureau = df_bureau.select_dtypes(include='object').columns

df_bureau = pd.get_dummies(df_bureau, columns=categorical_cols_bureau, dummy_na=False)

print("--- Info for df_bureau after encoding categorical variables ---")
df_bureau.info()
print("\n--- Head of df_bureau after encoding ---")
print(df_bureau.head())

selected_cols_previous_app = [
    'SK_ID_PREV',
    'SK_ID_CURR',
    'NAME_CONTRACT_STATUS',
    'AMT_APPLICATION',
    'AMT_CREDIT'
]

df_previous_application = df_previous_application[selected_cols_previous_app].copy()

print("--- Info for df_previous_application after filtering ---")
df_previous_application.info()
print("\n--- Head of df_previous_application after filtering ---")
print(df_previous_application.head())

categorical_col_prev_app = ['NAME_CONTRACT_STATUS']

df_previous_application = pd.get_dummies(df_previous_application, columns=categorical_col_prev_app, dummy_na=False)

print("--- Info for df_previous_application after encoding categorical variable ---")
df_previous_application.info()
print("\n--- Head of df_previous_application after encoding ---")
print(df_previous_application.head())

median_amt_goods_price = df_application_train['AMT_GOODS_PRICE'].median()
df_application_train['AMT_GOODS_PRICE'].fillna(median_amt_goods_price, inplace=True)
print(f"Missing values in 'AMT_GOODS_PRICE' imputed with median: {median_amt_goods_price}")

median_amt_annuity = df_application_train['AMT_ANNUITY'].median()
df_application_train['AMT_ANNUITY'].fillna(median_amt_annuity, inplace=True)
print(f"Missing values in 'AMT_ANNUITY' imputed with median: {median_amt_annuity}")

median_cnt_fam_members = df_application_train['CNT_FAM_MEMBERS'].median()
df_application_train['CNT_FAM_MEMBERS'].fillna(median_cnt_fam_members, inplace=True)
print(f"Missing values in 'CNT_FAM_MEMBERS' imputed with median: {median_cnt_fam_members}")

missing_values_after_imputation = df_application_train.isnull().sum()
missing_percent_after_imputation = 100 * df_application_train.isnull().sum() / len(df_application_train)

missing_df_after_imputation = pd.DataFrame({'missing_count': missing_values_after_imputation, 'missing_percent': missing_percent_after_imputation})
missing_df_after_imputation = missing_df_after_imputation[missing_df_after_imputation['missing_count'] > 0].sort_values(by='missing_percent', ascending=False)

print("\nMissing values in df_application_train after all specified imputations:")
print(missing_df_after_imputation)

median_amt_credit_sum = df_bureau['AMT_CREDIT_SUM'].median()
df_bureau['AMT_CREDIT_SUM'].fillna(median_amt_credit_sum, inplace=True)
print(f"Missing values in 'AMT_CREDIT_SUM' imputed with median: {median_amt_credit_sum}")
print('\nMissing values in df_bureau after imputing AMT_CREDIT_SUM:')
missing_values_bureau_after = df_bureau.isnull().sum()
missing_percent_bureau_after = 100 * df_bureau.isnull().sum() / len(df_bureau)
missing_df_bureau_after = pd.DataFrame({'missing_count': missing_values_bureau_after, 'missing_percent': missing_percent_bureau_after})
missing_df_bureau_after = missing_df_bureau_after[missing_df_bureau_after['missing_count'] > 0].sort_values(by='missing_percent', ascending=False)
print(missing_df_bureau_after)

print("--- Applying imputation strategies to df_bureau based on semantic analysis ---")

df_bureau['AMT_ANNUITY'].fillna(0, inplace=True)
print("Missing values in 'AMT_ANNUITY' (bureau) imputed with 0.")

df_bureau['AMT_CREDIT_MAX_OVERDUE'].fillna(0, inplace=True)
print("Missing values in 'AMT_CREDIT_MAX_OVERDUE' imputed with 0.")

df_bureau['AMT_CREDIT_SUM_LIMIT'].fillna(0, inplace=True)
print("Missing values in 'AMT_CREDIT_SUM_LIMIT' imputed with 0.")

df_bureau['AMT_CREDIT_SUM_DEBT'].fillna(0, inplace=True)
print("Missing values in 'AMT_CREDIT_SUM_DEBT' imputed with 0.")

df_bureau['DAYS_ENDDATE_FACT'].fillna(365243, inplace=True)
print("Missing values in 'DAYS_ENDDATE_FACT' imputed with 365243 (meaning 'not yet ended' or 'active').")

median_days_credit_enddate = df_bureau['DAYS_CREDIT_ENDDATE'].median()
df_bureau['DAYS_CREDIT_ENDDATE'].fillna(median_days_credit_enddate, inplace=True)
print(f"Missing values in 'DAYS_CREDIT_ENDDATE' imputed with median: {median_days_credit_enddate}.")

print("\nMissing values in df_bureau after all imputations:")
missing_values_bureau_final_after_imputation = df_bureau.isnull().sum()
missing_percent_bureau_final_after_imputation = 100 * df_bureau.isnull().sum() / len(df_bureau)
missing_df_bureau_final_after_imputation = pd.DataFrame({'missing_count': missing_values_bureau_final_after_imputation, 'missing_percent': missing_percent_bureau_final_after_imputation})
missing_df_bureau_final_after_imputation = missing_df_bureau_final_after_imputation[missing_df_bureau_final_after_imputation['missing_count'] > 0].sort_values(by='missing_percent', ascending=False)
print(missing_df_bureau_final_after_imputation)

missing_values_prev_app = df_previous_application.isnull().sum()
missing_percent_prev_app = 100 * df_previous_application.isnull().sum() / len(df_previous_application)

missing_df_prev_app = pd.DataFrame({'missing_count': missing_values_prev_app, 'missing_percent': missing_percent_prev_app})
missing_df_prev_app = missing_df_prev_app[missing_df_prev_app['missing_count'] > 0].sort_values(by='missing_percent', ascending=False)

print("Missing values in df_previous_application:")
print(missing_df_prev_app)

if 'AMT_CREDIT' in missing_df_prev_app.index and missing_df_prev_app.loc['AMT_CREDIT', 'missing_count'] > 0:
    median_amt_credit_prev_app = df_previous_application['AMT_CREDIT'].median()
    df_previous_application['AMT_CREDIT'] = df_previous_application['AMT_CREDIT'].fillna(median_amt_credit_prev_app)
    print(f"\nMissing value in 'AMT_CREDIT' (df_previous_application) imputed with median: {median_amt_credit_prev_app}.")

print("\nMissing values in df_previous_application after imputation:")
missing_values_prev_app_after_imputation = df_previous_application.isnull().sum()
missing_percent_prev_app_after_imputation = 100 * df_previous_application.isnull().sum() / len(df_previous_application)
missing_df_prev_app_after_imputation = pd.DataFrame({'missing_count': missing_values_prev_app_after_imputation, 'missing_percent': missing_percent_prev_app_after_imputation})
missing_df_prev_app_after_imputation = missing_df_prev_app_after_imputation[missing_df_prev_app_after_imputation['missing_count'] > 0].sort_values(by='missing_percent', ascending=False)
print(missing_df_prev_app_after_imputation)

median_days_entry_payment = df_installments_payments['DAYS_ENTRY_PAYMENT'].median()
df_installments_payments['DAYS_ENTRY_PAYMENT'] = df_installments_payments['DAYS_ENTRY_PAYMENT'].fillna(median_days_entry_payment)
print(f"Missing values in 'DAYS_ENTRY_PAYMENT' imputed with median: {median_days_entry_payment}")

median_amt_payment = df_installments_payments['AMT_PAYMENT'].median()
df_installments_payments['AMT_PAYMENT'] = df_installments_payments['AMT_PAYMENT'].fillna(median_amt_payment)
print(f"Missing values in 'AMT_PAYMENT' imputed with median: {median_amt_payment}")

print("\nMissing values in df_installments_payments after imputation:")
missing_values_installments_after_imputation = df_installments_payments.isnull().sum()
missing_percent_installments_after_imputation = 100 * df_installments_payments.isnull().sum() / len(df_installments_payments)
missing_df_installments_after_imputation = pd.DataFrame({'missing_count': missing_values_installments_after_imputation, 'missing_percent': missing_percent_installments_after_imputation})
missing_df_installments_after_imputation = missing_df_installments_after_imputation[missing_df_installments_after_imputation['missing_count'] > 0].sort_values(by='missing_percent', ascending=False)
print(missing_df_installments_after_imputation)

upper_bound_children_cap = df_application_train['CNT_CHILDREN'].quantile(0.999)
df_application_train['CNT_CHILDREN'] = df_application_train['CNT_CHILDREN'].clip(upper=upper_bound_children_cap)
print(f"'CNT_CHILDREN' has been capped at the 99.9th percentile: {int(upper_bound_children_cap)}")
print(f"New maximum for 'CNT_CHILDREN': {int(df_application_train['CNT_CHILDREN'].max())}")
print()

upper_bound_income_cap = df_application_train['AMT_INCOME_TOTAL'].quantile(0.999)
df_application_train['AMT_INCOME_TOTAL'] = df_application_train['AMT_INCOME_TOTAL'].clip(upper=upper_bound_income_cap)
print(f"'AMT_INCOME_TOTAL' has been capped at the 99.9th percentile: {int(upper_bound_income_cap)}")
print(f"New maximum for 'AMT_INCOME_TOTAL': {int(df_application_train['AMT_INCOME_TOTAL'].max())}")

import numpy as np

print("Original descriptive statistics for DAYS_EMPLOYED (before transformation):")
df_application_train['DAYS_EMPLOYED'].describe()

df_application_train['UNEMPLOYED_RETIRED'] = (df_application_train['DAYS_EMPLOYED'] == 365243).astype(int)
print("\nCreated 'UNEMPLOYED_RETIRED' feature: 1 if DAYS_EMPLOYED was 365243, 0 otherwise.")
print(f"Number of 'UNEMPLOYED_RETIRED' individuals: {df_application_train['UNEMPLOYED_RETIRED'].sum()}")

df_application_train.loc[df_application_train['DAYS_EMPLOYED'] == 365243, 'DAYS_EMPLOYED'] = 0
print("Set DAYS_EMPLOYED to 0 for 'UNEMPLOYED_RETIRED' individuals.")

df_application_train['DAYS_EMPLOYED'] = df_application_train['DAYS_EMPLOYED'].abs()
print("Converted remaining DAYS_EMPLOYED values to absolute (positive) days.")

print("\nDescriptive statistics for DAYS_EMPLOYED (after transformation and handling sentinel value):")
df_application_train['DAYS_EMPLOYED'].describe()

print("\nValue counts for DAYS_EMPLOYED (first 10 values, sorted ascending, after transformation):")
df_application_train['DAYS_EMPLOYED'].value_counts(dropna=False).sort_index().head(10)

print("\nValue counts for DAYS_EMPLOYED around 0 (after transformation):")
df_application_train[ (df_application_train['DAYS_EMPLOYED'] >= 0) & (df_application_train['DAYS_EMPLOYED'] <= 1) ]['DAYS_EMPLOYED'].value_counts(dropna=False).sort_index()

print("\nMaximum value of DAYS_EMPLOYED (after handling):")
print(df_application_train['DAYS_EMPLOYED'].max())

print("\nMinimum value of DAYS_EMPLOYED (after handling):")
print(df_application_train['DAYS_EMPLOYED'].min())

upper_bound_credit_max_overdue_cap = df_bureau['AMT_CREDIT_MAX_OVERDUE'].quantile(0.999)
df_bureau['AMT_CREDIT_MAX_OVERDUE'] = df_bureau['AMT_CREDIT_MAX_OVERDUE'].clip(upper=upper_bound_credit_max_overdue_cap)
print(f"'AMT_CREDIT_MAX_OVERDUE' has been capped at the 99.9th percentile: {upper_bound_credit_max_overdue_cap}")
print(f"New maximum for 'AMT_CREDIT_MAX_OVERDUE': {df_bureau['AMT_CREDIT_MAX_OVERDUE'].max()}")

upper_bound_credit_sum_cap = df_bureau['AMT_CREDIT_SUM'].quantile(0.999)
df_bureau['AMT_CREDIT_SUM'] = df_bureau['AMT_CREDIT_SUM'].clip(upper=upper_bound_credit_sum_cap)
print(f"'AMT_CREDIT_SUM' has been capped at the 99.9th percentile: {upper_bound_credit_sum_cap}")
print(f"New maximum for 'AMT_CREDIT_SUM': {df_bureau['AMT_CREDIT_SUM'].max()}")

upper_bound_credit_sum_debt_cap = df_bureau['AMT_CREDIT_SUM_DEBT'].quantile(0.999)
df_bureau['AMT_CREDIT_SUM_DEBT'] = df_bureau['AMT_CREDIT_SUM_DEBT'].clip(upper=upper_bound_credit_sum_debt_cap)
print(f"\n'AMT_CREDIT_SUM_DEBT' has been capped at the 99.9th percentile: {upper_bound_credit_sum_debt_cap}")
print(f"New maximum for 'AMT_CREDIT_SUM_DEBT': {df_bureau['AMT_CREDIT_SUM_DEBT'].max()}")

upper_bound_credit_sum_limit_cap = df_bureau['AMT_CREDIT_SUM_LIMIT'].quantile(0.999)
df_bureau['AMT_CREDIT_SUM_LIMIT'] = df_bureau['AMT_CREDIT_SUM_LIMIT'].clip(upper=upper_bound_credit_sum_limit_cap)
print(f"\n'AMT_CREDIT_SUM_LIMIT' has been capped at the 99.9th percentile: {upper_bound_credit_sum_limit_cap}")
print(f"New maximum for 'AMT_CREDIT_SUM_LIMIT': {df_bureau['AMT_CREDIT_SUM_LIMIT'].max()}")

upper_bound_credit_sum_overdue_cap = df_bureau['AMT_CREDIT_SUM_OVERDUE'].quantile(0.999)
df_bureau['AMT_CREDIT_SUM_OVERDUE'] = df_bureau['AMT_CREDIT_SUM_OVERDUE'].clip(upper=upper_bound_credit_sum_overdue_cap)
print(f"\n'AMT_CREDIT_SUM_OVERDUE' has been capped at the 99.9th percentile: {upper_bound_credit_sum_overdue_cap}")
print(f"New maximum for 'AMT_CREDIT_SUM_OVERDUE': {df_bureau['AMT_CREDIT_SUM_OVERDUE'].max()}")

lower_bound_days_credit_update_cap = df_bureau['DAYS_CREDIT_UPDATE'].quantile(0.001)
upper_bound_days_credit_update_cap = df_bureau['DAYS_CREDIT_UPDATE'].quantile(0.999)
df_bureau['DAYS_CREDIT_UPDATE'] = df_bureau['DAYS_CREDIT_UPDATE'].clip(lower=lower_bound_days_credit_update_cap, upper=upper_bound_days_credit_update_cap)
print(f"'DAYS_CREDIT_UPDATE' has been capped at the 0.1th percentile: {lower_bound_days_credit_update_cap} and 99.9th percentile: {upper_bound_days_credit_update_cap}")
print(f"New minimum for 'DAYS_CREDIT_UPDATE': {df_bureau['DAYS_CREDIT_UPDATE'].min()}")
print(f"New maximum for 'DAYS_CREDIT_UPDATE': {df_bureau['DAYS_CREDIT_UPDATE'].max()}")

upper_bound_annuity_cap = df_bureau['AMT_ANNUITY'].quantile(0.999)
df_bureau['AMT_ANNUITY'] = df_bureau['AMT_ANNUITY'].clip(upper=upper_bound_annuity_cap)
print(f"'AMT_ANNUITY' has been capped at the 99.9th percentile: {upper_bound_annuity_cap}")
print(f"New maximum for 'AMT_ANNUITY': {df_bureau['AMT_ANNUITY'].max()}")

df_application_train['CREDIT_INCOME_RATIO'] = df_application_train['AMT_CREDIT'] / df_application_train['AMT_INCOME_TOTAL']

df_application_train.to_csv('data\processed\df_application_train_processed.csv', index=False)
df_bureau.to_csv('data\processed\df_bureau_processed.csv', index=False)
df_previous_application.to_csv('data\processed\df_previous_application_processed.csv', index=False)
df_installments_payments.to_csv('data\processed\df_installments_payments_processed.csv', index=False)
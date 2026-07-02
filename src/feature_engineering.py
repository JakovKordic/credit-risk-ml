import pandas as pd
import os

path = 'data/processed/'

files_to_keep = [
    'df_application_train_processed.csv',
    'df_bureau_processed.csv',
    'df_previous_application_processed.csv',
    'df_installments_payments_processed.csv'
]

loaded_dataframes = {}
print(f"Pretraživanje direktorija: {path} za CSV dataframeove...")
for filename in os.listdir(path):
    file_path = os.path.join(path, filename)
    df_name = os.path.splitext(filename)[0]
    if filename.endswith('.csv'):
        try:
            loaded_dataframes[df_name] = pd.read_csv(file_path)
            print(f"Učitan dataframe '{df_name}' (CSV).")
        except Exception as e:
            print(f"Greška pri učitavanju CSV datoteke '{filename}': {e}")
if not loaded_dataframes:
    print(f"Nijedna datoteka nije pronađena ili učitana u direktoriju: {path}.")
else:
    print("\nUčitani dataframeovi:")
    for name, df in loaded_dataframes.items():
        print(f"- {name}: {df.shape[0]} redaka, {df.shape[1]} stupaca")

df_app_train = loaded_dataframes['df_application_train_processed'].copy()

df_app_train['ANNUITY_INCOME_RATIO'] = df_app_train['AMT_ANNUITY'] / df_app_train['AMT_INCOME_TOTAL']

print("Novi 'ANNUITY_INCOME_RATIO' feature je dodan u 'df_application_train_processed'.")
print("Prvih 5 redaka novog feature-a:")
print(df_app_train[['AMT_ANNUITY', 'AMT_INCOME_TOTAL', 'ANNUITY_INCOME_RATIO']].head())

loaded_dataframes['df_application_train_processed'] = df_app_train

df_app_train = loaded_dataframes['df_application_train_processed'].copy()

df_app_train['CREDIT_ANNUITY_RATIO'] = df_app_train['AMT_CREDIT'] / df_app_train['AMT_ANNUITY']

print("Novi 'CREDIT_ANNUITY_RATIO' feature je dodan u 'df_application_train_processed'.")
print("Prvih 5 redaka novog feature-a:")
print(df_app_train[['AMT_CREDIT', 'AMT_ANNUITY', 'CREDIT_ANNUITY_RATIO']].head())

loaded_dataframes['df_application_train_processed'] = df_app_train

df_app_train = loaded_dataframes['df_application_train_processed'].copy()

df_app_train['INCOME_PER_FAM_MEMBERS'] = df_app_train['AMT_INCOME_TOTAL'] / df_app_train['CNT_FAM_MEMBERS']

print("Novi 'INCOME_PER_FAM_MEMBERS' feature je dodan u 'df_application_train_processed'.")
print("Prvih 5 redaka novog feature-a:")
print(df_app_train[['AMT_INCOME_TOTAL', 'CNT_FAM_MEMBERS', 'INCOME_PER_FAM_MEMBERS']].head())

loaded_dataframes['df_application_train_processed'] = df_app_train

df_app_train = loaded_dataframes['df_application_train_processed'].copy()

df_app_train['DAYS_EMPLOYED_RATIO'] = df_app_train['DAYS_EMPLOYED'] / df_app_train['DAYS_BIRTH']

print("Novi 'DAYS_EMPLOYED_RATIO' feature je dodan u 'df_application_train_processed'.")
print("Prvih 5 redaka novog feature-a:")
print(df_app_train[['DAYS_EMPLOYED', 'DAYS_BIRTH', 'DAYS_EMPLOYED_RATIO']].head())

loaded_dataframes['df_application_train_processed'] = df_app_train

df_app_train = loaded_dataframes['df_application_train_processed'].copy()

df_app_train['ADULTS_IN_FAMILY'] = df_app_train['CNT_FAM_MEMBERS'] - df_app_train['CNT_CHILDREN']

df_app_train['ADULTS_IN_FAMILY'] = df_app_train['ADULTS_IN_FAMILY'].apply(lambda x: max(1, x))

print("Novi 'ADULTS_IN_FAMILY' feature je dodan u 'df_application_train_processed'.")
print("Prvih 5 redaka novog feature-a:")
print(df_app_train[['CNT_FAM_MEMBERS', 'CNT_CHILDREN', 'ADULTS_IN_FAMILY']].head())

loaded_dataframes['df_application_train_processed'] = df_app_train

import numpy as np

df_bureau = loaded_dataframes['df_bureau_processed'].copy()

print("--- Feature Engineering za 'df_bureau_processed' ---")

df_bureau['CREDIT_ACTIVE_DAYS'] = df_bureau['DAYS_CREDIT_ENDDATE'] - df_bureau['DAYS_CREDIT']

df_bureau['CREDIT_DAYS_OVERLAP'] = df_bureau['DAYS_CREDIT_ENDDATE'] - df_bureau['DAYS_CREDIT_UPDATE']

zero_credit_sum_count = (df_bureau['AMT_CREDIT_SUM'] == 0).sum()
print(f"Broj redaka gdje je 'AMT_CREDIT_SUM' (nazivnik za omjere) jednak 0: {zero_credit_sum_count}")

df_bureau['AMT_CREDIT_SUM_RATIO'] = np.where(
    df_bureau['AMT_CREDIT_SUM'] == 0,0,
    df_bureau['AMT_CREDIT_SUM_DEBT'] / df_bureau['AMT_CREDIT_SUM']
)

df_bureau['AMT_CREDIT_LIMIT_RATIO'] = np.where(
    df_bureau['AMT_CREDIT_SUM'] == 0,0,
    df_bureau['AMT_CREDIT_SUM_LIMIT'] / df_bureau['AMT_CREDIT_SUM']
)

loaded_dataframes['df_bureau_processed'] = df_bureau

print("Nove značajke dodane u 'df_bureau_processed'.")
print("Oblik 'df_bureau_processed' nakon feature engineeringa:", loaded_dataframes['df_bureau_processed'].shape)
print("Prvih 5 redaka s novim značajkama:")
loaded_dataframes['df_bureau_processed'][[ 'CREDIT_ACTIVE_DAYS', 'CREDIT_DAYS_OVERLAP', 'AMT_CREDIT_SUM_RATIO', 'AMT_CREDIT_LIMIT_RATIO']].head()

import numpy as np

df_prev_app = loaded_dataframes['df_previous_application_processed'].copy()

print("--- Feature Engineering za 'df_previous_application_processed' ---")

zero_application_amount_count = (df_prev_app['AMT_APPLICATION'] == 0).sum()
print(f"Broj redaka gdje je 'AMT_APPLICATION' (nazivnik za omjer) jednak 0: {zero_application_amount_count}")

df_prev_app['AMT_CREDIT_TO_APPLICATION_RATIO'] = np.where(
    df_prev_app['AMT_APPLICATION'] == 0,0,
    df_prev_app['AMT_CREDIT'] / df_prev_app['AMT_APPLICATION']
)

df_prev_app['AMT_CREDIT_DIFF_APPLICATION'] = df_prev_app['AMT_CREDIT'] - df_prev_app['AMT_APPLICATION']

loaded_dataframes['df_previous_application_processed'] = df_prev_app

print("Novi 'AMT_CREDIT_TO_APPLICATION_RATIO' i 'AMT_CREDIT_DIFF_APPLICATION' featurei su dodani u 'df_previous_application_processed'.")
print("Oblik 'df_previous_application_processed' nakon feature engineeringa:", loaded_dataframes['df_previous_application_processed'].shape)
print("Prvih 5 redaka s novim značajkama:")
loaded_dataframes['df_previous_application_processed'][[ 'AMT_APPLICATION', 'AMT_CREDIT', 'AMT_CREDIT_TO_APPLICATION_RATIO', 'AMT_CREDIT_DIFF_APPLICATION']].head()

import numpy as np

df_inst_pay = loaded_dataframes['df_installments_payments_processed'].copy()

print("--- Feature Engineering za 'df_installments_payments_processed' ---")

df_inst_pay['DAYS_LATE_PAYMENT'] = (df_inst_pay['DAYS_ENTRY_PAYMENT'] - df_inst_pay['DAYS_INSTALMENT']).apply(lambda x: max(0, x))

df_inst_pay['DAYS_EARLY_PAYMENT'] = (df_inst_pay['DAYS_INSTALMENT'] - df_inst_pay['DAYS_ENTRY_PAYMENT']).apply(lambda x: max(0, x))

df_inst_pay['AMT_DIFFERENCE'] = df_inst_pay['AMT_PAYMENT'] - df_inst_pay['AMT_INSTALMENT']

df_inst_pay['PAYMENT_RATIO'] = np.where(
    df_inst_pay['AMT_INSTALMENT'] == 0,0,
    df_inst_pay['AMT_PAYMENT'] / df_inst_pay['AMT_INSTALMENT']
)

loaded_dataframes['df_installments_payments_processed'] = df_inst_pay

print("Nove značajke dodane u 'df_installments_payments_processed'.")
print("Oblik 'df_installments_payments_processed' nakon feature engineeringa:", loaded_dataframes['df_installments_payments_processed'].shape)
print("Prvih 5 redaka s novim značajkama:")
loaded_dataframes['df_installments_payments_processed'][[ 'DAYS_INSTALMENT', 'DAYS_ENTRY_PAYMENT', 'DAYS_LATE_PAYMENT', 'DAYS_EARLY_PAYMENT', 'AMT_INSTALMENT', 'AMT_PAYMENT', 'AMT_DIFFERENCE', 'PAYMENT_RATIO']].head()

import pandas as pd
import numpy as np

print("--- Agregacija značajki iz df_bureau_processed ---")
df_bureau = loaded_dataframes['df_bureau_processed'].copy()

bureau_numeric_cols = [
    'DAYS_CREDIT', 'CREDIT_DAY_OVERDUE', 'DAYS_CREDIT_ENDDATE',
    'DAYS_ENDDATE_FACT', 'AMT_CREDIT_MAX_OVERDUE', 'CNT_CREDIT_PROLONG',
    'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT',
    'AMT_CREDIT_SUM_OVERDUE', 'DAYS_CREDIT_UPDATE', 'AMT_ANNUITY'
]

bureau_ohe_cols = [col for col in df_bureau.columns if col.startswith('BUREAU_')]

bureau_agg_dict = {}
for col in bureau_numeric_cols:
    bureau_agg_dict[col] = ['min', 'max', 'mean', 'sum', 'count']
for col in bureau_ohe_cols:
    bureau_agg_dict[col] = ['mean', 'sum']

bureau_agg = df_bureau.groupby('SK_ID_CURR').agg(bureau_agg_dict)

bureau_agg.columns = ['_'.join(col).strip().upper() + '_BUREAU' for col in bureau_agg.columns.values]

loaded_dataframes['df_bureau_agg'] = bureau_agg
print(f"Agregirani feature-i iz 'df_bureau_processed' su pripremljeni. Oblik: {bureau_agg.shape}")

print("\n--- Agregacija značajki iz df_previous_application_processed ---")
df_prev_app = loaded_dataframes['df_previous_application_processed'].copy()

prev_app_numeric_cols = [
    'AMT_APPLICATION', 'AMT_CREDIT'
]

prev_app_ohe_cols = [col for col in df_prev_app.columns if col.startswith('NAME_CONTRACT_STATUS_')]

prev_app_agg_dict = {}
for col in prev_app_numeric_cols:
    prev_app_agg_dict[col] = ['min', 'max', 'mean', 'sum', 'count']
for col in prev_app_ohe_cols:
    prev_app_agg_dict[col] = ['mean', 'sum']

prev_app_agg = df_prev_app.groupby('SK_ID_CURR').agg(prev_app_agg_dict)

prev_app_agg.columns = ['_'.join(col).strip().upper() + '_PREV_APP' for col in prev_app_agg.columns.values]

loaded_dataframes['df_previous_application_agg'] = prev_app_agg
print(f"Agregirani feature-i iz 'df_previous_application_processed' su pripremljeni. Oblik: {prev_app_agg.shape}")

print("\n--- Agregacija značajki iz df_installments_payments_processed ---")
df_inst_pay = loaded_dataframes['df_installments_payments_processed'].copy()

inst_pay_numeric_cols = [
    'NUM_INSTALMENT_VERSION', 'NUM_INSTALMENT_NUMBER', 'DAYS_INSTALMENT',
    'DAYS_ENTRY_PAYMENT', 'AMT_INSTALMENT', 'AMT_PAYMENT'
]

inst_pay_agg_dict = {}
for col in inst_pay_numeric_cols:
    inst_pay_agg_dict[col] = ['min', 'max', 'mean', 'sum', 'count']

inst_pay_agg = df_inst_pay.groupby('SK_ID_CURR').agg(inst_pay_agg_dict)

inst_pay_agg.columns = ['_'.join(col).strip().upper() + '_INST_PAY' for col in inst_pay_agg.columns.values]

loaded_dataframes['df_installments_payments_agg'] = inst_pay_agg
print(f"Agregirani feature-i iz 'df_installments_payments_processed' su pripremljeni. Oblik: {inst_pay_agg.shape}")

print("\nSve agregirane tablice su pripremljene i spremljene u 'loaded_dataframes' za kasnije spajanje.")

import pandas as pd

df_app_train = loaded_dataframes['df_application_train_processed'].copy()

bureau_agg = loaded_dataframes['df_bureau_agg']
prev_app_agg = loaded_dataframes['df_previous_application_agg']
inst_pay_agg = loaded_dataframes['df_installments_payments_agg']

print("--- Spajanje agregiranih značajki na df_application_train_processed ---")

df_app_train = df_app_train.merge(bureau_agg, on='SK_ID_CURR', how='left')
print(f"Nakon spajanja 'df_bureau_agg': {df_app_train.shape}")

df_app_train = df_app_train.merge(prev_app_agg, on='SK_ID_CURR', how='left')
print(f"Nakon spajanja 'df_previous_application_agg': {df_app_train.shape}")

df_app_train = df_app_train.merge(inst_pay_agg, on='SK_ID_CURR', how='left')
print(f"Nakon spajanja 'df_installments_payments_agg': {df_app_train.shape}")

loaded_dataframes['df_application_train_processed'] = df_app_train

print("\nFinalni oblik 'df_application_train_processed' nakon svih spajanja:")
print(loaded_dataframes['df_application_train_processed'].shape)
print("Prvih 5 redaka s novim značajkama:")
print(loaded_dataframes['df_application_train_processed'].head())

df_app_train_final = loaded_dataframes['df_application_train_processed']

missing_values = df_app_train_final.isnull().sum()

missing_values = missing_values[missing_values > 0]

missing_percent = 100 * missing_values / len(df_app_train_final)

missing_df = pd.DataFrame({'Missing Count': missing_values, 'Missing Percent': missing_percent})

missing_df = missing_df.sort_values(by='Missing Percent', ascending=False)

print("Nedostajuće vrijednosti u 'df_application_train_processed' (samo stupci s > 0 nedostajućih vrijednosti):")


df_app_train_final = loaded_dataframes['df_application_train_processed']

initial_nan_count = df_app_train_final.isnull().sum().sum()
df_app_train_final = df_app_train_final.fillna(0)
final_nan_count = df_app_train_final.isnull().sum().sum()

print(f"Broj NaN vrijednosti prije imputacije: {initial_nan_count}")
print(f"Broj NaN vrijednosti nakon imputacije s 0: {final_nan_count}")

loaded_dataframes['df_application_train_processed'] = df_app_train_final

print("\nPonovna provjera nedostajućih vrijednosti:")
missing_values_after = df_app_train_final.isnull().sum()
missing_values_after = missing_values_after[missing_values_after > 0]

if not missing_values_after.empty:
    print("Još uvijek postoje nedostajuće vrijednosti:")
else:
    print("Sve nedostajuće vrijednosti su uspješno popunjene.")

df_app_train_final = loaded_dataframes['df_application_train_processed']

output_path = os.path.join(path, 'df_application_train_final.csv')

df_app_train_final.to_csv(output_path, index=False)

print(f"Finalni DataFrame spremljen je na: {output_path}")
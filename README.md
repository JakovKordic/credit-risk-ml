# Credit Risk Scoring pomoću ML modela
Cilj: Razviti ML pipeline koji na tenelju podataka o klijentuna predviđa rizik neurednog otplaćivanja kredita

# Objašnjenja tablica i ključeva za spajanje
1. Tablica application_train.csv predstavlja glavnu tablicu podataka o trenutnim kreditnim aplikacijama klijenata. Svaki redak u ovoj tablici odgovara jednoj kreditnoj prijavi, a stupci opisuju različite karakteristike podnositelja zahtjeva i same aplikacije. Najvažniji stupci su: 
*	SK_ID_CURR — ID trenutnog klijenta/aplikacije
*	TARGET — ciljna varijabla (1 - poteškoće, 0 - bez poteškoća)
*	AMT_INCOME_TOTAL — prihod klijenta
*	AMT_CREDIT — iznos kredita
*	AMT_ANNUITY — anuitet
*	demografski i financijski podaci klijenta
2. Tablica bureau.csv sadrži informacije o prošlim kreditima koje su klijenti Home Credita imali u drugim financijskim institucijama. Svaki redak u ovoj tablici predstavlja jedan kredit iz povijesti klijenta. Jedan klijent može imati više zapisa u ovoj tablici.
Veza s glavnom tablicom:
application_train.SK_ID_CURR = bureau.SK_ID_CURR
3. Tablica previous_application.csv sadrži podatke o prethodnim kreditnim aplikacijama koje su klijenti Home Credita podnijeli izravno u Home Creditu. Svaki redak u ovoj tablici odgovara jednoj prethodnoj aplikaciji. Jedan klijent može imati više prethodnih aplikacija. Veza s glavnom tablicom: application_train.SK_ID_CURR = previous_application.SK_ID_CURR
4. Tablica installments_payments.csv sadrži povijest otplate rata za prethodne kredite koje su klijenti imali kod Home Credita. Svaki redak u ovoj tablici predstavlja jednu planiranu ili stvarnu otplatu rate. Ova tablica bilježi detalje o tome kako su klijenti plaćali svoje rate, uključujući iznose i datume. Jedan klijent može imati više prethodnih aplikacija. Veza s glavnom tablicom: application_train.SK_ID_CURR = previous_application.SK_ID_CURR


Prije pokretanja ikakvog koda u data/raw folderu moraju biti:
application_train.csv,
bureau.csv,
previous_application.csv
installments_payments.csv

koji se nalaze na:
https://www.kaggle.com/competitions/home-credit-default-risk/data

i trebaju se instalirati dependencies:

python -m pip install -r requirements.txt

Tek nakon toga šokrenuti:
1. clean_data.py
2. feature_engineering.py
3. train_and_evaluate.py

## Rezultati evaluacije:

| Model | roc_auc | pr_auc | accuracy | precision | recall | f1_score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression (Class Weight) | 0.7241 | 0.2006 | 0.6607 | 0.1475 | 0.6702 | 0.2418 |
| Logistic Regression (SMOTE) | 0.6626 | 0.1654 | 0.9067 | 0.2666 | 0.0890 | 0.1335 |
| Extra Trees (Class Weight) | 0.6972 | 0.1786 | 0.9194 | 0.7500 | 0.0024 | 0.0048 |
| Extra Trees (SMOTE) | 0.6801 | 0.1587 | 0.9151 | 0.2705 | 0.0306 | 0.0550 |
| XGBoost (Class Weight) | 0.7347 | 0.2190 | 0.7353 | 0.1705 | 0.5896 | 0.2646 |
| XGBoost (SMOTE) | 0.7283 | 0.2030 | 0.9181 | 0.4051 | 0.0318 | 0.0590 |
| Random Forest (Class Weight) | 0.6943 | 0.1796 | 0.9193 | 0.6667 | 0.0008 | 0.0016 |

Accuracy metrika nije dovoljna za odluku zbog ekstremne neuravnoteženosti podataka. Ostale pokazuju da je XGBoost model (weight_class parametar) najbolji model po f1_ score-u iako ni njegov nije vrlo visok, vidljivo je da ima visok recall (0.5896) iako nešto niži od Logistic Regression (Class Weight modela) što ukazuje na njegovu malo slabiju sposobnost prepoznavanja rizičnih klijenata, ali njegov loš precision značajno ruši njegov f1_score, ima najviši ROC-AUC, ima najviši PR-AUC. Iako je f1_score za najbolji model (0.2646) niži nego očekivan on ipak ima smisla kada se razmisli o poslovnom kontekstu ovih podataka (velika neuravnoteženost podataka, teškoća predviđanja tko će imati problema s otplatom kredita) model je prihvatljiv i spreman za daljnja poboljšanja.

Kod threshold tuning-a uspostavljeno je da najbolji F1 score konvergira negdje oko threshold = 0.6 no ipak je threshold = 0.5 procijenjen kao bolji zbog poslovne važnosti recall-a.

Odabrani model s obzirom na viši recall (pogotovo za prirodu ovog problema) pokazuje da nema pretjerani problem s prepoznavanjem manjinske klase, no loš precision govori da postoji puno false-positive slučajeva koji su, iako predstavljaju problem, tolerabilni zbog poslovnog konteksta (znatno je bolje ne posuditi od slučaja gdje posudimo i ne dobijemo povrat).

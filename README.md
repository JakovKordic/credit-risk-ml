# credit-risk-ml
Treniranje modela za prepoznavanje klijenata koji će imati problema u otplaćivanju kredita s obzirom na njihove podatke

Prije pokretanja ikakvog koda u data/raw folderu moraju biti:
application_train.csv,
bureau.csv,
previous_application.csv
installments_payments.csv

koji se nalaze na:
https://www.kaggle.com/competitions/home-credit-default-risk/data

i trebaju se instalirati paketi:

python -m pip install -r requirements.txt

Tek nakon toga šokrenuti:
1. clean_data.py
2. feature_engineering.py
3. train_and_evaluate.py
# Titanic Survival Prediction

Un progetto di Machine Learning per prevedere la sopravvivenza dei passeggeri del Titanic, basato sul dataset classico reso disponibile da Kaggle.
Include analisi dei dati, feature engineering, modelli predittivi (albero decisionale, regressione lineare) e ottimizzazione tramite Grid Search.
Struttura del progetto
```
titanic_project/
├── titanic_cleaned.csv         # Dataset preprocessato
├── titanic_features.csv        # Features pronte per il modello
├── titanic_target.csv          # Variabile target (Sopravvissuto)
├── features.py                 # Funzione di feature engineering
├── decision_tree_model.py      # Modello Decision Tree + valutazione
├── grid_search_dt.py           # Decision Tree con Grid Search
├── linear_regression_model.py  # Regressione lineare e valutazione
├── outlier_removal.py          # Rimozione outlier con IQR
├── README.md                   # Questo file
```
# Dataset

Il dataset originale è stato preprocessato per includere:

    Gestione dei valori mancanti

    Codifica delle variabili categoriche (Sex, Embarked, ecc.)

    Normalizzazione e scaling

    Creazione di nuove feature

Modelli implementati
Decision Tree Classifier

    Valutazione con Accuracy, Classification Report, Confusion Matrix

    Visualizzazione ad albero

    Ottimizzazione con GridSearchCV

Linear Regression

    Utilizzata per classificazione binaria con soglia (≥ 0.5)

    Valutazione con R², RMSE, e Accuracy

    Conversione da regressione continua a classificazione

Analisi e Visualizzazione

    Box plot per l’individuazione degli outlier

    Rimozione outlier tramite metodo IQR

    Heatmap della matrice di confusione

    Grafici interpretativi con Matplotlib e Seaborn

Setup e Requisiti

Assicurati di avere Python 3.8+ e installa le dipendenze principali:
```
pip install -r requirements.txt
```
Contenuto esempio di requirements.txt:
```
pandas
numpy
scikit-learn
matplotlib
seaborn
```
Esecuzione

Esempio per allenare e testare il Decision Tree:
```
python decision_tree_model.py
```
Esempio per eseguire la Grid Search:
```
python grid_search_dt.py
```
Risultati ottenuti

    Accuracy finale (Decision Tree ottimizzato): ~0.82

    Migliori parametri trovati via Grid Search

    Identificazione e rimozione outlier per miglioramento del dataset

*Autore: Giacomo Visciotti-Simone Verrengia*

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    accuracy_score,
    classification_report,
    confusion_matrix
)
from features import feature

# 1. Carica il dataset e crea X, y
df = pd.read_csv('titanic_cleaned.csv')
X, y = feature(df)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score, mean_squared_error,
    accuracy_score, classification_report, confusion_matrix
)
from features import feature

# 1. Carica e crea X, y
df = pd.read_csv('titanic_cleaned.csv')
X, y = feature(df)

# ————— 1) BOX-PLOT INIZIALE CON OUTLIER EVIDENZIATI —————
plt.figure(figsize=(12, 6))
# fliersize e flierprops per rendere i cerchietti degli outlier più visibili
sns.boxplot(data=X, orient='h',
            fliersize=6,
            flierprops=dict(marker='o', markerfacecolor='red', alpha=0.6))
plt.title("Box-plot delle feature (con outlier evidenziati)")
plt.xlabel("Valori standardizzati / codificati")
plt.tight_layout()
plt.show()

# ————— 2) RIMOZIONE DEGLI OUTLIER (IQR) —————
X_clean = X.copy()
# per tenere traccia delle righe da eliminare
mask = pd.Series(True, index=X_clean.index)

# calcola e applica per ogni colonna numerica
for col in X_clean.select_dtypes(include=[np.number]).columns:
    Q1 = X_clean[col].quantile(0.25)
    Q3 = X_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    # mantieni True solo dove il valore è entro i limiti
    mask &= X_clean[col].between(lower, upper)

# applica il filtro sia su X che su y
X_clean = X_clean[mask]
y_clean = y[mask]

print(f"Rimosse {len(X) - len(X_clean)} righe (~{100*(1-len(X_clean)/len(X)):.1f}%) per outlier eccessivi")



# ————— 3) BOX-PLOT DOPO PULIZIA —————
plt.figure(figsize=(12, 6))
sns.boxplot(data=X_clean, orient='h',
            fliersize=6,
            flierprops=dict(marker='o', markerfacecolor='green', alpha=0.6))
plt.title("Box-plot delle feature (dopo rimozione outlier)")
plt.xlabel("Valori standardizzati / codificati")
plt.tight_layout()
plt.show()

# riassegno i valori per semplicità
X = X_clean
y = y_clean
# # ————— 2) RIMOZIONE DELLE FEATURE MULTICOLLINEARI —————
# # Calcola la matrice di correlazione assoluta
# corr_matrix = X.corr().abs()

# # # Creiamo una maschera per la metà superiore (inclusa diagonale)
# upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# # Soglia di correlazione oltre la quale consideriamo "troppo" collineari
# threshold = 0.8

# # Individua le colonne da eliminare:
# to_drop = [col for col in upper.columns if any(upper[col] >= threshold)]
# print("Feature rimosse per alta collinearità (|corr| ≥ 0.8):", to_drop)

# # Drop delle colonne collineari
# X_reduced = X.drop(columns=to_drop)



# 3. Split in train e test
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y,
    test_size=0.2,
    random_state=73,
    stratify=y
)

# 4. Allena il modello di Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# 5. Previsioni continue sul test set
y_pred_cont = lr.predict(X_test)

# 6. Valutazione continua: R^2 e RMSE
r2  = r2_score(y_test, y_pred_cont)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_cont))
print(f"\nLinear Regression (continua): R^2 = {r2:.3f}, RMSE = {rmse:.3f}")

# 7. Conversione in predizioni binarie (soglia 0.5) e accuracy
y_pred_class = (y_pred_cont >= 0.5).astype(int)
acc = accuracy_score(y_test, y_pred_class)
print(f"Classification (post-threshold): Accuracy = {acc:.3f}\n")
print(classification_report(y_test, y_pred_class, target_names=['No', 'Yes']))

# 8. (Opzionale) Plot della confusion matrix
cm = confusion_matrix(y_test, y_pred_class)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred No','Pred Sì'],
            yticklabels=['True No','True Sì'])
plt.title("Confusion Matrix")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()

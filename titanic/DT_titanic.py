import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from features import feature
# 1. Carica il dataset pulito
df = pd.read_csv('titanic_cleaned.csv')
# richiamo la funzione feature
X, y = feature(df)

# 2. Split in train e test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=73,
    stratify=y
)

# 3. Crea e allena il Decision Tree
dt = DecisionTreeClassifier(
    criterion='gini',    # o 'entropy'
    max_depth=5,         # puoi regolarlo per evitare overfitting
    random_state=73
)
dt.fit(X_train, y_train)


# Definisci la griglia di parametri da testare
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2']
}

# Istanzia un DecisionTreeClassifier base
dt_base = DecisionTreeClassifier(random_state=73)
# 4. Previsioni sul test set
y_pred = dt.predict(X_test)

# 5. Valutazione
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy sul test set: {acc:.3f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Non Sopravvissuto', 'Sopravvissuto']))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)


# 7. Plot della Confusion Matrix
plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Pred Non Sopravvissuto', 'Pred Sopravvissuto'],
    yticklabels=['Vero Non Sopravvissuto', 'Vero Sopravvissuto']
)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# Imposta il GridSearch con 5-fold cross-validation
grid_search = GridSearchCV(
    estimator=dt_base,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,        # usa tutti i core
    verbose=1
)

# Esegui la ricerca sui dati di training
grid_search.fit(X_train, y_train)

# Stampa i migliori parametri e il relativo score CV
print("Best CV score:", grid_search.best_score_)
print("Best parameters:", grid_search.best_params_)

#  Valuta il modello ottimizzato sul test set
best_dt = grid_search.best_estimator_
y_pred_gs = best_dt.predict(X_test)

print("\nTest set performance del modello ottimizzato:")
print("Accuracy:", accuracy_score(y_test, y_pred_gs))
print(classification_report(y_test, y_pred_gs, target_names=['Non Sopravvissuto', 'Sopravvissuto']))
cm_gs = confusion_matrix(y_test, y_pred_gs)

# Plot della confusion matrix del modello ottimizzato
plt.figure(figsize=(6,5))
sns.heatmap(
    cm_gs,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Pred No', 'Pred Sì'],
    yticklabels=['True No', 'True Sì']
)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix - Best Decision Tree')
plt.tight_layout()
plt.show()


# 6. Visualizzazione dell’albero (opzionale)
plt.figure(figsize=(16, 10))
plot_tree(
    dt,
    feature_names=X.columns,
    class_names=['No', 'Yes'],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree per la Sopravvivenza sul Titanic")
plt.show()
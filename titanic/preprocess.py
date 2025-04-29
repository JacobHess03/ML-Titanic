import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Carica il CSV
df = pd.read_csv('titanic.csv')

df_1 = df.copy()

# 2. Seleziona solo le colonne richieste
colonne = [
    'PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age',
    'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'
]
df_1 = df_1[colonne]

# 3. Rimuove i duplicati
df_1 = df_1.drop_duplicates()

# 4. Riempie i valori mancanti di Age con la media
df_1['Age'].fillna(df_1['Age'].mean(), inplace=True)

# 5. Elimina la colonna Cabin
df_1 = df_1.drop(columns=['Cabin'])

# 6. Rimuove tutte le righe con dati mancanti (es. in Embarked)
df_1 = df_1.dropna()

# 7. (Opzionale) Salva il dataset pulito
df_1.to_csv('titanic_cleaned.csv', index=False)


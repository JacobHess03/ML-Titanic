import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

def feature(df):
    # ===== 6. CREAZIONE DI NUOVE FEATURE =====

    # 6.1 FamilySize = SibSp + Parch + 1 (se stesso)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # 6.2 IsAlone = 1 se FamilySize == 1, 0 altrimenti
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # 6.3 Estrazione del titolo dal nome
    #    Estrae la stringa tra la virgola e il punto (es. "Smith, Mr. John" → "Mr")
    df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.', expand=False)

    # Raggruppa i titoli rari in un'unica categoria "Rare"
    rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 
                'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')
    df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})

    # ===== 7. PREPARAZIONE FINALE PRIMA DEI MODELLI =====

    # 7.1 Selezione label (target) e drop di colonne non più necessarie
    y = df['Survived']
    df_model = df.drop(columns=[
        'PassengerId', 'Name', 'Ticket', 'Survived'
    ])

    # 7.2 Encoding delle categoriche
    #   - Sex: LabelEncoder (binary)
    le_sex = LabelEncoder()
    df_model['Sex'] = le_sex.fit_transform(df_model['Sex'])
    #   - Embarked e Title: One-hot encoding
    df_model = pd.get_dummies(df_model, columns=['Embarked', 'Title'], drop_first=True)

    # 7.3 Scaling delle numeriche
    scaler = StandardScaler()
    for col in ['Age', 'Fare', 'FamilySize']:
        df_model[col] = scaler.fit_transform(df_model[[col]])

    # 7.4 Costruzione finale del set di feature
    X = df_model.copy()

    # Ora X contiene:
    #   - Pclass (int)
    #   - Sex (0/1)
    #   - Age (scaled)
    #   - SibSp, Parch     [puoi anche rimuoverli, se preferisci tener solo FamilySize]
    #   - Fare (scaled)
    #   - FamilySize (scaled)
    #   - IsAlone (0/1)
    #   - Embarked_Q, Embarked_S
    #   - Title_Master, Title_Miss, Title_Mr, Title_Mrs, Title_Rare

   
    print("Feature matrix X e target y pronte. Dimensione X:", X.shape)


    # 2. Unisci X e y
    df_corr = pd.concat([X, y.rename('Survived')], axis=1)

    # 3. Calcola la matrice di correlazione
    corr_matrix = df_corr.corr()

    # 4. (Opzionale) Stampa
    print(corr_matrix)

    # 5. Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True
    )
    plt.title('Heatmap della Matrice di Correlazione (incl. Survived)')
    plt.tight_layout()
    plt.show()
    return X, y

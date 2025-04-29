import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# Carica il file CSV
df = pd.read_csv('titanic_cleaned.csv')

# 8. Calcola la matrice di correlazione
correlation_matrix = df.corr(numeric_only=True)

# 9. Crea una heatmap
plt.figure(figsize=(10,8))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    square=True
)
plt.title('Heatmap della Correlazione - Titanic Dataset')
plt.show()



# 2. Definisci i bin di età ogni 5 anni
max_age = int(df['Age'].max())
bins = list(range(0, max_age + 5, 5))
labels = [f"{i}-{i+4}" for i in bins[:-1]]
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

# 3. Raggruppa per fasce di età e stato di sopravvivenza
counts = df.groupby(['AgeGroup', 'Survived']).size().reset_index(name='Count')

# 4. Barplot con hue Survived
plt.figure(figsize=(14, 6))
ax = sns.barplot(
    x='AgeGroup',
    y='Count',
    hue='Survived',
    data=counts,
    dodge=True,
    palette={0: 'blue', 1: 'orange'}  # imposta esplicitamente i colori
)

# 5. Aggiungi “pin” sui bar: punti centrati su ogni barra
# Calcola gli offset orizzontali
n_groups = len(labels)
bar_width = 0.8 / 2  # total width 0.8, diviso per due hue

plt.xlabel('Fasce di età (5 anni)')
plt.ylabel('Numero di Passeggeri')
plt.title('Conteggio per Fasce di Età e Stato di Sopravvivenza (bin 5 anni)')
plt.xticks(rotation=45)
# 6. Costruisci una legenda manuale con patch colorate
patch_no  = mpatches.Patch(color='blue',   label='No')
patch_yes = mpatches.Patch(color='orange', label='Sì')
plt.legend(
    handles=[patch_no, patch_yes],
    title='Survived',
    loc='upper center',
    bbox_to_anchor=(0.9, 1),
    ncol=2
)
plt.tight_layout()
plt.show()

# Raggruppa per Sex e Survived, contando le occorrenze
counts_sex = df.groupby(['Sex', 'Survived']).size().reset_index(name='Count')

ax = sns.barplot(
    x='Sex',
    y='Count',
    hue='Survived',
    data=counts_sex,
    dodge=True,
    palette={0: 'blue', 1: 'orange'}
)

# Aggiungi i “pin” sui bar
bar_width = 0.8 / 2
sex_labels = ['female', 'male']

plt.xlabel('Sex')
plt.ylabel('Numero di Passeggeri')
plt.title('Conteggio per Sesso e Stato di Sopravvivenza')

patch_no  = mpatches.Patch(color='blue',   label='No')
patch_yes = mpatches.Patch(color='orange', label='Sì')
plt.legend(
    handles=[patch_no, patch_yes],
    title='Survived',
    loc='upper center',
    bbox_to_anchor=(0.9, 1),
    ncol=2
)
plt.tight_layout()
plt.show()





# (preprocessing invariato...)

# 2. Definisci i bin di età ogni 5 anni
max_age = int(df['Age'].max())
bins = list(range(0, max_age + 5, 5))
labels = [f"{i}-{i+4}" for i in bins[:-1]]
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

# 3. Raggruppa con observed=True per silenziare il FutureWarning
counts_age_sex = (
    df
    .groupby(['AgeGroup', 'Sex', 'Survived'], observed=True)
    .size()
    .reset_index(name='Count')
)

# 4. FacetGrid su Sex
g = sns.FacetGrid(
    counts_age_sex,
    col='Sex',
    sharey=True,
    height=6,
    aspect=1.2
)
g.map_dataframe(
    sns.barplot,
    x='AgeGroup',
    y='Count',
    hue='Survived',
    palette={0: 'blue', 1: 'orange'},
    dodge=True,
    order=labels  # assicura l’ordine corretto delle etichette
)




# 6. Costruisci una legenda manuale con patch colorate
patch_no  = mpatches.Patch(color='blue',   label='No')
patch_yes = mpatches.Patch(color='orange', label='Sì')
plt.legend(
    handles=[patch_no, patch_yes],
    title='Survived',
    loc='upper center',
    bbox_to_anchor=(0.9, 1),
    ncol=2
)

# 7. Etichette e layout
g.set_axis_labels('Fasce di età (5 anni)', 'Numero di Passeggeri')
g.set_titles(col_template="{col_name}")  # 'female' / 'male'
plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import seaborn as sns

# Metrics
metrics = ['MAE', 'RMSE', 'MAPE']

# 3 models
dummy = [3.7620, 7.7303, 29.8141]  # Modello dummy non ha dati
agcrn = [10.6209, 20.3866, 67.0415]
stnscm = [2.5289, 4.6835, 19.8475]


# Bar graph
plt.figure(figsize=(12, 7))

x = np.arange(len(metrics))
width = 0.25  # Ridotto per fare spazio al terzo modello

# Creare le tre barre, una per ogni modello
plt.bar(x - width, dummy, width, label='Dummy', color='#FF9999')
plt.bar(x, agcrn, width, label='AGCRN', color='#66B2FF')
plt.bar(x + width, stnscm, width, label='STNSCM', color='#99FF99')

plt.xlabel('Metrics', fontsize=12)
plt.ylabel('Values', fontsize=12)
plt.title('NYC 30 min', fontsize=14)
plt.xticks(x, metrics, fontsize=11)
plt.legend(fontsize=10)

# Add values above bars
for i in range(len(metrics)):
    plt.text(i - width, dummy[i] + 0.5, f'{dummy[i]:.2f}', ha='center', fontsize=9)
    plt.text(i, agcrn[i] + 0.5, f'{agcrn[i]:.2f}', ha='center', fontsize=9)
    plt.text(i + width, stnscm[i] + 0.5, f'{stnscm[i]:.2f}', ha='center', fontsize=9)

plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# Bar graph with seaborn
data = []
for i, metric in enumerate(metrics):
    data.append({'Metric': metric, 'Value': dummy[i], 'Model': 'Dummy'})
    data.append({'Metric': metric, 'Value': agcrn[i], 'Model': 'AGCRN'})
    data.append({'Metric': metric, 'Value': stnscm[i], 'Model': 'STNSCM'})

df = pd.DataFrame(data)

# Impostazione dello stile di Seaborn
sns.set_theme(style="whitegrid")
plt.figure(figsize=(14, 8))

# Creazione del grafico a barre con Seaborn
ax = sns.barplot(
    x='Metric', 
    y='Value',
    hue='Model',
    data=df,
    palette={"Dummy": "#FF9999", "AGCRN": "#66B2FF", "STNSCM": "#99FF99"},
    errwidth=0,
    alpha=0.8
)

# plt.title('Confronto delle metriche di errore tra modelli', fontsize=16, pad=20)
plt.xlabel('Metric', fontsize=14, labelpad=10)
plt.ylabel('Value', fontsize=14, labelpad=10)
plt.legend(title='Model', fontsize=12, title_fontsize=13)

# Valori sopra le barre
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', fontsize=10)

# Personalizzazione dei bordi e dello sfondo
sns.despine(left=False, bottom=False)
plt.tight_layout()
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# 3 plot separati -> migliore per me
# Creazione di tre subplot separati, uno per ogni metrica
fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=False)
metrics_df = df.pivot(index='Model', columns='Metric', values='Value').reset_index()

# Reorder the models in the desired order: STNSCM, Dummy, AGCRN
model_order = ['STNSCM', 'Dummy', 'AGCRN']
metrics_df['Model'] = pd.Categorical(metrics_df['Model'], categories=model_order, ordered=True)
metrics_df = metrics_df.sort_values('Model')

# Colori personalizzati e coerenti
palette = {"Dummy": "#FF9999", "AGCRN": "#66B2FF", "STNSCM": "#99FF99"}

# Grafico per MAE
sns.barplot(x='Model', y='MAE', data=metrics_df, ax=axes[0], palette=palette, errwidth=0, order=model_order)
axes[0].set_title('MAE', fontsize=14)

# Fix: Aggiungi etichette a tutte le barre nel primo grafico
for p in axes[0].patches:
    height = p.get_height()
    axes[0].text(p.get_x() + p.get_width()/2., height + height*0.02,
                f'{height:.2f}', ha='center', fontsize=10)

# Grafico per RMSE
sns.barplot(x='Model', y='RMSE', data=metrics_df, ax=axes[1], palette=palette, errwidth=0, order=model_order)
axes[1].set_title('RMSE', fontsize=14)

# Fix: Aggiungi etichette a tutte le barre nel secondo grafico
for p in axes[1].patches:
    height = p.get_height()
    axes[1].text(p.get_x() + p.get_width()/2., height + height*0.02,
                f'{height:.2f}', ha='center', fontsize=10)

# Grafico per MAPE
sns.barplot(x='Model', y='MAPE', data=metrics_df, ax=axes[2], palette=palette, errwidth=0, order=model_order)
axes[2].set_title('MAPE', fontsize=14)

# Fix: Aggiungi etichette a tutte le barre nel terzo grafico
for p in axes[2].patches:
    height = p.get_height()
    axes[2].text(p.get_x() + p.get_width()/2., height + height*0.02,
                f'{height:.2f}', ha='center', fontsize=10)

# Personalizzazioni finali
for ax in axes:
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    sns.despine(ax=ax, left=False, bottom=False)

plt.tight_layout()
plt.show()
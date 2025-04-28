import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import seaborn as sns

# Dati forniti
metrics = ['MAE', 'RMSE', 'MAPE']
dummy = []  # Modello dummy non ha dati
agcrn = [10.6209, 20.3866, 67.0415]
stnscn = [2.5289, 4.6835, 19.8475]

# Normalize data for radar graph
max_values = [max(agcrn[i], stnscn[i]) for i in range(len(metrics))]
agcrn_norm = [agcrn[i]/max_values[i] for i in range(len(metrics))]
stnscn_norm = [stnscn[i]/max_values[i] for i in range(len(metrics))]


# # RADAR
# def radar_factory(num_vars, frame='circle'):
#     theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
#     theta += np.pi/2

#     spoke_angles = theta.copy()
#     spoke_angles[:] = 2*np.pi-spoke_angles

#     theta = np.append(theta, theta[0])

#     # Calcola i punti del grafico
#     x = np.cos(theta)
#     y = np.sin(theta)
    
#     # Configura gli assi
#     fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
#     # Imposta i raggi
#     ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
    
#     # Imposta le etichette angolari
#     ax.set_thetagrids(np.degrees(spoke_angles), metrics)
    
#     return fig, ax

# # Create radar graph
# fig, ax = radar_factory(len(metrics))

# # Add data
# agcrn_radar = np.append(agcrn_norm, agcrn_norm[0])
# stnscn_radar = np.append(stnscn_norm, stnscn_norm[0])
# theta = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
# theta = np.append(theta, theta[0]) + np.pi/2


# ax.plot(theta, agcrn_radar, 'o-', linewidth=2, label='AGCRN')
# ax.fill(theta, agcrn_radar, alpha=0.25)
# ax.plot(theta, stnscn_radar, 'o-', linewidth=2, label='STNSCN')
# ax.fill(theta, stnscn_radar, alpha=0.25)

# ax.set_title('NYC 30 min', size=15, position=(0.5, 1.1))
# ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

# plt.tight_layout()
# plt.show()


# Bar graph
plt.figure(figsize=(10, 6))

x = np.arange(len(metrics))
width = 0.35

plt.bar(x - width/2, agcrn, width, label='AGCRN')
plt.bar(x + width/2, stnscn, width, label='STNSCN')

plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('NYC 30 min')
plt.xticks(x, metrics)
plt.legend()

# Add values above bars
for i in range(len(metrics)):
    plt.text(i - width/2, agcrn[i] + 0.5, f'{agcrn[i]:.2f}', ha='center')
    plt.text(i + width/2, stnscn[i] + 0.5, f'{stnscn[i]:.2f}', ha='center')

plt.tight_layout()
plt.show()



# Bar graph with seaborn
data = []
for i, metric in enumerate(metrics):
    data.append({'Metrica': metric, 'Valore': agcrn[i], 'Modello': 'AGCRN'})
    data.append({'Metrica': metric, 'Valore': stnscn[i], 'Modello': 'STNSCN'})

df = pd.DataFrame(data)

# Impostazione dello stile di Seaborn
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 8))

# Creazione del grafico a barre con Seaborn
ax = sns.barplot(
    x='Metrica', 
    y='Valore', 
    hue='Modello', 
    data=df,
    palette="viridis",
    errwidth=0,
    alpha=0.8
)

# Personalizzazione del grafico
plt.title('Confronto delle metriche di errore tra modelli', fontsize=16, pad=20)
plt.xlabel('Metrica', fontsize=14, labelpad=10)
plt.ylabel('Valore', fontsize=14, labelpad=10)
plt.legend(title='Modello', fontsize=12, title_fontsize=13)

# Aggiunta dei valori sopra le barre
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', fontsize=11)

# Personalizzazione dei bordi e dello sfondo
sns.despine(left=False, bottom=False)
plt.tight_layout()

# Aggiunta della griglia orizzontale più sottile
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Salvataggio dell'immagine ad alta risoluzione
plt.show()

# BONUS: Versione con scale separate per una migliore visualizzazione (dato che MAPE ha valori molto più alti)
plt.figure(figsize=(15, 10))

# Creazione di tre subplot separati, uno per ogni metrica
fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=False)
metrics_df = df.pivot(index='Modello', columns='Metrica', values='Valore').reset_index()

# Colori personalizzati e coerenti
palette = {"AGCRN": "#2D7DD2", "STNSCN": "#F45D01"}

# Grafico per MAE
sns.barplot(x='Modello', y='MAE', data=metrics_df, ax=axes[0], palette=palette, errwidth=0)
axes[0].set_title('MAE', fontsize=14)
axes[0].bar_label(axes[0].containers[0], fmt='%.2f')

# Grafico per RMSE
sns.barplot(x='Modello', y='RMSE', data=metrics_df, ax=axes[1], palette=palette, errwidth=0)
axes[1].set_title('RMSE', fontsize=14)
axes[1].bar_label(axes[1].containers[0], fmt='%.2f')

# Grafico per MAPE
sns.barplot(x='Modello', y='MAPE', data=metrics_df, ax=axes[2], palette=palette, errwidth=0)
axes[2].set_title('MAPE', fontsize=14)
axes[2].bar_label(axes[2].containers[0], fmt='%.2f')

# Titolo principale
fig.suptitle('Confronto delle metriche per ciascun modello', fontsize=16, y=1.05)

# Personalizzazioni finali
for ax in axes:
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    sns.despine(ax=ax, left=False, bottom=False)

plt.tight_layout()
plt.show()
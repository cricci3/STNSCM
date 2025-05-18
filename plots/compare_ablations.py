import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Metrics
metrics = ['MAE', 'RMSE', 'MAPE']

# Base model
stnscn = [2.5289, 4.6835, 19.8475]  # Modello base STNSCN

# Ablation tests
no_geo_mtrx30 = [2.7080, 5.0836, 21.7520]  # 30 min
no_od_mtrx = [2.8307, 5.4357, 23.3339]  # 30 min
exp1 = [5.5859, 12.0095, 48.2285]
exp2 = [5.5893, 13.2999, 46.5104]
exp3 = [2.5831, 4.4862, 20.6942]
exp3_pt2 = [2.7573, 5.3226, 22.4174]

# Creare un dizionario con tutti i dati
ablation_data = {
    'STNSCN (base)': stnscn,
    'No Geo Mtrx': no_geo_mtrx30,
    'No OD Mtrx': no_od_mtrx,
    'Exp1': exp1,
    'Exp2': exp2,
    'Exp3': exp3,
    'Exp3 Pt2': exp3_pt2
}

# Creare dataframe per i grafici
ablation_df = pd.DataFrame(columns=['Metrica', 'Valore', 'Configurazione'])

for config_name, values in ablation_data.items():
    for i, metric in enumerate(metrics):
        ablation_df = pd.concat([ablation_df, pd.DataFrame({
            'Metrica': metric,
            'Valore': values[i],
            'Configurazione': config_name
        }, index=[0])], ignore_index=True)

# Soluzione 1: Bar plot con tutte le configurazioni per ogni metrica
sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=False)
fig.suptitle('Confronto STNSCN Base vs Ablation Tests', fontsize=16, y=1.02)

# Definire una palette con colore evidenziato per il modello base
all_configs = ablation_df['Configurazione'].unique()
palette = {}
for config in all_configs:
    if config == 'STNSCN (base)':
        palette[config] = '#FF5733'  # Colore evidenziato per il base
    else:
        palette[config] = sns.color_palette("Blues_r", len(all_configs) - 1)[list(all_configs).index(config) % (len(all_configs) - 1)]

for i, metric in enumerate(metrics):
    metric_data = ablation_df[ablation_df['Metrica'] == metric]
    
    # Ordina i dati: prima il modello base, poi gli altri ordinati per valore
    base_data = metric_data[metric_data['Configurazione'] == 'STNSCN (base)']
    other_data = metric_data[metric_data['Configurazione'] != 'STNSCN (base)'].sort_values('Valore')
    ordered_data = pd.concat([base_data, other_data], ignore_index=True)
    
    ax = sns.barplot(
        x='Configurazione',
        y='Valore',
        data=ordered_data,
        palette=palette,
        ax=axes[i],
        errwidth=0
    )
    
    # Aggiungi i valori sopra le barre
    for j, p in enumerate(ax.patches):
        ax.annotate(
            f'{p.get_height():.2f}',
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='bottom',
            xytext=(0, 5),
            textcoords='offset points'
        )
    
    axes[i].set_title(f'{metric}', fontsize=14)
    axes[i].set_xlabel('')
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()

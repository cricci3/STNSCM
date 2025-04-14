import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import haversine_distances


# 1. Carica il CSV e crea intervalli temporali (es. 30 min)
df = pd.read_csv('202503-citibike-tripdata.csv', sep=',')
df['time_slot'] = pd.to_datetime(df['started_at']).dt.floor('30min')
print(df.columns)

all_stations = list(set(df['start_station_id'].unique()).union(set(df['end_station_id'].unique())))
# 2. Conta le bici in arrivo (inflow) per ogni stazione e timeslot
inflow = df.groupby(['time_slot', 'end_station_id']).size().unstack(fill_value=0)
print(inflow.shape)
# 3. Conta le bici in partenza (outflow)
outflow = df.groupby(['time_slot', 'start_station_id']).size().unstack(fill_value=0)
print(outflow.shape)

# Crea un MultiIndex con tutti i timeslot e stazioni
time_slots = df['time_slot'].unique()
multi_index = pd.MultiIndex.from_product([time_slots, all_stations], names=['time_slot', 'station_id'])

# Ri-indexa inflow e outflow
inflow = inflow.reindex(columns=all_stations, fill_value=0)
outflow = outflow.reindex(columns=all_stations, fill_value=0)

# Allinea anche le righe (timeslot)
inflow = inflow.reindex(time_slots, fill_value=0)
outflow = outflow.reindex(time_slots, fill_value=0)
print(inflow.shape, outflow.shape)  # Ora dovrebbero essere identici!
# 4. Combina in un unico tensore
X_t = np.stack([inflow.values, outflow.values], axis=-1)


print("Shape di X_t:", X_t.shape)  # Deve essere (timeslot, stazioni, 2)


# 1. Estrai giorno e ora
time_features = df.groupby('time_slot').agg({
    'time_slot': lambda x: x.dt.dayofweek.iloc[0],  # 0=lunedì, 6=domenica
    'time_slot': lambda x: x.dt.hour.iloc[0]        # 0-23
}).rename(columns={'time_slot': 'hour'})

# 2. Aggiungi meteo (esempio con dati fittizi)
time_features['temperature'] = np.random.normal(20, 5, len(time_features))  # Media 20°C

# 3. Converti in numpy
C_t = time_features.values



print("Shape di C_t:", C_t.shape)  # Deve essere (timeslot, feature)

# 1. Ottieni coordinate stazioni
stations = df[['start_station_id', 'start_lat', 'start_lng']].drop_duplicates()

# 2. Calcola A_geo (distanza geografica)
coords = stations[['start_lat', 'start_lng']].values
A_geo = haversine_distances(coords) * 6371  # 6371 = raggio Terra in km

# 3. Calcola A_trans (transizioni storiche)
transitions = df.groupby(['start_station_id', 'end_station_id']).size().unstack(fill_value=0)
A_trans = transitions / transitions.sum(axis=1)  # Normalizza


np.savez('dataset_pronto.npz',
         train_x=X_t[:800],      # 80% training
         val_x=X_t[800:900],     # 10% validation
         test_x=X_t[900:],       # 10% test
         train_x_time=C_t[:800],
         val_x_time=C_t[800:900],
         test_x_time=C_t[900:],
         train_pos=np.arange(800),  # Posizioni temporali (es. 0, 1, 2, ...)
         A_geo=A_geo,
         A_trans=A_trans)



from numpy import load

import numpy as np

data = np.load("data/train_data/BJ/all_data_r1_d1_w1_30min.npz",allow_pickle=True)
for file in data.files:
    print(file, data[file].shape)

print("-----------------")


data = np.load("dataset_pronto.npz",allow_pickle=True)
for file in data.files:
    print(file, data[file].shape)
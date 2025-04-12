import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd


# Carica il dataset
def explore_dataset(dataset_path):
    print(f"Caricamento del dataset da: {dataset_path}")
    data = np.load(dataset_path, allow_pickle=True)
    
    # 1. Elenco di tutte le chiavi nel file NPZ
    print("\n1. Chiavi disponibili nel dataset:")
    for key in data.files:
        print(f"  - {key}")
    
    # 2. Analisi della struttura di base
    print("\n2. Struttura dei dati:")
    for key in data.files:
        if isinstance(data[key], np.ndarray):
            print(f"  - {key}: shape={data[key].shape}, dtype={data[key].dtype}")
        else:
            print(f"  - {key}: type={type(data[key])}")
    
    # 3. Esaminare time_feature_index (struttura delle caratteristiche temporali)
    if 'time_feature_index' in data.files:
        print("\n3. Indice delle caratteristiche temporali:")
        try:
            time_index = data['time_feature_index'].item()
            for feature, index in time_index.items():
                print(f"  - {feature}: {index}")
        except:
            print(f"  Impossibile estrarre dettagli: {data['time_feature_index']}")
    
    # 4. Analisi dei dati di input
    if 'train_x' in data.files:
        x_data = data['train_x']
        x_transposed = x_data.transpose((0, 2, 1, 3))
        print("\n4. Analisi dei dati di input (train_x):")
        print(f"  - Forma originale: {x_data.shape}")
        print(f"  - Forma trasposta: [batch, node_num, time, dim] = {x_transposed.shape}")
        print(f"  - Quindi ci sono: {x_transposed.shape[1]} nodi/stazioni")
        print(f"  - Ogni sequenza ha: {x_transposed.shape[2]} timestep")
        print(f"  - Ogni timestep ha: {x_transposed.shape[3]} caratteristiche")
        
        # Statistiche di base
        print(f"  - Valore minimo: {x_transposed.min()}")
        print(f"  - Valore massimo: {x_transposed.max()}")
        print(f"  - Media: {x_transposed.mean()}")
        print(f"  - Deviazione standard: {x_transposed.std()}")
    
    # 5. Analisi dei dati target
    if 'train_target' in data.files:
        target_data = data['train_target']
        target_transposed = target_data.transpose((0, 2, 1, 3))
        print("\n5. Analisi dei dati target (train_target):")
        print(f"  - Forma trasposta: {target_transposed.shape}")
        
        # Statistiche dei target
        print(f"  - Valore minimo: {target_transposed.min()}")
        print(f"  - Valore massimo: {target_transposed.max()}")
        print(f"  - Media: {target_transposed.mean()}")
        print(f"  - Deviazione standard: {target_transposed.std()}")
    
    # 6. Esaminare i dati temporali
    if 'train_x_time' in data.files:
        x_time = data['train_x_time']
        print("\n6. Analisi dei dati temporali (train_x_time):")
        print(f"  - Forma: {x_time.shape}")
        if x_time.size > 0:
            if np.issubdtype(x_time.dtype, np.datetime64) or np.issubdtype(x_time.dtype, np.number):
                print(f"  - Primo valore: {x_time.flatten()[0]}")
                print(f"  - Ultimo valore: {x_time.flatten()[-1]}")
            else:
                print(f"  - Tipo di dato non analizzabile direttamente: {x_time.dtype}")
    
    # 7. Visualizzazione di un campione
    if 'train_x' in data.files:
        print("\n7. Visualizzazione campione:")
        sample_idx = 0
        node_idx = 0
        
        # Prendere un batch, un nodo e visualizzare i dati nel tempo
        try:
            sample = x_transposed[sample_idx, node_idx, :, :]
            print(f"  - Dati per batch {sample_idx}, nodo {node_idx}:")
            print(f"  - Forma: {sample.shape}")
            
            if sample.shape[1] <= 5:  # Se ci sono poche caratteristiche, mostra tutti i valori
                for feature_idx in range(sample.shape[1]):
                    print(f"    Feature {feature_idx}: {sample[:, feature_idx][:5]}... (primi 5 valori)")
            else:
                # Usa PCA per visualizzare dati con molte caratteristiche
                print("  - Troppe caratteristiche per la visualizzazione diretta, applico PCA...")
    
        except Exception as e:
            print(f"  Errore durante la visualizzazione del campione: {e}")
    
    # 8. Mappa di correlazione tra le caratteristiche (per un nodo)
    if 'train_x' in data.files and x_transposed.shape[3] > 1:
        try:
            print("\n8. Mappa di correlazione tra le caratteristiche:")
            node_data = x_transposed[0, 0, :, :]  # Primo batch, primo nodo
            corr_matrix = np.corrcoef(node_data, rowvar=False)
            print(f"  - Matrice di correlazione per il nodo 0: shape={corr_matrix.shape}")
            
            # Mostra alcuni valori di correlazione (prime 3 feature con tutte le altre)
            n_to_show = min(3, corr_matrix.shape[0])
            for i in range(n_to_show):
                print(f"  - Correlazioni della feature {i}: {corr_matrix[i, :n_to_show+1]}")
        except Exception as e:
            print(f"  Errore durante il calcolo della correlazione: {e}")
    
    # 9. Analisi delle informazioni meteo (se presenti)
    if 'time_weather_data' in data.files:
        print("\n9. Analisi dei dati meteorologici:")
        try:
            weather_data = data['time_weather_data']
            print(f"  - Forma: {weather_data.shape if isinstance(weather_data, np.ndarray) else type(weather_data)}")
            if isinstance(weather_data, np.ndarray) and weather_data.size > 0:
                print(f"  - Tipo di dato: {weather_data.dtype}")
                if weather_data.ndim <= 2:
                    print(f"  - Primi valori: {weather_data.flatten()[:5]}")
        except Exception as e:
            print(f"  Errore durante l'analisi dei dati meteo: {e}")
    
    return data

# Funzione principale
if __name__ == "__main__":
    # Sostituisci con il percorso al tuo file NPZ
    dataset_path = "data/train_data/BJ/all_data_r1_d1_w1_30min.npz"
    
    # Esplora il dataset
    data = explore_dataset(dataset_path)
    
    print("\nAnalisi completata. Ora puoi usare 'data' per ulteriori esplorazioni interattive.")
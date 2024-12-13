from mpi4py import MPI
import pandas as pd
import numpy as np

# Initialisation MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Chargement des données sur le processus maître
if rank == 0:
    try:
        data = pd.read_csv("Global_Superstore2.csv", encoding="ISO-8859-1")
        print("Colonnes disponibles :", data.columns)
        print("\nAperçu des 5 premières lignes :")
        print(data.head())
        data_chunks = np.array_split(data, size)
    except Exception as e:
        print(f"Erreur lors du chargement des données : {e}")
        data_chunks = None
else:
    data_chunks = None

# Distribution des fragments aux processus
local_data = comm.scatter(data_chunks, root=0)

# Vérification locale
print(f"Processus {rank} - Aperçu des données locales :")
print(local_data.head())


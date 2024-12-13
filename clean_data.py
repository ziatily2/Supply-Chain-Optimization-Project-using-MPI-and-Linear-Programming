from mpi4py import MPI
import pandas as pd

# Initialisation MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Chargement des données nettoyées
try:
    data = pd.read_csv("Global_Superstore2.csv", encoding="ISO-8859-1")
    data['Sales'] = pd.to_numeric(data['Sales'], errors='coerce')
    data['Profit'] = pd.to_numeric(data['Profit'], errors='coerce')
    data.dropna(inplace=True)
    data.to_csv("cleaned_data.csv", index=False)
    print("Données nettoyées sauvegardées dans 'cleaned_data.csv'.")
except Exception as e:
    print(f"Erreur lors du nettoyage des données : {e}")


from mpi4py import MPI
import pandas as pd
import matplotlib.pyplot as plt

# Initialisation MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Chargement des données
data = pd.read_csv("cleaned_data.csv", encoding="ISO-8859-1")

# Analyse exploratoire
total_sales = data.groupby('Country')['Sales'].sum()
average_profit = data.groupby('Country')['Profit'].mean()

print("\nVentes totales par pays :")
print(total_sales)

print("\nProfit moyen par pays :")
print(average_profit)

# Visualisation (exécutée seulement par le processus maître)
if rank == 0:
    total_sales.sort_values().plot(kind="bar", color="skyblue", title="Ventes Totales par Pays")
    plt.show()

    average_profit.sort_values().plot(kind="bar", color="orange", title="Profit Moyen par Pays")
    plt.show()


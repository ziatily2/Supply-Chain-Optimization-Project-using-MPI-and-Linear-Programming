from mpi4py import MPI
from scipy.optimize import linprog
import numpy as np

# Initialisation MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Configuration du problème linéaire sur le processus maître
if rank == 0:
    costs = np.array([10, 15, 20, 25, 30])  # Exemple de coûts
    A_eq = np.ones((1, len(costs)))  # Contraintes d'égalité
    b_eq = np.array([100])  # Quantité totale demandée
    bounds = [(0, None)] * len(costs)
else:
    costs, A_eq, b_eq, bounds = None, None, None, None

# Distribution des données
local_cost = comm.scatter(costs if rank == 0 else None, root=0)

# Configuration correcte de A_eq et b_eq pour chaque processus
local_A_eq = np.array([[1]])  # Matrice 2D correcte
local_b_eq = np.array([100])  # Tableau 1D correct

# Optimisation
try:
    result = linprog(
        [local_cost], 
        A_eq=local_A_eq, 
        b_eq=local_b_eq, 
        bounds=[(0, None)], 
        method="highs"
    )

    if result.success:
        print(f"\nEntrepôt {rank + 1} - Coût minimal trouvé : {result.fun:.2f}")
        print(f"Quantités optimales expédiées : {result.x}")
    else:
        print(f"\nEntrepôt {rank + 1} - Erreur : {result.message}")
except Exception as e:
    print(f"\nEntrepôt {rank + 1} - Erreur : {e}")


from mpi4py import MPI
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Initialisation MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Chargement des données
try:
    data = pd.read_csv("cleaned_data.csv", encoding="ISO-8859-1")
except Exception as e:
    print(f"Erreur lors du chargement des données : {e}")
    data = pd.DataFrame()

if not data.empty:
    # Modélisation
    X = data[['Sales']].values
    y = data['Profit'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    print(f"\nErreur quadratique moyenne (RMSE) globale : {rmse}")


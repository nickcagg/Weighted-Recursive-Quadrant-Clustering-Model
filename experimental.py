import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

type Packet = dict[int, int]
EPSILON = 1e-10

def sigmoid(x, L, k, x0, b):
“”“General sigmoid: L / (1 + exp(-k * (x - x0))) + b”””
return L / (1 + np.exp(-k * (x - x0))) + b

class NLearner:
def **init**(self, size_r: int = 20, size_c: int = 8):
self.size_r = size_r
self.size_c = size_c
self.data: np.ndarray | None = None
self.centers: dict[str, tuple] | None = None
self.model_components: pd.DataFrame | None = None

```
def generate_data(self, low: int = 1, high: int = 10) -> None:
    rng = np.random.default_rng()
    self.data = rng.integers(low=low, high=high, size=(self.size_r, self.size_c))
    self.data = np.loadtxt('data.csv', delimiter=',', skiprows=1)

def get_centers(self) -> dict[str, tuple]:
    if self.data is None:
        raise ValueError("Data not generated. Call generate_data() first.")

    self.centers = {
        "col": tuple(self.data.mean(axis=0)),
        "row": tuple(self.data.mean(axis=1)),
    }
    return self.centers

def encode_data(self) -> pd.DataFrame:
    if self.data is None:
        raise ValueError("Data not generated. Call generate_data() first.")
    if self.centers is None:
        raise ValueError("Centers not computed. Call get_centers() first.")

    df = pd.DataFrame(self.data)
    col_mean = np.array(self.centers["col"])
    row_mean = np.array(self.centers["row"])

    col_grid, row_grid = np.meshgrid(col_mean, row_mean)
    mean_matrix = (col_grid + row_grid) / 2

    deviation = df.values - mean_matrix

    squared = pd.DataFrame(deviation, columns=df.columns) ** 2 + EPSILON
    compiled_df = np.log(squared)
    compiled_df = compiled_df.replace(0, EPSILON)

    self.model_components = compiled_df
    return compiled_df

def compress_model(self) -> pd.DataFrame:
    model = self.model_components.copy() * 360
    model = np.deg2rad(model)
    model['R'] = model.mean(axis=1, numeric_only=True).astype(float).replace(0, EPSILON)
    return model[['R']]

def unpack(self) -> pd.DataFrame:
    if self.model_components is None:
        raise ValueError("Model not encoded. Call encode_arr() first.")

    packaged_model = self.compress_model()
    packaged_model['D'] = 10 / np.sin(packaged_model['R'])
    packaged_model['p'] = packaged_model['D'].rank(pct=True)
    return packaged_model[['p', 'D']]

def fit_sigmoid(self, pack: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = pack['p'].values
    y = pack['D'].values

    # Initial guesses: L=range, k=1, x0=midpoint, b=min
    L0 = y.max() - y.min()
    k0 = 1.0
    x0_0 = x.mean()
    b0 = y.min()
    p0 = [L0, k0, x0_0, b0]

    popt, pcov = curve_fit(sigmoid, x, y, p0=p0, maxfev=10000)
    return popt, pcov

def run(self) -> None:
    self.generate_data()
    self.get_centers()
    self.encode_data()
    pack = self.unpack()

    popt, pcov = self.fit_sigmoid(pack)
    L, k, x0, b = popt
    print(f"Sigmoid fit: L={L:.4f}, k={k:.4f}, x0={x0:.4f}, b={b:.4f}")

    x_fit = np.linspace(pack['p'].min(), pack['p'].max(), 300)
    y_fit = sigmoid(x_fit, *popt)

    plt.scatter(pack['p'], pack['D'], label='Data', zorder=5)
    plt.plot(x_fit, y_fit, color='red', label='Sigmoid fit')
    plt.xlabel('p')
    plt.ylabel('D')
    plt.title('Sigmoid Fit')
    plt.legend()
    plt.tight_layout()
    plt.show()
```

if **name** == “**main**”:
model = NLearner(size_r=20, size_c=8)
model.run()
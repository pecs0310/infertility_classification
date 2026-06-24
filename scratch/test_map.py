import time
import numpy as np
import pandas as pd

n = 250000
unique_cats = [f"cat_{i}" for i in range(100)]
vals = np.random.choice(unique_cats, size=n)
mapping = {cat: np.random.rand() for cat in unique_cats}
gmean = 0.25

# Method 1: List comprehension
t0 = time.time()
res1 = [mapping.get(v, gmean) for v in vals]
t1 = time.time()
print(f"List comprehension: {t1 - t0:.4f}s")

# Method 2: pd.Series.map
t0 = time.time()
res2 = pd.Series(vals).map(mapping).fillna(gmean).values
t1 = time.time()
print(f"pd.Series.map: {t1 - t0:.4f}s")

import time
import math
import numpy as np
import pandas as pd

n = 250000
unique_cats = [f"cat_{i}" for i in range(100)]
fv = np.random.choice(unique_cats, size=n)
fy = np.random.choice([0, 1], size=n)

alpha = 1.0
min_cnt = 10
clip = 3.0
t_pos = float(fy.sum())
t_neg = float(len(fy) - t_pos)

# Method 1: Original loop
t0 = time.time()
mapping1 = {}
for cat in np.unique(fv):
    m = fv == cat
    cnt = int(m.sum())
    if cnt < min_cnt:
        mapping1[cat] = 0.0
        continue
    pos = float(fy[m].sum())
    neg = float(cnt - pos)
    woe = math.log((pos + alpha) / (t_neg + alpha)) - math.log((neg + alpha) / (t_pos + alpha))
    mapping1[cat] = float(np.clip(woe, -clip, clip))
t1 = time.time()
print(f"Original loop: {t1 - t0:.4f}s")

# Method 2: Pandas groupby
t0 = time.time()
df_temp = pd.DataFrame({'fv': fv, 'target': fy})
grouped = df_temp.groupby('fv')['target'].agg(['count', 'sum'])
cnt = grouped['count']
pos = grouped['sum']
neg = cnt - pos
woe_vals = np.log((pos + alpha) / (t_neg + alpha)) - np.log((neg + alpha) / (t_pos + alpha))
woe_vals = np.clip(woe_vals, -clip, clip)
woe_vals[cnt < min_cnt] = 0.0
mapping2 = woe_vals.to_dict()
t1 = time.time()
print(f"Pandas groupby: {t1 - t0:.4f}s")

# Assert values are equal
for k in mapping1:
    assert abs(mapping1[k] - mapping2[k]) < 1e-5, f"Mismatch at {k}: {mapping1[k]} vs {mapping2[k]}"
print("Values match perfectly!")

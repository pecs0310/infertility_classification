import os
import numpy as np
import pandas as pd

# Define paths relative to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SUB_DIR = os.path.join(BASE_DIR, "submission file")

# File names
V18_FILE = os.path.join(SUB_DIR, "submission_v18_bag_0.740822.csv")
TEAM_BEST_FILE = os.path.join(SUB_DIR, "team_best_submit_0.741849.csv")

print(f"Loading v18 submission from: {V18_FILE}")
print(f"Loading Team Best submission from: {TEAM_BEST_FILE}")

# Load submissions
df_v18 = pd.read_csv(V18_FILE)
df_team = pd.read_csv(TEAM_BEST_FILE)

# Validate alignment
assert (df_v18["ID"] == df_team["ID"]).all(), "IDs are not aligned between submissions!"
print(f"Loaded successfully. Total rows: {len(df_v18)}")

class ECDFReference:
    def __init__(self, ref):
        self.sorted = np.sort(np.asarray(ref, dtype=float))
        self.n = max(len(self.sorted), 1)

    def transform(self, x):
        x = np.asarray(x, dtype=float)
        left = np.searchsorted(self.sorted, x, side="left")
        right = np.searchsorted(self.sorted, x, side="right")
        ranks = (left + right) / 2.0
        return ranks / self.n

# Get ranks
print("Computing ECDF ranks...")
ecdf_v18 = ECDFReference(df_v18["probability"])
ecdf_team = ECDFReference(df_team["probability"])

rank_v18 = ecdf_v18.transform(df_v18["probability"])
rank_team = ecdf_team.transform(df_team["probability"])

def map_ranks_to_probabilities(ranks, ref_probs):
    sorted_ref = np.sort(ref_probs)
    n = len(sorted_ref)
    indices = ranks * (n - 1)
    return np.interp(indices, np.arange(n), sorted_ref)

# Define blend weights (w_v18, w_team)
blend_weights = [
    (0.5, 0.5, "50_50"),
    (0.4, 0.6, "40_60"),
    (0.3, 0.7, "30_70"),
    (0.6, 0.4, "60_40"),
]

for w_v18, w_team, suffix in blend_weights:
    print(f"\nBlending with weights: v18 ({w_v18:.1f}) + Team Best ({w_team:.1f})")
    
    # Blended rank
    blended_rank = w_v18 * rank_v18 + w_team * rank_team
    
    # Map back to the original probability space of the Team Best submission
    blended_probs = map_ranks_to_probabilities(blended_rank, df_team["probability"])
    
    # Create submission
    sub = df_v18.copy()
    sub["probability"] = blended_probs
    
    # Save files to submission file/ folder
    out_file = os.path.join(SUB_DIR, f"submission_v18_blend_{suffix}.csv")
    sub.to_csv(out_file, index=False)
    print(f"Saved: {out_file}")

print("\nBlending process complete!")

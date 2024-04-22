import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pickle
import json
import argparse
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

parser = argparse.ArgumentParser()
parser.add_argument("runs_file")
parser.add_argument("--legend", action="store_true")
args = parser.parse_args()
runs_file = args.runs_file
legend = args.legend

run_hashes = json.load(open(runs_file, 'rb'))

df = pd.DataFrame()
for run_name, run_dir in run_hashes.items():
    for seed_dir in os.listdir(run_dir):
        if not seed_dir.startswith("seed"):
            continue
        log = pickle.load(open(os.path.join(run_dir, seed_dir, "info.pkl"), "rb"))
        eval_returns = np.array(log["Eval Returns"]).flatten().tolist()
        df = pd.concat([
            df, pd.DataFrame(
                {
                    "Eval Returns": eval_returns,
                    "N": [i for i in range(len(eval_returns))],
                    "algorithm": [run_name for _ in range(len(eval_returns))],
                    "seed": [seed_dir for _ in range(len(eval_returns))]
                }
            )
        ])

# Smooth the data using a 1D filter
smoothed_returns = df.groupby(['algorithm', 'seed'])['Eval Returns'].transform(lambda x: gaussian_filter1d(x, sigma=5))

# Add smoothed data to DataFrame
df['Smoothed Returns'] = smoothed_returns

# Plot with smoothed data and confidence intervals
# sns.lineplot(data=df, y="Eval Returns", x="N", hue='algorithm', style='algorithm', errorbar=("se", 1), alpha=0.001, legend="auto" if legend else False)
sns.lineplot(data=df, y="Smoothed Returns", x="N", hue='algorithm', style='algorithm', errorbar=("se", 1), legend="auto" if legend else False)
plt.savefig(runs_file[:-4] + "png")
print(f"Saved results in {runs_file[:-4]}png .")

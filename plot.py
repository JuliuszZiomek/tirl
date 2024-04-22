import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pickle
import json
import argparse
import seaborn as sns

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

sns.lineplot(df, y="Eval Returns", x="N", hue='algorithm', style='algorithm', errorbar=("se", 1), legend="auto" if legend else False)
plt.savefig(runs_file[:-4] + ".png")
print(f"Saved results in {runs_file[:-4]}.png .")


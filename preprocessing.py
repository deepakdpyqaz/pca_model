from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
from pathlib import Path
from sklearn.decomposition import PCA
import numpy as np
EMOTION_MAP = {
    "anger":0,
    "disgust":1,
    "fearful":2,
    "happy":3,
    "neutral":4,
    "sad":5,
    "surprised":6,
    "calm":7,
    "boredom":8,
}
def standardize(fname):
    df = pd.read_csv(fname)
    df.iloc[:,0] = df.iloc[:,0].map(EMOTION_MAP)
    fname=fname.stem
    scaler = StandardScaler()
    scaler.fit(df.iloc[:,1:])
    df.iloc[:,1:] = scaler.transform(df.iloc[:,1:])
    df.to_csv(f'data/{fname}_pca0.csv', index=False)
    print(f"Finished {fname}")
    return fname

def apply_pca(fname):
    n_components = [i for i in range(10,55,5)]
    n_components.append(52)
    try:
        for n in n_components:
            df = pd.read_csv(fname)
            pca = PCA(n_components=n)
            y = pd.DataFrame(df.iloc[:,0].map(EMOTION_MAP))
            pca.fit(df.iloc[:,1:])
            X = pca.transform(df.iloc[:,1:])
            scaler = StandardScaler()
            X = pd.DataFrame(scaler.fit_transform(X))
            df = pd.concat([y,X], axis=1, ignore_index=True)
            df.to_csv(f'data/{fname.stem}_pca{n}.csv', index=False)
            print(f"Finished {fname.stem}_pca{n}")
        return fname.stem
    except Exception as e:
        print(f"Exception {e} in {fname.stem}")
if __name__ == "__main__":
    datasets = ["Ravdess","German","Persian","Italian","Bangla"]
    csv_files = [Path(f"data/{dataset}.csv") for dataset in datasets]
    with Pool() as p:
        p.map(standardize, csv_files)
    with Pool() as p:
        p.map(apply_pca, csv_files)
    print("DONE")
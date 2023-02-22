import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from multiprocessing import Pool

if not os.path.exists("figure"):
    os.mkdir("figure")

directory = "confusion_matrix"

def save_plot(file):
    df = pd.read_csv(os.path.join(directory, file))
    sns.heatmap(df.to_numpy(), annot=True, fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(file.replace(".csv", ""))
    plt.savefig(os.path.join("figure", file.replace(".csv", ".png")))
    plt.close()
    print("Saved: {}".format(file))

if __name__ == "__main__":
    pool = Pool()
    pool.map(save_plot, os.listdir(directory))
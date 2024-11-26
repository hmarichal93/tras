import numpy as np
import pandas as pd

from pathlib import Path



class Comparison:
    def __init__(self, coorecorder_path, tras_path):
        self.tras_width = self.compute_tras_width(tras_path)
        self.coorecorder_width = self.compute_coorecorder_width(coorecorder_path)

    def load_coorecorder_file(self, path):
        with open(path, "r") as f:
            lines = f.readlines()

        y, x = [], []
        for l in lines[11:]:
            l = l.replace("\n","").split(",")
            y.append(float(l[0]))
            x.append(float(l[1]))
        return pd.DataFrame({"x": x, "y": y})

    def compute_coorecorder_width(self, path):
        df = self.load_coorecorder_file(path)
        widths = self.process_coorecorder_width(df)
        return np.array(widths).tolist()

    def process_coorecorder_width(self, df):
        "Input is a dataframe with the x and y coordinates of pixels. Output is a list of widths"
        x, y = df["x"].values, df["y"].values
        widths = []
        for i in range(1, len(x)):
            width = np.sqrt((x[i] - x[i-1])**2 + (y[i] - y[i-1])**2)
            widths.append(width)
        return widths


    def load_tras_file(self, path):
        df = pd.read_csv(path)
        return df
    def compute_tras_width(self, path):
        df = self.load_tras_file(path)
        widths = self.process_tras_width(df)
        return np.array(widths).tolist()

    def process_tras_width(self, df):
        return df["Width [mm]"].values


def load_measurements(data_path):
    files = data_path.glob("*.pos")

    tras_measurements = []
    coorecorder_measurements = []
    for coorecorder_file in files:
        tras_file = str(coorecorder_file).replace(".pos", ".csv")
        c = Comparison(coorecorder_file, tras_file)
        if len(c.tras_width) != len(c.coorecorder_width):
            print(f"Skipping {coorecorder_file}")
            continue
        tras_measurements += c.tras_width
        coorecorder_measurements += c.coorecorder_width

    return coorecorder_measurements, tras_measurements

def pearson_coorelation(x,y):
    from scipy.stats import pearsonr
    # Compute Pearson's correlation coefficient and p-value
    r, p_value = pearsonr(x, y)

    # Sample size
    n = len(x)  # Ensure x and y have the same length

    # Print results
    print(f"Pearson's r: {r:.4f}, p-value: {p_value:.10f}, n={n}")
    return r, p_value, n

def main():
    data_path = Path("./experiments/coorecorder/data_2")
    coorecorder_measurements, tras_measurements = load_measurements(data_path)
    import matplotlib.pyplot as plt
    coorecorder_measurements = np.array(coorecorder_measurements)
    tras_measurements = np.array(tras_measurements)
    plt.figure()
    #smaller markers

    fit = np.polyfit(coorecorder_measurements, tras_measurements, 1)
    strline = f"y = {fit[0]:.4f}x + {fit[1]:.4f}"
    r, p_value, n = pearson_coorelation(coorecorder_measurements, tras_measurements)
    #compute rmse
    rmse = np.sqrt(np.mean((tras_measurements - coorecorder_measurements)**2))

    plt.scatter(coorecorder_measurements, tras_measurements, s=1, label="Measurements")

    fit_fn = np.poly1d(fit)
    plt.plot(coorecorder_measurements, fit_fn(coorecorder_measurements), '--k', alpha=0.1, label="Fit 1d")
    plt.xlabel("Coorecorder (mm)")
    plt.ylabel("TRAS (mm)")
    plt.title("Ring Width Measurements")
    #add to figure text "strline\nR={r:.2f}, p={p_value:.2f}, n={n}"
    plt.text(0.65, 0.2, f"n={n}\n{strline}\nr={r:.4f}\nRMSE={rmse:.4f}",
             horizontalalignment='left',
             verticalalignment='center', transform=plt.gca().transAxes)
    #plt.grid()
    plt.legend()
    plt.savefig("coorecorder_tras_comparison.png")
    plt.show()

    print(f"Fit: {fit}")

    df = pd.DataFrame({"coorecorder": coorecorder_measurements, "tras": tras_measurements})
    df.to_csv("./experiments/data/comparison.csv")
    return








if __name__=="__main__":
    main()
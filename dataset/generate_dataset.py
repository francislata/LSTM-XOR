import numpy as np
import pandas as pd
from tqdm import tqdm

def generate_samples(string_length, samples_count, csv_filename_path):
    X = []
    y = []
    
    for _ in tqdm(range(samples_count)):
        current_x = np.random.choice([0, 1], size=(string_length,))
        current_y = current_x[0]

        for x in current_x[1:]:
            current_y = np.bitwise_xor(current_y, x)

        X.append(' '.join([str(x) for x in current_x]))
        y.append(current_y)

    dataframe = pd.DataFrame({"inputs": X, "labels": y})
    dataframe.to_csv(csv_filename_path, index=False)

if __name__ == "__main__":
    print("Generating training dataset...")
    generate_samples(2, 100000, "training.csv")
    print("Done generating training dataset!\n")

    print("Generating validation dataset...")
    generate_samples(2, 10000, "validation.csv")
    print("Done generating validation dataset!")

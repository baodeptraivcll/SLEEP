# ponytail: Script to calculate class distribution of the entire dataset and output Table 1 format.
import os
import numpy as np

def calculate_distribution(data_dir):
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' does not exist.")
        return

    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')]
    print(f"Analyzing {len(files)} NPZ files in '{data_dir}'...")

    class_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    class_names = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}

    for f in files:
        data = np.load(f)
        y = data['y']
        for val in y:
            if val in class_counts:
                class_counts[val] += 1
            else:
                print(f"Warning: Unexpected class label {val} in file {f}")

    total_epochs = sum(class_counts.values())

    print("\n" + "=" * 45)
    print("      CLASS DISTRIBUTION (Table 1 Backup)")
    print("=" * 45)
    print(f"{'Stage':<10} {'Epochs':>12} {'Percentage':>15}")
    print("-" * 45)
    for label in sorted(class_counts.keys()):
        count = class_counts[label]
        pct = (count / total_epochs) * 100 if total_epochs > 0 else 0.0
        print(f"{class_names[label]:<10} {count:>12,} {pct:>14.2f}%")
    print("-" * 45)
    print(f"{'Total':<10} {total_epochs:>12,} {100.0:>14.2f}%")
    print("=" * 45)

if __name__ == "__main__":
    # Detect relative path
    default_path = "./SC_Data" if os.path.exists("./SC_Data") else "d:/SLEEP/SC_Data"
    calculate_distribution(default_path)

import os
import glob
import numpy as np
import mne
import argparse
from scipy.signal import butter, lfilter

# ==============================================================================
# CONFIGURATION
# ==============================================================================
EPOCH_SEC_SIZE = 30
TARGET_HZ = 100
EPOCH_SIZE = EPOCH_SEC_SIZE * TARGET_HZ
EEG_CHANNELS = ['EEG Fpz-Cz']
WAKE_TRIM_MINUTES = 30
WAKE_TRIM_EPOCHS = (WAKE_TRIM_MINUTES * 60) // EPOCH_SEC_SIZE

# AASM Label mapping
ann2label = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
    "Sleep stage ?": -1,
    "Movement time": -1
}

# ==============================================================================
# SIGNAL PROCESSING FUNCTIONS
# ==============================================================================
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def z_score_normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std if std > 0 else data

# ==============================================================================
# DATA EXTRACTION
# ==============================================================================
def process_subject(psg_file, ann_file, output_dir):
    print(f"Processing: {os.path.basename(psg_file)}")
    
    # 1. Read EDF files
    try:
        raw = mne.io.read_raw_edf(psg_file, preload=True, verbose=False)
        annots = mne.read_annotations(ann_file)
    except Exception as e:
        print(f"Error reading {psg_file}: {e}")
        return

    # 2. Resample and extract channel
    raw.pick_channels(EEG_CHANNELS)
    if raw.info['sfreq'] != TARGET_HZ:
        raw.resample(TARGET_HZ, npad="auto", verbose=False)
    
    # Extract signals and apply filtering & normalization
    data = raw.get_data()[0]
    data = butter_bandpass_filter(data, 0.5, 30.0, TARGET_HZ, order=5)
    data = z_score_normalize(data)

    # 3. Map annotations to epochs
    labels = []
    for annot in annots:
        desc = annot['description']
        onset = annot['onset']
        duration = annot['duration']
        
        # Determine number of 30s epochs in this annotation
        num_epochs = int(np.round(duration / EPOCH_SEC_SIZE))
        label = ann2label.get(desc, -1)
        labels.extend([label] * num_epochs)

    labels = np.array(labels)
    
    # Slice the data into 30s epochs
    num_epochs = min(len(labels), len(data) // EPOCH_SIZE)
    labels = labels[:num_epochs]
    data_epochs = data[:num_epochs * EPOCH_SIZE].reshape((num_epochs, 1, EPOCH_SIZE))
    
    # 4. Wake Trimming Logic
    # Find the first and last non-Wake epochs
    sleep_indices = np.where(labels > 0)[0]
    if len(sleep_indices) == 0:
        print("No sleep stages found. Skipping.")
        return

    first_sleep = sleep_indices[0]
    last_sleep = sleep_indices[-1]

    # Calculate keeping bounds
    start_idx = max(0, first_sleep - WAKE_TRIM_EPOCHS)
    end_idx = min(len(labels), last_sleep + WAKE_TRIM_EPOCHS + 1)

    # Trim the arrays
    labels = labels[start_idx:end_idx]
    data_epochs = data_epochs[start_idx:end_idx]
    
    # Remove UNKNOWN/MOVEMENT epochs (-1)
    valid_indices = np.where(labels != -1)[0]
    labels = labels[valid_indices]
    data_epochs = data_epochs[valid_indices]

    # 5. Save as NPZ
    subject_id = os.path.basename(psg_file).split('-')[0][:7] # Extract e.g., SC4001E
    out_file = os.path.join(output_dir, f"{subject_id}.npz")
    np.savez(out_file, x=data_epochs, y=labels)
    print(f"Saved {len(labels)} epochs to {out_file}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/sleep-edf-database-expanded-1.0.0/sleep-cassette",
                        help="Directory containing the PSG and Hypnogram EDF files.")
    parser.add_argument("--output_dir", type=str, default="./data/preprocessed",
                        help="Directory to save the processed .npz files.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Match PSG and Annotation files
    psg_files = sorted(glob.glob(os.path.join(args.data_dir, "*PSG.edf")))
    ann_files = sorted(glob.glob(os.path.join(args.data_dir, "*Hypnogram.edf")))
    
    if len(psg_files) == 0:
        print(f"No PSG files found in {args.data_dir}.")
        return

    for psg_file, ann_file in zip(psg_files, ann_files):
        # Double check matching files
        if psg_file.split('-')[0] == ann_file.split('-')[0]:
            process_subject(psg_file, ann_file, args.output_dir)
        else:
            print(f"Mismatch: {psg_file} and {ann_file}")

if __name__ == "__main__":
    main()

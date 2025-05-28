import librosa
import librosa.display
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import warnings
import json # For saving normalization parameters

# Suppress some common librosa warnings if they are not critical
warnings.filterwarnings('ignore', category=UserWarning)

# --- Configuration ---
SOUNDS_DIR = 'sounds'
OUTPUT_RAW_CSV = 'sound_features.csv' # Non-normalized raw features
OUTPUT_NORMALIZED_CSV = 'sound_features_normalized.csv'
NORMALIZATION_PARAMS_JSON = 'sound_normalization_params.json' # To save min/max for denormalization
SR = 44100 # Standard sample rate for consistent feature extraction.
HOP_LENGTH = 512 # Standard hop length for frame-based features
FRAME_LENGTH = 2048 # Standard frame length for FFT-based features

# --- Feature Extraction Function ---
def extract_single_audio_features(audio_path, sr=SR, hop_length=HOP_LENGTH, frame_length=FRAME_LENGTH):
    """
    Extracts 10 selected aggregate features from a single audio file.
    These features are based on commonly used MIR descriptors and their
    single-value summary statistics for an entire audio track.
    Attack and Decay times are simplified heuristics.
    """
    try:
        y, sr = librosa.load(audio_path, sr=sr)
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return None

    features = {}

    # Check if audio is too short for some features
    if len(y) < sr * 0.5: # e.g., less than half a second
        print(f"Warning: Audio file {os.path.basename(audio_path)} is very short ({len(y)/sr:.2f}s). Some features might be unreliable.")
        # Assign default/zero values for very short tracks
        for feature_name in [
            'spectral_centroid_mean', 'spectral_rolloff_85_mean', 'spectral_flux_mean',
            'spectral_bandwidth_mean', 'spectral_flatness_mean', 'zero_crossing_rate_mean',
            'rms_db_mean', 'tempo_bpm', 'attack_time_s', 'decay_time_s'
        ]:
            features[feature_name] = 0.0
        return features

    # 1. Spectral Centroid
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length, n_fft=frame_length)
    features['spectral_centroid_mean'] = np.mean(cent)

    # 2. Spectral Rolloff (85%)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length, n_fft=frame_length, roll_percent=0.85)
    features['spectral_rolloff_85_mean'] = np.mean(rolloff)

    # 3. Spectral Flux (Approximation via mean of onset strength envelope)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    features['spectral_flux_mean'] = np.mean(onset_env)

    # 4. Spectral Bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length, n_fft=frame_length)
    features['spectral_bandwidth_mean'] = np.mean(bandwidth)

    # 5. Spectral Flatness
    flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length, n_fft=frame_length)
    features['spectral_flatness_mean'] = np.mean(flatness)

    # 6. Zero-Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)
    features['zero_crossing_rate_mean'] = np.mean(zcr)

    # 7. RMS Energy (Mean, in dB)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)
    rms_db = librosa.amplitude_to_db(rms, ref=np.max) # Using np.max for reference to get relative dBFS
    features['rms_db_mean'] = np.mean(rms_db)

    # 8. Tempo (BPM)
    if len(y) / sr >= 2.0: # At least 2 seconds for a somewhat reliable tempo estimate
        # FIX: Removed 'units='bpm'' as it's not a valid argument for beat_track
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
        features['tempo_bpm'] = tempo
    else:
        features['tempo_bpm'] = 0.0 # Assign a default if too short for reliable tempo

    # 9. Attack Time (Simplified Global Heuristic)
    rms_frames = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    time_points = librosa.frames_to_time(np.arange(len(rms_frames)), sr=sr, hop_length=hop_length)

    peak_rms = np.max(rms_frames)
    if peak_rms == 0:
        features['attack_time_s'] = 0.0
    else:
        idx_10_percent = np.where(rms_frames >= 0.1 * peak_rms)[0]
        idx_90_percent = np.where(rms_frames >= 0.9 * peak_rms)[0]

        if len(idx_10_percent) > 0 and len(idx_90_percent) > 0 and idx_90_percent[0] > idx_10_percent[0]:
            features['attack_time_s'] = time_points[idx_90_percent[0]] - time_points[idx_10_percent[0]]
        elif len(idx_90_percent) > 0:
            features['attack_time_s'] = time_points[idx_90_percent[0]]
        else:
            features['attack_time_s'] = time_points[-1]

    # 10. Decay Time (Simplified Global Heuristic)
    if peak_rms == 0:
        features['decay_time_s'] = 0.0
    else:
        idx_above_90_from_end = np.where(rms_frames >= 0.9 * peak_rms)[0]
        idx_below_10_from_end = np.where(rms_frames <= 0.1 * peak_rms)[0]

        if len(idx_above_90_from_end) > 0 and len(idx_below_10_from_end) > 0:
            last_peak_idx = idx_above_90_from_end[-1]
            first_low_after_peak_idx = idx_below_10_from_end[idx_below_10_from_end > last_peak_idx]
            if len(first_low_after_peak_idx) > 0:
                features['decay_time_s'] = time_points[first_low_after_peak_idx[0]] - time_points[last_peak_idx]
            else:
                features['decay_time_s'] = time_points[-1] - time_points[last_peak_idx]
        else:
            features['decay_time_s'] = 0.0

    return features

# --- Main Script Execution ---
def main():
    # Create the 'sounds' directory if it doesn't exist
    if not os.path.exists(SOUNDS_DIR):
        os.makedirs(SOUNDS_DIR)
        print(f"Created directory: '{SOUNDS_DIR}'")
        print(f"Please place your audio files (e.g., echoes.wav, rhythm.wav, script.wav) in this directory.")
        print("Then, run the script again.")
        return

    audio_files = [f for f in os.listdir(SOUNDS_DIR) if f.lower().endswith(('.wav', '.mp3', '.flac'))]

    if not audio_files:
        print(f"No audio files found in '{SOUNDS_DIR}'. Please place your audio files there.")
        return

    print(f"Found {len(audio_files)} audio files in '{SOUNDS_DIR}'. Extracting features...")

    all_features_data = [] # To store dicts of features for each file

    for i, audio_file in enumerate(audio_files):
        print(f"Processing ({i+1}/{len(audio_files)}): {audio_file}")
        file_path = os.path.join(SOUNDS_DIR, audio_file)
        
        # Ensure 'filename' is consistently the base name without extension for indexing
        base_filename_no_ext = os.path.splitext(audio_file)[0]
        
        extracted_features = extract_single_audio_features(file_path, sr=SR, hop_length=HOP_LENGTH, frame_length=FRAME_LENGTH)
        
        if extracted_features:
            extracted_features['filename_id'] = base_filename_no_ext
            all_features_data.append(extracted_features)
        else:
            print(f"Skipping {audio_file} due to extraction error.")

    if not all_features_data:
        print("No features extracted successfully from any audio file. Exiting.")
        return

    # Create DataFrame from collected feature dictionaries
    raw_features_df = pd.DataFrame(all_features_data).set_index('filename_id')
    
    # Define the exact order of columns for consistency
    feature_column_order = [
        'spectral_centroid_mean', 'spectral_rolloff_85_mean', 'spectral_flux_mean',
        'spectral_bandwidth_mean', 'spectral_flatness_mean', 'zero_crossing_rate_mean',
        'rms_db_mean', 'tempo_bpm', 'attack_time_s', 'decay_time_s'
    ]
    raw_features_df = raw_features_df[feature_column_order]


    raw_features_df.to_csv(OUTPUT_RAW_CSV)
    print(f"\nRaw (non-normalized) features saved to '{OUTPUT_RAW_CSV}'.")
    print("Example of raw features (first 3 rows):")
    print(raw_features_df.head(3))

    # --- Normalization ---
    print("\nNormalizing features...")
    scaler = MinMaxScaler(feature_range=(0, 1)) # Normalize to [0, 1] range

    # Fit the scaler on the raw features and transform
    normalized_features_array = scaler.fit_transform(raw_features_df)
    normalized_features_df = pd.DataFrame(normalized_features_array, columns=raw_features_df.columns, index=raw_features_df.index)
    normalized_features_df.to_csv(OUTPUT_NORMALIZED_CSV)
    print(f"Normalized features saved to '{OUTPUT_NORMALIZED_CSV}'.")
    print("Example of normalized features (first 3 rows):")
    print(normalized_features_df.head(3))

    # --- Save Normalization Parameters ---
    normalization_params = {
        'feature_names': raw_features_df.columns.tolist(),
        'min_values': scaler.data_min_.tolist(),
        'max_values': scaler.data_max_.tolist()
    }
    with open(NORMALIZATION_PARAMS_JSON, 'w') as f:
        json.dump(normalization_params, f, indent=4)
    print(f"\nNormalization parameters saved to '{NORMALIZATION_PARAMS_JSON}'.")

    print("\nScript finished successfully!")
    print("You can now use 'sound_features_normalized.csv' and 'sound_normalization_params.json' in your PyTorch training script.")

if __name__ == '__main__':
    main()

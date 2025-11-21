import os
import glob
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
# class weighting removed - training will run without compute_class_weight
import tensorflow as tf


LABEL_MAP = {"calm": 0, "excited": 1}


def load_csv_signal(path):
    """Load voltage signal from CSV.

    Rules:
    - Prefer columns whose name suggests voltage/signal (e.g. 'voltage', 'volt', 'v', 'ecg', 'signal', 'lead').
    - If a 'time' column exists, drop it and use the remaining numeric columns.
    - If multiple candidate numeric columns remain, pick the one with largest variance.
    - Return a 1D numpy float array (voltage/time series) or raise if none found.
    """
    df = pd.read_csv(path)
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        numeric = df.apply(pd.to_numeric, errors='coerce')
        numeric = numeric.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        raise ValueError(f"No numeric data found in {path}")
    
    # Prefer explicit voltage/signal-like column names
    name_pattern = re.compile(r'volt|^v$|ecg|signal|lead|amplitude', re.IGNORECASE)
    candidates = [c for c in numeric.columns if name_pattern.search(str(c))]
    if candidates:
        col = candidates[0]
        signal = numeric[col].to_numpy().astype(float)
        return signal

    # If there's a time column, drop it and use the remaining numeric columns
    time_pattern = re.compile(r'time|^t$|timestamp', re.IGNORECASE)
    time_cols = [c for c in numeric.columns if time_pattern.search(str(c))]
    if len(numeric.columns) > 1 and time_cols:
        numeric = numeric.drop(columns=time_cols, errors='ignore')

    # If multiple numeric columns remain, pick the one with largest variance
    if numeric.shape[1] > 1:
        variances = numeric.var(axis=0, skipna=True)
        col = variances.idxmax()
        signal = numeric[col].to_numpy().astype(float)
    else:
        signal = numeric.iloc[:, 0].to_numpy().astype(float)

    return signal


def make_chunks(signal, chunk_size=1000):
    """Split signal into non-overlapping fixed-size chunks"""
    chunks = []
    n = len(signal)
    for i in range(0, n, chunk_size):
        chunk = signal[i:i+chunk_size]
        if len(chunk) == chunk_size:
            chunks.append(chunk)
    return chunks


def extract_features_from_chunk(chunk):
    """Compute simple time-domain + FFT band-energy features for a 1D chunk.

    Returns a 1D numpy array of features.
    Features: mean, std, median, min, max, ptp, rms, low_band_energy, mid_band_energy, high_band_energy
    """
    a = np.asarray(chunk, dtype=np.float32)
    mean = a.mean()
    std = a.std()
    median = np.median(a)
    amin = a.min()
    amax = a.max()
    ptp = amax - amin
    rms = np.sqrt(np.mean(a * a))

    # FFT band energies (relative): split rfft magnitude-squared into 3 bands
    try:
        fft = np.fft.rfft(a)
        psd = np.abs(fft) ** 2
        n = psd.shape[0]
        if n < 3:
            low = mid = high = 0.0
        else:
            low = psd[: max(1, n // 10)].sum()
            mid = psd[max(1, n // 10): max(1, n // 2)].sum()
            high = psd[max(1, n // 2):].sum()
            total = low + mid + high + 1e-12
            low = low / total
            mid = mid / total
            high = high / total
    except Exception:
        low = mid = high = 0.0

    # Additional time-domain higher-order moments
    try:
        skew = np.mean((a - mean) ** 3) / (std ** 3 + 1e-12)
    except Exception:
        skew = 0.0
    try:
        kurt = np.mean((a - mean) ** 4) / (std ** 4 + 1e-12) - 3.0
    except Exception:
        kurt = 0.0

    # Spectral centroid, dominant frequency, spectral entropy, spectral flatness
    try:
        freqs = np.fft.rfftfreq(a.shape[0], d=1.0)
        psd_sum = psd.sum() + 1e-12
        spectral_centroid = (freqs * psd).sum() / psd_sum
        dominant_idx = int(np.argmax(psd)) if psd.size > 0 else 0
        dominant_freq = float(freqs[dominant_idx]) if freqs.size > dominant_idx else 0.0
        # spectral entropy (normalized)
        p = psd / psd_sum
        spectral_entropy = -float(np.sum(p * np.log(p + 1e-12))) / (np.log(p.size + 1e-12))
        # spectral flatness: geometric mean / arithmetic mean
        geo_mean = float(np.exp(np.mean(np.log(psd + 1e-12))))
        arith_mean = float(np.mean(psd) + 1e-12)
        spectral_flatness = geo_mean / arith_mean
    except Exception:
        spectral_centroid = 0.0
        dominant_freq = 0.0
        spectral_entropy = 0.0
        spectral_flatness = 0.0

    return np.array([
        mean, std, median, amin, amax, ptp, rms, low, mid, high,
        skew, kurt, spectral_centroid, spectral_entropy, dominant_freq, spectral_flatness
    ], dtype=np.float32)


def extract_features_from_chunks(chunks):
    """Vectorized extraction for a list/array of chunks."""
    feats = [extract_features_from_chunk(c) for c in chunks]
    return np.vstack(feats)


def build_dataset(csv_dir='csvs', chunk_size=1000):
    """Load all CSVs, chunk them, and return X, y with labels from filenames"""
    X = []
    y = []
    files = glob.glob(os.path.join(csv_dir, '*.csv'))
    if len(files) == 0:
        raise FileNotFoundError(f"No CSVs found in {csv_dir}")
    for f in files:
        name = os.path.basename(f).lower()
        label = None
        for k in LABEL_MAP.keys():
            if name.startswith(k):
                label = LABEL_MAP[k]
                break
        if label is None:
            for k in LABEL_MAP.keys():
                if k in name:
                    label = LABEL_MAP[k]
                    break
        if label is None:
            print(f"Skipping {f} (unknown label)")
            continue
        try:
            signal = load_csv_signal(f)
        except Exception as e:
            print(f"Failed to read {f}: {e}")
            continue

        # Drop numeric NaNs from the signal to avoid NaN propagation in normalization/training
        n_nans = int(np.isnan(signal).sum())
        if n_nans > 0:
            print(f"Found {n_nans} NaNs in {f}; dropping NaN samples before chunking")
            signal = signal[~np.isnan(signal)]

        # If file becomes too short after NaN removal, skip it
        if len(signal) < chunk_size:
            print(f"Skipping {f}: length after NaN removal {len(signal)} < chunk_size {chunk_size}")
            continue

        # Per-file z-score normalization: normalize the whole recording before chunking
        # This preserves relative amplitude/baseline differences across chunks from the same file
        try:
            s_mean = float(np.mean(signal))
            s_std = float(np.std(signal))
            if s_std == 0.0:
                signal = signal - s_mean
            else:
                signal = (signal - s_mean) / (s_std + 1e-8)
        except Exception:
            # On any failure, fallback to original signal
            pass

        chunks = make_chunks(signal, chunk_size=chunk_size)
        for c in chunks:
            X.append(c)
            y.append(label)
    X = np.array(X)
    y = np.array(y)
    # Also return per-chunk engineered features
    X_feats = extract_features_from_chunks(X)
    return X, X_feats, y


def build_conv1d_model(input_length=1000):
    """Build Conv1D model: Conv blocks → Global pooling → Dense layers
    
    Input: (batch, 1000, 1) for raw ECG signals
    Output: (batch, 3) class probabilities
    """
    inputs = tf.keras.layers.Input(shape=(input_length, 1))
    
    # Block 1: 64 filters, kernel=7
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=7, activation='relu', 
                                padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool1D(pool_size=2)(x)  # (batch, 500, 64)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Block 2: 128 filters, kernel=5
    x = tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu', 
                                padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool1D(pool_size=2)(x)  # (batch, 250, 128)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Block 3: 256 filters, kernel=3
    x = tf.keras.layers.Conv1D(filters=256, kernel_size=3, activation='relu', 
                                padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool1D(pool_size=2)(x)  # (batch, 125, 256)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Global Average Pooling: (batch, 125, 256) → (batch, 256)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    # Build a feature branch input placeholder for engineered features
    # feature input now matches extended engineered feature vector
    feat_input = tf.keras.layers.Input(shape=(16,), name='feat_input')

    # Dense head for conv features
    x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    # Small dense head for engineered features
    f = tf.keras.layers.Dense(64, activation='relu')(feat_input)
    f = tf.keras.layers.BatchNormalization()(f)

    # Concatenate learned conv features with engineered features
    combined = tf.keras.layers.concatenate([x, f])

    # Final classification head
    combined = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(combined)
    # Single sigmoid output for binary classification (calm vs excited)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(combined)
    
    model = tf.keras.Model(inputs=[inputs, feat_input], outputs=outputs)
    # Use a smaller learning rate and gradient clipping to improve numerical stability
    # Add AUC and PR-AUC metrics for monitoring
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.AUC(name='pr_auc', curve='PR')]
    )
    return model


def build_conv1d_model_overfit(input_length=1000):
    """Build a high-capacity model variant with minimal regularization to force overfitting.

    This removes L2 and Dropout so the network can memorize a very small dataset.
    """
    inputs = tf.keras.layers.Input(shape=(input_length, 1))

    x = tf.keras.layers.Conv1D(filters=128, kernel_size=7, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool1D(pool_size=2)(x)

    x = tf.keras.layers.Conv1D(filters=256, kernel_size=5, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool1D(pool_size=2)(x)

    x = tf.keras.layers.Conv1D(filters=512, kernel_size=3, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool1D(pool_size=2)(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    # Overfit model: also accept engineered features and concatenate (keeps high capacity)
    feat_input = tf.keras.layers.Input(shape=(16,), name='feat_input')
    f = tf.keras.layers.Dense(64, activation='relu')(feat_input)
    f = tf.keras.layers.BatchNormalization()(f)
    combined = tf.keras.layers.concatenate([x, f])
    combined = tf.keras.layers.Dense(256, activation='relu')(combined)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(combined)

    model = tf.keras.Model(inputs=[inputs, feat_input], outputs=outputs)
    # Higher lr to accelerate memorization (small dataset); no clipnorm here
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.AUC(name='pr_auc', curve='PR')]
    )
    return model


def train_and_evaluate(csv_dir='csvs', chunk_size=1000, epochs=30, batch_size=8, save_path='ecg_model.h5', overfit=False):
    """Train Conv1D model on ECG chunks with class weighting to handle imbalance

    If `overfit=True`, select a tiny balanced subset (few examples per class) and train
    with a high-capacity, low-regularization model to force memorization. This is
    useful to check whether the model has capacity and whether data/labels are learnable.
    """
    print('Building dataset...')
    X, X_feats, y = build_dataset(csv_dir, chunk_size)
    if len(X) == 0:
        raise RuntimeError('No training samples found. Check CSV files and chunk_size.')

    print(f'Loaded {len(X)} samples')

    # Preprocessing: cast to float32. Files were normalized per-file at load time,
    # so we do NOT apply per-chunk z-score normalization here.
    X = X.astype(np.float32)

    # CRITICAL: Reshape for Conv1D (batch, timesteps, channels)
    X = X.reshape((-1, chunk_size, 1))
    print(f'X shape after reshape: {X.shape}')
    print(f'X_feats shape: {X_feats.shape}')
    
    # Train-test split with stratification
    X_train, X_test, Xf_train, Xf_test, y_train, y_test = train_test_split(
        X, X_feats, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f'Training samples: {len(X_train)}, Test samples: {len(X_test)}')
    print(f'Class distribution in train: {np.bincount(y_train)}')
    
    # No class weights — train normally unless overfitting is requested
    if overfit:
        # Build tiny balanced subset from training data (N per class)
        N = 8
        chosen_idx = []
        for cls in np.unique(y_train):
            idx = np.where(y_train == cls)[0]
            if len(idx) < N:
                raise RuntimeError(f'Not enough examples to overfit for class {cls} (need {N})')
            chosen = np.random.choice(idx, size=N, replace=False)
            chosen_idx.extend(chosen.tolist())
        chosen_idx = np.array(chosen_idx)
        X_train_small = X_train[chosen_idx]
        y_train_small = y_train[chosen_idx]
        print(f'Overfit mode: training on {len(X_train_small)} examples (balanced {N} per class)')

        model = build_conv1d_model_overfit(input_length=chunk_size)
        print('\nModel architecture (overfit mode):')
        model.summary()

        # Train longer to memorize
        model.fit([X_train_small, Xf_train[chosen_idx]], y_train_small, validation_data=([X_test, Xf_test], y_test),
              epochs=epochs, batch_size=max(1, batch_size//2), shuffle=True)
    else:
        # Build and train standard model
        model = build_conv1d_model(input_length=chunk_size)
        print('\nModel architecture:')
        model.summary()

        print(f'\nTraining without class weights (epochs={epochs}, batch_size={batch_size})...')
        model.fit(
            [X_train, Xf_train], y_train,
            validation_data=([X_test, Xf_test], y_test),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True
        )
    
    # Evaluate
    print('\nEvaluating on test set...')
    eval_results = model.evaluate([X_test, Xf_test], y_test, verbose=0)
    # eval_results: [loss, accuracy, auc?, pr_auc?] depending on compiled metrics
    loss = eval_results[0]
    acc = eval_results[1] if len(eval_results) > 1 else None
    auc_val = eval_results[2] if len(eval_results) > 2 else None
    pr_auc_val = eval_results[3] if len(eval_results) > 3 else None
    print_str = f'Test loss: {loss:.4f}'
    if acc is not None:
        print_str += f", Test accuracy: {acc:.4f}"
    if auc_val is not None:
        print_str += f", Test AUC: {auc_val:.4f}"
    if pr_auc_val is not None:
        print_str += f", Test PR-AUC: {pr_auc_val:.4f}"
    print(print_str)

    # Predictions
    y_pred_probs = model.predict([X_test, Xf_test], verbose=0)

    # Handle single-output sigmoid model: y_pred_probs shape -> (n,1)
    if y_pred_probs.ndim > 1 and y_pred_probs.shape[1] == 1:
        pos_probs = y_pred_probs.ravel()
        y_pred = (pos_probs >= 0.5).astype(int)
    else:
        # fallback for multi-output softmax (not expected now)
        pos_probs = y_pred_probs[:, 1]
        y_pred = np.argmax(y_pred_probs, axis=1)

    print('\nClassification report:')
    print(classification_report(y_test, y_pred, target_names=list(LABEL_MAP.keys())))

    # Compute ROC-AUC and PR-AUC (sklearn) for the binary positive class if possible
    try:
        if len(np.unique(y_test)) == 2:
            roc = roc_auc_score(y_test, pos_probs)
            ap = average_precision_score(y_test, pos_probs)
            print(f"\nROC AUC: {roc:.4f}, PR AUC (avg precision): {ap:.4f}")
        else:
            print('\nROC AUC / PR AUC not computed: single-class in test set')
    except Exception as e:
        print(f"Failed to compute sklearn AUC metrics: {e}")
    
    print('\nConfusion matrix:')
    print(confusion_matrix(y_test, y_pred))
    
    # Save model
    model.save(save_path)
    print(f'\nModel saved to {save_path}')


def infer_from_csv(model_path, csv_path, chunk_size=1000):
    """Load trained model and predict on a single CSV file
    
    Returns: (predictions, probabilities) where predictions are class labels (0/1/2)
    """
    model = tf.keras.models.load_model(model_path)
    signal = load_csv_signal(csv_path)
    # Drop NaNs and normalize per-file like training
    if np.isnan(signal).any():
        signal = signal[~np.isnan(signal)]
    if len(signal) < chunk_size:
        raise RuntimeError('CSV too short to produce any full chunks')
    try:
        s_mean = float(np.mean(signal))
        s_std = float(np.std(signal))
        if s_std == 0.0:
            signal = signal - s_mean
        else:
            signal = (signal - s_mean) / (s_std + 1e-8)
    except Exception:
        pass

    chunks = make_chunks(signal, chunk_size=chunk_size)
    if len(chunks) == 0:
        raise RuntimeError('CSV too short to produce any full chunks after normalization')

    # Preprocess same as training (chunks are created from per-file-normalized signal)
    X = np.array(chunks).astype(np.float32)
    X = X.reshape((-1, chunk_size, 1))

    # Compute engineered features for these chunks
    X_feats = extract_features_from_chunks(chunks)

    # Predict
    probs = model.predict([X, X_feats], verbose=0)
    if probs.ndim > 1 and probs.shape[1] == 1:
        probs_flat = probs.ravel()
        preds = (probs_flat >= 0.5).astype(int)
    else:
        probs_flat = probs.ravel()
        preds = np.argmax(probs, axis=1)

    # Map predictions back to class labels
    inv_map = {v: k for k, v in LABEL_MAP.items()}
    results = [inv_map[int(p)] for p in preds]
    return results, probs_flat


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train Conv1D ECG classifier.')
    parser.add_argument('--csv_dir', type=str, default='csvs', help='Directory with CSV files')
    parser.add_argument('--chunk_size', type=int, default=1000, help='Chunk size (samples per example)')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--save', type=str, default='ecg_model.h5', help='Model save path')
    parser.add_argument('--overfit', action='store_true', help='Overfit on small balanced subset')
    args = parser.parse_args()

    train_and_evaluate(
        csv_dir=args.csv_dir,
        chunk_size=args.chunk_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_path=args.save,
        overfit=args.overfit
    )

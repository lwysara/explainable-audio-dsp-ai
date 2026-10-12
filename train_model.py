from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import joblib
from scipy.signal import butter, sosfilt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "cats_dogs"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "data"
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "cat_dog_model.joblib"
FEATURE_DATA_PATH = OUTPUT_DIR / "feature_dataset.csv"
SR = 22050
LOW, HIGH = 300, 4000
EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aiff", ".aif"}
FEATURES = [
    "RMS", "Peak", "ZCR", "Crest_Factor", "Spectral_Centroid",
    "Spectral_Bandwidth", "Spectral_Rolloff", "Spectral_Flatness",
    "Dominant_Frequency",
] + [f"MFCC_{i}" for i in range(1, 14)]

def audio_files(folder):
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTS) if folder.exists() else []

def bandpass(y, sr):
    nyq = sr / 2
    lo, hi = max(LOW / nyq, 1e-5), min(HIGH / nyq, 0.99)
    if lo >= hi:
        return y
    return sosfilt(butter(5, [lo, hi], btype="bandpass", output="sos"), y)

def extract_features(y, sr):
    rms = float(np.sqrt(np.mean(y ** 2)))
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    crest = float(peak / (rms + 1e-12))
    mag = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), d=1 / sr)
    dominant = float(freqs[np.argmax(mag)]) if len(mag) else 0.0
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    out = {
        "RMS": rms, "Peak": peak, "ZCR": zcr, "Crest_Factor": crest,
        "Spectral_Centroid": centroid, "Spectral_Bandwidth": bandwidth,
        "Spectral_Rolloff": rolloff, "Spectral_Flatness": flatness,
        "Dominant_Frequency": dominant,
    }
    out.update({f"MFCC_{i+1}": float(np.mean(mfcc[i])) for i in range(13)})
    return out

def main():
    cats = audio_files(DATA / "cats")
    dogs = audio_files(DATA / "dogs")
    if not cats or not dogs:
        cats, dogs = audio_files(DATA / "cat"), audio_files(DATA / "dog")
    print(f"Dataset folder: {DATA}")
    print(f"Cat files: {len(cats)} | Dog files: {len(dogs)}")
    if not cats or not dogs:
        raise SystemExit("找不到資料。請將音檔放在 cats_dogs/cats/ 與 cats_dogs/dogs/。")

    rows, failures = [], []
    for label, paths in [("cat", cats), ("dog", dogs)]:
        for i, path in enumerate(paths, 1):
            try:
                y, sr = librosa.load(str(path), sr=SR, mono=True)
                if len(y) < 512:
                    failures.append((str(path), "音檔太短"))
                    continue
                feat = extract_features(bandpass(y, sr), sr)
                feat["Label"] = label
                feat["File"] = str(path.relative_to(DATA))
                rows.append(feat)
            except Exception as exc:
                failures.append((str(path), str(exc)))
            if i % 100 == 0:
                print(f"{label}: {i}/{len(paths)}")

    df = pd.DataFrame(rows)
    if df.empty or df["Label"].nunique() != 2:
        raise SystemExit("沒有足夠的可用音檔，請檢查音檔格式。")
    counts = df["Label"].value_counts()
    print("\nUsable samples:\n", counts.to_string())
    if counts.min() < 2:
        raise SystemExit("每個類別至少需要 2 個可用音檔。")

    # Save the reference features for cloud-side Cat-vs-Dog distributions.
    df.to_csv(FEATURE_DATA_PATH, index=False)
    X, y = df[FEATURES].astype(float), df["Label"].astype(str)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    model = RandomForestClassifier(
        n_estimators=400, random_state=42, class_weight="balanced", n_jobs=-1
    )
    model.fit(Xtr_s, ytr)
    pred = model.predict(Xte_s)
    acc = float(accuracy_score(yte, pred))
    cm = confusion_matrix(yte, pred, labels=["cat", "dog"])
    report = classification_report(
        yte, pred, labels=["cat", "dog"], output_dict=True, zero_division=0
    )
    perm = permutation_importance(
        model, Xte_s, yte, n_repeats=5, random_state=42,
        scoring="f1_macro", n_jobs=-1
    )
    perm_rows = [
        {"Feature": FEATURES[i], "Importance": float(perm.importances_mean[i]),
         "Std": float(perm.importances_std[i])}
        for i in range(len(FEATURES))
    ]
    perm_rows.sort(key=lambda row: row["Importance"], reverse=True)

    bundle = {
        "model": model,
        "scaler": scaler,
        "feature_keys": FEATURES,
        "sample_rate": SR,
        "lowcut": LOW,
        "highcut": HIGH,
        "training_medians": Xtr.median().to_dict(),
        "test_accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "permutation_importance": perm_rows,
        "classes": [str(c) for c in model.classes_],
    }
    joblib.dump(bundle, MODEL_PATH)

    print(f"\nTest accuracy: {acc:.4f}")
    print("Confusion matrix (rows=actual Cat/Dog, columns=predicted Cat/Dog):\n", cm)
    print("\nClassification report:\n", classification_report(yte, pred, zero_division=0))
    print(f"\nModel saved: {MODEL_PATH}")
    print(f"Feature dataset saved: {FEATURE_DATA_PATH}")
    if failures:
        print(f"Skipped {len(failures)} files; first 10:", failures[:10])
    print("\nNext: streamlit run app.py")

if __name__ == "__main__":
    main()

from pathlib import Path
import io
import numpy as np
import pandas as pd
import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import joblib
from scipy.signal import butter, sosfilt

st.set_page_config(page_title="Cat & Dog Audio DSP AI", page_icon="🐱", layout="wide")
st.title("🐱🐶 Cat & Dog Audio DSP + AI")
st.caption("上傳音訊 → Waveform / FFT / Spectrogram / MFCC → Random Forest 預測與可解釋特徵分析")

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "cat_dog_model.joblib"
FEATURE_DATA_PATH = ROOT / "data" / "feature_dataset.csv"
SR = 22050
LOWCUT, HIGHCUT = 300, 4000
FEATURES = [
    "RMS", "Peak", "ZCR", "Crest_Factor", "Spectral_Centroid",
    "Spectral_Bandwidth", "Spectral_Rolloff", "Spectral_Flatness",
    "Dominant_Frequency",
] + [f"MFCC_{i}" for i in range(1, 14)]

def bandpass(y, sr, low=LOWCUT, high=HIGHCUT):
    nyq = sr / 2
    lo, hi = max(low / nyq, 1e-5), min(high / nyq, 0.99)
    if lo >= hi:
        return y
    return sosfilt(butter(5, [lo, hi], btype="bandpass", output="sos"), y)

def extract_features(y, sr):
    if len(y) == 0:
        raise ValueError("音訊沒有可分析的樣本。")
    rms = float(np.sqrt(np.mean(y ** 2)))
    peak = float(np.max(np.abs(y)))
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

@st.cache_resource(show_spinner=False)
def load_model_bundle():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

@st.cache_data(show_spinner=False)
def load_feature_dataset():
    if not FEATURE_DATA_PATH.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(FEATURE_DATA_PATH)
        if "Label" in df.columns and all(c in df.columns for c in FEATURES):
            return df
    except Exception:
        pass
    return pd.DataFrame()

bundle = load_model_bundle()
if bundle is None:
    st.error("找不到訓練好的模型檔 `models/cat_dog_model.joblib`。")
    st.markdown("請先在自己的 Mac 上放入 `cats_dogs/cats/` 與 `cats_dogs/dogs/`，執行 `python train_model.py`，再把產生的模型檔上傳至 GitHub。")
    st.code("python train_model.py\nstreamlit run app.py", language="bash")
    st.stop()

model = bundle["model"]
scaler = bundle["scaler"]
feature_names = bundle.get("feature_keys", FEATURES)
sr_model = int(bundle.get("sample_rate", SR))
lowcut = float(bundle.get("lowcut", LOWCUT))
highcut = float(bundle.get("highcut", HIGHCUT))
train_df = load_feature_dataset()
if train_df.empty and bundle.get("training_rows"):
    train_df = pd.DataFrame(bundle["training_rows"])

st.sidebar.header("模型設定")
st.sidebar.write(f"模型取樣率：{sr_model} Hz")
st.sidebar.write(f"特徵濾波範圍：{lowcut:g}–{highcut:g} Hz")
st.sidebar.caption("預測機率代表模型對這筆音訊的估計，不等於整體模型準確率。")

upload = st.file_uploader("📁 上傳音訊", type=["wav", "mp3", "flac", "m4a", "ogg", "aiff", "aif"])
record = st.audio_input("🎙️ 使用麥克風錄音（若瀏覽器與裝置支援）") if hasattr(st, "audio_input") else None
source = upload if upload is not None else record
if source is None:
    st.info("請先上傳貓叫或狗叫音訊，或使用上方錄音功能。")
    st.stop()

raw_bytes = source.getvalue()
st.audio(raw_bytes)
try:
    y_raw, sr = librosa.load(io.BytesIO(raw_bytes), sr=sr_model, mono=True)
    if len(y_raw) < int(0.25 * sr):
        st.error("音訊太短，請上傳至少約 0.25 秒的音訊。")
        st.stop()

    y = bandpass(y_raw, sr, lowcut, highcut)
    feats = extract_features(y, sr)
    x = pd.DataFrame([[feats[name] for name in feature_names]], columns=feature_names)
    x_scaled = scaler.transform(x)
    pred = str(model.predict(x_scaled)[0]).lower()
    probs = model.predict_proba(x_scaled)[0]
    classes = [str(c).lower() for c in model.classes_]
    prob_map = {classes[i]: float(probs[i]) for i in range(len(classes))}
    confidence = prob_map.get(pred, float(np.max(probs)))

    st.subheader("AI 預測結果")
    c1, c2, c3 = st.columns(3)
    c1.metric("辨識結果", "🐱 Cat／貓叫聲" if pred == "cat" else "🐶 Dog／狗叫聲" if pred == "dog" else pred.upper())
    c2.metric("Cat 機率", f"{prob_map.get('cat', 0)*100:.1f}%")
    c3.metric("Dog 機率", f"{prob_map.get('dog', 0)*100:.1f}%")
    st.progress(float(np.clip(confidence, 0, 1)), text=f"此音訊的預測類別機率：{confidence*100:.1f}%")

    st.subheader("① Audio DSP 分析")
    t1, t2, t3, t4 = st.tabs(["🌊 Waveform", "📈 FFT", "🔥 Spectrogram", "🎼 MFCC"])
    with t1:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(np.arange(len(y_raw))/sr, y_raw, linewidth=.65, label="Raw")
        ax.plot(np.arange(len(y))/sr, y, linewidth=.65, label="Bandpass filtered")
        ax.set(xlabel="Time (s)", ylabel="Amplitude", title="Time-Domain Waveform")
        ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
        st.pyplot(fig); plt.close(fig)
    with t2:
        spectrum = np.abs(np.fft.rfft(y_raw))
        freqs = np.fft.rfftfreq(len(y_raw), d=1/sr)
        db = 20*np.log10(spectrum/(spectrum.max()+1e-12)+1e-12)
        dominant = float(freqs[np.argmax(spectrum)])
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(freqs, db, linewidth=.65)
        ax.axvline(dominant, linestyle="--", label=f"Dominant = {dominant:.1f} Hz")
        ax.set_xlim(0, min(10000, sr/2))
        ax.set(xlabel="Frequency (Hz)", ylabel="Magnitude (dB)", title="FFT Spectrum")
        ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
        st.pyplot(fig); plt.close(fig)
    with t3:
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="hz", ax=ax, cmap="magma")
        ax.set_ylim(0, min(10000, sr/2)); ax.set_title("STFT Spectrogram")
        fig.colorbar(img, ax=ax, format="%+2.0f dB"); fig.tight_layout()
        st.pyplot(fig); plt.close(fig)
    with t4:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(mfcc, x_axis="time", ax=ax, cmap="coolwarm")
        ax.set_title("MFCC"); ax.set_ylabel("MFCC coefficient")
        fig.colorbar(img, ax=ax); fig.tight_layout()
        st.pyplot(fig); plt.close(fig)

    st.subheader("② DSP Feature Values")
    st.dataframe(pd.DataFrame({"Feature": list(feats), "Value": list(feats.values())}), use_container_width=True)

    st.subheader("③ Global Feature Importance")
    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False)
    top = importance.head(15).sort_values("Importance")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(top["Feature"], top["Importance"])
    ax.set_xlabel("Random Forest impurity importance")
    ax.grid(axis="x", alpha=.2); fig.tight_layout()
    st.pyplot(fig); plt.close(fig)
    with st.expander("查看完整 Global Feature Importance 表"):
        st.dataframe(importance, use_container_width=True)
    st.caption("Global Feature Importance 描述模型整體使用特徵的程度，不代表物理因果關係。")

    perm_rows = bundle.get("permutation_importance")
    if perm_rows:
        st.subheader("④ Permutation Importance")
        perm = pd.DataFrame(perm_rows).sort_values("Importance", ascending=False)
        q = perm.head(15).sort_values("Importance")
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(q["Feature"], q["Importance"], xerr=q.get("Std"))
        ax.set_xlabel("Mean decrease in validation score")
        ax.grid(axis="x", alpha=.2); fig.tight_layout()
        st.pyplot(fig); plt.close(fig)
        with st.expander("查看完整 Permutation Importance 表"):
            st.dataframe(perm, use_container_width=True)

    st.subheader("⑤ Cat vs Dog Feature Distribution")
    if not train_df.empty and "Label" in train_df.columns:
        selected_feature = st.selectbox(
            "選擇 DSP Feature 比較 Cat 與 Dog",
            feature_names,
            index=feature_names.index("Spectral_Centroid") if "Spectral_Centroid" in feature_names else 0,
        )
        catv = train_df.loc[train_df["Label"].astype(str).str.lower()=="cat", selected_feature].dropna()
        dogv = train_df.loc[train_df["Label"].astype(str).str.lower()=="dog", selected_feature].dropna()
        fig, ax = plt.subplots(figsize=(9, 5))
        if len(catv): ax.hist(catv, bins=30, alpha=.55, label="Cat")
        if len(dogv): ax.hist(dogv, bins=30, alpha=.55, label="Dog")
        if len(catv): ax.axvline(catv.mean(), linestyle="--", label="Cat Mean")
        if len(dogv): ax.axvline(dogv.mean(), linestyle="--", label="Dog Mean")
        ax.set_xlabel(selected_feature); ax.set_ylabel("Samples")
        ax.set_title(f"Cat vs Dog Distribution: {selected_feature}")
        ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
        st.pyplot(fig); plt.close(fig)
        a, b, c = st.columns(3)
        a.metric("Cat samples", len(catv)); b.metric("Dog samples", len(dogv))
        gap = abs(float(catv.mean()) - float(dogv.mean())) if len(catv) and len(dogv) else float("nan")
        c.metric("Absolute Mean Gap", f"{gap:.4g}" if np.isfinite(gap) else "N/A")
    else:
        st.info("尚未找到 `data/feature_dataset.csv`。AI 預測與 DSP 分析仍可使用；若要保留 Cat vs Dog Feature Distribution，請在本機執行 `python train_model.py`，並一併上傳產生的 CSV。")

    st.subheader("⑥ Local Feature Impact")
    st.write("逐一將單一特徵替換為訓練資料中位數，觀察目前預測類別機率的變化。這是近似模型解釋，不是因果證明。")
    medians = bundle.get("training_medians", {})
    if medians:
        base_p = confidence
        pred_class_index = classes.index(pred)
        impacts = []
        for i, name in enumerate(feature_names):
            altered = x.copy()
            altered.loc[0, name] = float(medians[name])
            altered_prob = float(model.predict_proba(scaler.transform(altered))[0][pred_class_index])
            impacts.append({
                "Feature": name,
                "Original_Value": float(x.loc[0, name]),
                "Training_Median": float(medians[name]),
                "Impact": base_p - altered_prob,
            })
        local = pd.DataFrame(impacts)
        local["Abs_Impact"] = local["Impact"].abs()
        local = local.sort_values("Abs_Impact", ascending=False)
        chart = local.head(10).sort_values("Impact")
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(chart["Feature"], chart["Impact"]); ax.axvline(0, linestyle="--")
        ax.set_xlabel("Original class probability − probability after median replacement")
        ax.grid(axis="x", alpha=.2); fig.tight_layout()
        st.pyplot(fig); plt.close(fig)
        shown = local.head(15).copy()
        shown["Impact (%)"] = (shown["Impact"]*100).round(2)
        st.dataframe(shown[["Feature", "Original_Value", "Training_Median", "Impact (%)"]], use_container_width=True)
    else:
        st.warning("模型檔沒有 training_medians，請重新執行 train_model.py 產生新版模型。")

    st.subheader("⑦ 模型評估（訓練時保存）")
    if "test_accuracy" in bundle:
        st.metric("Hold-out Test Accuracy", f"{float(bundle['test_accuracy'])*100:.2f}%")
        st.caption("此為訓練時保留測試集的結果，不是目前上傳音訊的正確率。")
    if "confusion_matrix" in bundle:
        cm = np.asarray(bundle["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Cat", "Dog"]); ax.set_yticklabels(["Cat", "Dog"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix")
        for i in range(min(2, cm.shape[0])):
            for j in range(min(2, cm.shape[1])):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center")
        fig.colorbar(im, ax=ax); fig.tight_layout()
        st.pyplot(fig); plt.close(fig)
    if "classification_report" in bundle:
        with st.expander("查看 Classification Report"):
            st.dataframe(pd.DataFrame(bundle["classification_report"]).T, use_container_width=True)

except Exception as exc:
    st.error(f"音訊分析失敗：{exc}")
    st.exception(exc)

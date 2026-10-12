import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfilt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ============================================================
# PAGE
# ============================================================
st.set_page_config(
    page_title="Cat vs Dog Audio AI",
    page_icon="🐱",
    layout="wide"
)

st.title("🐱 Cat / 🐶 Dog Audio AI")
st.caption("Upload / Microphone → DSP → Random Forest → Cat or Dog")

# ============================================================
# FEATURES
# ============================================================
FEATURES = [
    "RMS",
    "Peak",
    "ZCR",
    "Crest_Factor",
    "Spectral_Centroid",
    "Spectral_Bandwidth",
    "Spectral_Rolloff",
    "Spectral_Flatness",
    "Dominant_Frequency"
] + [f"MFCC_{i+1}" for i in range(13)]


# ============================================================
# DSP
# ============================================================
def bandpass_filter(signal, sr, lowcut=300, highcut=4000):
    nyquist = sr / 2

    low = max(lowcut / nyquist, 0.001)
    high = min(highcut / nyquist, 0.99)

    if low >= high:
        return signal

    sos = butter(
        5,
        [low, high],
        btype="bandpass",
        output="sos"
    )

    return sosfilt(sos, signal)


def extract_features(signal, sr):

    signal = np.asarray(signal, dtype=np.float32)

    # normalize
    peak = np.max(np.abs(signal))

    if peak > 0:
        signal = signal / peak

    features = {}

    # ----------------------------
    # Time Domain
    # ----------------------------
    rms = np.sqrt(np.mean(signal ** 2))

    zcr = np.mean(
        librosa.feature.zero_crossing_rate(
            y=signal
        )
    )

    features["RMS"] = float(rms)
    features["Peak"] = float(np.max(np.abs(signal)))
    features["ZCR"] = float(zcr)
    features["Crest_Factor"] = float(
        features["Peak"] / (rms + 1e-12)
    )

    # ----------------------------
    # Frequency Domain
    # ----------------------------
    features["Spectral_Centroid"] = float(
        np.mean(
            librosa.feature.spectral_centroid(
                y=signal,
                sr=sr
            )
        )
    )

    features["Spectral_Bandwidth"] = float(
        np.mean(
            librosa.feature.spectral_bandwidth(
                y=signal,
                sr=sr
            )
        )
    )

    features["Spectral_Rolloff"] = float(
        np.mean(
            librosa.feature.spectral_rolloff(
                y=signal,
                sr=sr
            )
        )
    )

    features["Spectral_Flatness"] = float(
        np.mean(
            librosa.feature.spectral_flatness(
                y=signal
            )
        )
    )

    # ----------------------------
    # FFT
    # ----------------------------
    fft = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(
        len(signal),
        1 / sr
    )

    magnitude = np.abs(fft)

    if len(magnitude) > 1:
        idx = np.argmax(magnitude[1:]) + 1
        features["Dominant_Frequency"] = float(
            freqs[idx]
        )
    else:
        features["Dominant_Frequency"] = 0.0

    # ----------------------------
    # MFCC
    # ----------------------------
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=13
    )

    for i in range(13):
        features[f"MFCC_{i+1}"] = float(
            np.mean(mfcc[i])
        )

    return features


# ============================================================
# DATASET
# ============================================================
def find_files(folder):

    extensions = (
        ".wav",
        ".mp3",
        ".flac",
        ".m4a",
        ".ogg"
    )

    result = []

    if not os.path.exists(folder):
        return result

    for root, _, files in os.walk(folder):

        for filename in files:

            if filename.lower().endswith(
                extensions
            ):
                result.append(
                    os.path.join(root, filename)
                )

    return result


def get_cat_dog_files(dataset_path):

    cat_files = []
    dog_files = []

    # Supported structures:
    #
    # cats_dogs/
    #   cat/
    #   dog/
    #
    # or
    #
    # cats_dogs/
    #   cats/
    #   dogs/

    for name in os.listdir(dataset_path):

        full = os.path.join(
            dataset_path,
            name
        )

        if not os.path.isdir(full):
            continue

        lower = name.lower()

        if lower in ["cat", "cats"]:
            cat_files.extend(
                find_files(full)
            )

        elif lower in ["dog", "dogs"]:
            dog_files.extend(
                find_files(full)
            )

    return cat_files, dog_files


# ============================================================
# TRAIN MODEL
# ============================================================
@st.cache_resource
def train_model(dataset_path, max_files):

    cat_files, dog_files = get_cat_dog_files(
        dataset_path
    )

    cat_files = cat_files[:max_files]
    dog_files = dog_files[:max_files]

    if len(cat_files) == 0:
        raise ValueError(
            "找不到 Cat 音檔。\n"
            "請確認：cats_dogs/cat/*.wav"
        )

    if len(dog_files) == 0:
        raise ValueError(
            "找不到 Dog 音檔。\n"
            "請確認：cats_dogs/dog/*.wav"
        )

    X = []
    y = []

    total = len(cat_files) + len(dog_files)
    progress = st.progress(0)
    current = 0

    for label, files in [
        ("cat", cat_files),
        ("dog", dog_files)
    ]:

        for file_path in files:

            try:

                signal, sr = librosa.load(
                    file_path,
                    sr=22050,
                    mono=True
                )

                signal = bandpass_filter(
                    signal,
                    sr
                )

                features = extract_features(
                    signal,
                    sr
                )

                X.append([
                    features[name]
                    for name in FEATURES
                ])

                y.append(label)

            except Exception as e:
                print(
                    "Skip:",
                    file_path,
                    e
                )

            current += 1

            progress.progress(
                min(current / total, 1.0)
            )

    progress.empty()

    if len(X) < 10:
        raise ValueError(
            "有效音檔太少，至少需要 10 個以上。"
        )

    X = np.asarray(X)
    y = np.asarray(y)

    # train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    prediction = model.predict(
        X_test_scaled
    )

    accuracy = accuracy_score(
        y_test,
        prediction
    )

    cm = confusion_matrix(
        y_test,
        prediction,
        labels=["cat", "dog"]
    )

    report = classification_report(
        y_test,
        prediction,
        output_dict=True,
        zero_division=0
    )

    return (
        model,
        scaler,
        accuracy,
        cm,
        report
    )


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("⚙️ Model Settings")

dataset_path = st.sidebar.text_input(
    "Cats / Dogs Dataset Path",
    value="./cats_dogs"
)

max_files = st.sidebar.slider(
    "每類最多使用幾個音檔",
    10,
    1000,
    300,
    10
)

lowcut = st.sidebar.number_input(
    "Low Cut (Hz)",
    20,
    5000,
    300,
    50
)

highcut = st.sidebar.number_input(
    "High Cut (Hz)",
    500,
    10000,
    4000,
    100
)

train_button = st.sidebar.button(
    "🚀 Train Cat/Dog AI",
    type="primary"
)


# ============================================================
# TRAIN
# ============================================================
if train_button:

    if not os.path.exists(dataset_path):

        st.error(
            f"找不到 Dataset：{dataset_path}"
        )

        st.stop()

    with st.spinner(
        "正在讀取真實 Cat/Dog 音檔並訓練模型..."
    ):

        try:

            (
                model,
                scaler,
                accuracy,
                cm,
                report
            ) = train_model(
                dataset_path,
                max_files
            )

            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.accuracy = accuracy
            st.session_state.cm = cm
            st.session_state.report = report

            st.success(
                "✅ AI Model 訓練完成！"
            )

        except Exception as e:

            st.error(
                f"Training Error:\n{e}"
            )


# ============================================================
# MODEL RESULT
# ============================================================
if "accuracy" in st.session_state:

    st.subheader("📈 Model Validation")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Accuracy",
        f"{st.session_state.accuracy * 100:.2f}%"
    )

    report = st.session_state.report

    c2.metric(
        "Cat F1",
        f"{report['cat']['f1-score']:.3f}"
    )

    c3.metric(
        "Dog F1",
        f"{report['dog']['f1-score']:.3f}"
    )


# ============================================================
# AUDIO INPUT
# ============================================================
st.markdown("---")

st.header("🎙️ Audio Input")

tab_upload, tab_record = st.tabs([
    "📁 Upload Audio",
    "🎙️ Record Microphone"
])

audio_data = None
audio_name = None

with tab_upload:

    uploaded = st.file_uploader(
        "上傳 Cat / Dog 音檔",
        type=[
            "wav",
            "mp3",
            "flac",
            "m4a",
            "ogg"
        ]
    )

    if uploaded is not None:

        audio_data = uploaded.getvalue()
        audio_name = uploaded.name

        st.audio(
            audio_data
        )


with tab_record:

    st.write(
        "使用瀏覽器麥克風錄製。"
    )

    recorded = st.audio_input(
        "🎙️ 開始錄音"
    )

    if recorded is not None:

        audio_data = recorded.getvalue()
        audio_name = "microphone.wav"

        st.audio(
            audio_data
        )


# ============================================================
# PREDICTION
# ============================================================
if audio_data is not None:

    if "model" not in st.session_state:

        st.warning(
            "請先在左側按「🚀 Train Cat/Dog AI」。"
        )

        st.stop()

    # --------------------------------------------
    # LOAD
    # --------------------------------------------
    try:

        signal, sr = librosa.load(
            io.BytesIO(audio_data),
            sr=22050,
            mono=True
        )

    except Exception as e:

        st.error(
            f"無法讀取音檔：{e}"
        )

        st.stop()

    # --------------------------------------------
    # DSP
    # --------------------------------------------
    filtered_signal = bandpass_filter(
        signal,
        sr,
        lowcut,
        highcut
    )

    # --------------------------------------------
    # FEATURES
    # --------------------------------------------
    features = extract_features(
        filtered_signal,
        sr
    )

    X = np.array([
        features[name]
        for name in FEATURES
    ]).reshape(1, -1)

    scaler = st.session_state.scaler
    model = st.session_state.model

    X_scaled = scaler.transform(X)

    prediction = model.predict(
        X_scaled
    )[0]

    probabilities = model.predict_proba(
        X_scaled
    )[0]

    classes = list(model.classes_)

    cat_probability = probabilities[
        classes.index("cat")
    ]

    dog_probability = probabilities[
        classes.index("dog")
    ]

    confidence = max(
        cat_probability,
        dog_probability
    )

    # ========================================================
    # RESULT
    # ========================================================
    st.markdown("---")

    st.header("🤖 AI Prediction")

    c1, c2, c3 = st.columns(
        [2, 1, 1]
    )

    with c1:

        if prediction == "cat":

            st.success(
                "## 🐱 CAT / 貓叫聲"
            )

        else:

            st.warning(
                "## 🐶 DOG / 狗叫聲"
            )

    with c2:

        st.metric(
            "🐱 Cat",
            f"{cat_probability * 100:.1f}%"
        )

    with c3:

        st.metric(
            "🐶 Dog",
            f"{dog_probability * 100:.1f}%"
        )

    st.subheader(
        "🎯 Confidence"
    )

    st.progress(
        float(confidence)
    )

    st.write(
        f"Confidence: **{confidence * 100:.2f}%**"
    )

    # ========================================================
    # DSP
    # ========================================================
    st.markdown("---")

    st.header(
        "📊 Audio DSP Analysis"
    )

    dsp_wave, dsp_fft, dsp_spec, dsp_mfcc = st.tabs([
        "🌊 Waveform",
        "📈 FFT",
        "🔥 Spectrogram",
        "🎼 MFCC"
    ])

    # --------------------------------------------
    # WAVEFORM
    # --------------------------------------------
    with dsp_wave:

        time = np.arange(
            len(signal)
        ) / sr

        fig, ax = plt.subplots(
            figsize=(12, 4)
        )

        ax.plot(
            time,
            signal,
            label="Raw"
        )

        ax.plot(
            time,
            filtered_signal,
            label="Filtered",
            alpha=0.7
        )

        ax.set_title(
            "Time Domain Waveform"
        )

        ax.set_xlabel(
            "Time (sec)"
        )

        ax.set_ylabel(
            "Amplitude"
        )

        ax.legend()
        ax.grid(True)

        st.pyplot(fig)

    # --------------------------------------------
    # FFT
    # --------------------------------------------
    with dsp_fft:

        fft = np.fft.rfft(
            filtered_signal
        )

        freqs = np.fft.rfftfreq(
            len(filtered_signal),
            1 / sr
        )

        magnitude_db = (
            20 *
            np.log10(
                np.abs(fft) + 1e-12
            )
        )

        fig, ax = plt.subplots(
            figsize=(12, 4)
        )

        ax.plot(
            freqs,
            magnitude_db
        )

        ax.axvline(
            features[
                "Dominant_Frequency"
            ],
            linestyle="--",
            label=(
                "Dominant = "
                f"{features['Dominant_Frequency']:.1f} Hz"
            )
        )

        ax.set_xlim(
            0,
            min(
                10000,
                sr / 2
            )
        )

        ax.set_title(
            "FFT Frequency Spectrum"
        )

        ax.set_xlabel(
            "Frequency (Hz)"
        )

        ax.set_ylabel(
            "Magnitude (dB)"
        )

        ax.legend()
        ax.grid(True)

        st.pyplot(fig)

    # --------------------------------------------
    # SPECTROGRAM
    # --------------------------------------------
    with dsp_spec:

        D = librosa.stft(
            filtered_signal,
            n_fft=1024,
            hop_length=256
        )

        D_db = librosa.amplitude_to_db(
            np.abs(D),
            ref=np.max
        )

        fig, ax = plt.subplots(
            figsize=(12, 5)
        )

        image = librosa.display.specshow(
            D_db,
            sr=sr,
            hop_length=256,
            x_axis="time",
            y_axis="hz",
            ax=ax
        )

        ax.set_ylim(
            0,
            min(
                10000,
                sr / 2
            )
        )

        ax.set_title(
            "STFT Spectrogram"
        )

        fig.colorbar(
            image,
            ax=ax
        )

        st.pyplot(fig)

    # --------------------------------------------
    # MFCC
    # --------------------------------------------
    with dsp_mfcc:

        mfcc = librosa.feature.mfcc(
            y=filtered_signal,
            sr=sr,
            n_mfcc=13,
            n_fft=1024,
            hop_length=256
        )

        fig, ax = plt.subplots(
            figsize=(12, 5)
        )

        image = librosa.display.specshow(
            mfcc,
            x_axis="time",
            ax=ax
        )

        ax.set_title(
            "MFCC"
        )

        fig.colorbar(
            image,
            ax=ax
        )

        st.pyplot(fig)

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================
    st.markdown("---")

    st.header(
        "🌲 Random Forest Feature Importance"
    )

    importance = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_
    }).sort_values(
        "Importance",
        ascending=True
    )

    fig, ax = plt.subplots(
        figsize=(10, 9)
    )

    ax.barh(
        importance["Feature"],
        importance["Importance"]
    )

    ax.set_xlabel(
        "Importance"
    )

    ax.set_title(
        "DSP Features Used by Random Forest"
    )

    ax.grid(
        True,
        axis="x"
    )

    plt.tight_layout()

    st.pyplot(fig)

    st.subheader(
        "🏆 Top 5 DSP Features"
    )

    st.dataframe(
        importance
        .sort_values(
            "Importance",
            ascending=False
        )
        .head(5)
        .round(5),
        use_container_width=True
    )

    # ========================================================
    # FEATURE TABLE
    # ========================================================
    st.markdown("---")

    st.header(
        "🔬 DSP Feature Values"
    )

    feature_table = pd.DataFrame({
        "Feature": FEATURES,
        "Value": [
            features[name]
            for name in FEATURES
        ]
    })

    st.dataframe(
        feature_table.round(5),
        use_container_width=True
    )

import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import librosa
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance

st.set_page_config(page_title='Explainable Audio DSP AI', page_icon='🎙️', layout='wide')
st.title('🎙️ Explainable Audio DSP + Machine Learning')
st.caption('Real Cats/Dogs Dataset → DSP Features → Random Forest → Global Importance → Distribution → Local Explanation')

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / 'cats_dogs'
EXTS = {'.wav','.mp3','.flac','.m4a','.ogg','.aiff','.aif'}
FEATURE_KEYS = ['RMS','Peak','ZCR','Crest_Factor','Spectral_Centroid','Spectral_Bandwidth','Spectral_Rolloff','Spectral_Flatness','Dominant_Frequency'] + [f'MFCC_{i+1}' for i in range(13)]

def bandpass_filter(x, sr, lowcut=300, highcut=4000, order=5):
    nyq = 0.5 * sr
    low, high = max(lowcut/nyq,1e-5), min(highcut/nyq,0.99)
    if low >= high: return x
    return sosfilt(butter(order,[low,high],btype='bandpass',output='sos'), x)

def extract_dsp_features(x, sr):
    f = {}
    rms = float(np.sqrt(np.mean(x**2))); peak = float(np.max(np.abs(x)))
    f['RMS']=rms; f['Peak']=peak
    f['ZCR']=float(np.mean(librosa.feature.zero_crossing_rate(y=x)))
    f['Crest_Factor']=float(peak/(rms+1e-12))
    f['Spectral_Centroid']=float(np.mean(librosa.feature.spectral_centroid(y=x,sr=sr)))
    f['Spectral_Bandwidth']=float(np.mean(librosa.feature.spectral_bandwidth(y=x,sr=sr)))
    f['Spectral_Rolloff']=float(np.mean(librosa.feature.spectral_rolloff(y=x,sr=sr)))
    f['Spectral_Flatness']=float(np.mean(librosa.feature.spectral_flatness(y=x)))
    fft=np.fft.rfft(x); freqs=np.fft.rfftfreq(len(x),1/sr)
    f['Dominant_Frequency']=float(freqs[np.argmax(np.abs(fft))]) if len(freqs) else 0.0
    mfcc=librosa.feature.mfcc(y=x,sr=sr,n_mfcc=13)
    for i in range(13): f[f'MFCC_{i+1}']=float(np.mean(mfcc[i]))
    return f

def audio_files(folder):
    return sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in EXTS)

def locate_dataset():
    for cn,dn in [('cats','dogs'),('cat','dog')]:
        c=DATASET_DIR/cn; d=DATASET_DIR/dn
        if c.exists() and d.exists():
            cf,df=audio_files(c),audio_files(d)
            if cf and df: return cf,df
    return [],[]

@st.cache_data(show_spinner=False)
def build_dataset(dataset_path, lowcut, highcut):
    base=Path(dataset_path); rows=[]; errors=[]
    dirs=[('cat',base/'cats'),('dog',base/'dogs')]
    if not dirs[0][1].exists() or not dirs[1][1].exists():
        dirs=[('cat',base/'cat'),('dog',base/'dog')]
    for label,folder in dirs:
        for p in audio_files(folder):
            try:
                x,sr=librosa.load(str(p),sr=22050,mono=True)
                if len(x)<1000: continue
                x=bandpass_filter(x,sr,lowcut,highcut)
                row=extract_dsp_features(x,sr); row['Label']=label; row['File']=str(p.relative_to(base)); rows.append(row)
            except Exception as e: errors.append({'File':str(p),'Error':str(e)})
    return pd.DataFrame(rows),pd.DataFrame(errors)

@st.cache_resource(show_spinner=False)
def train_model(df_json):
    df=pd.read_json(df_json); X=df[FEATURE_KEYS].astype(float); y=df.Label
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.20,random_state=42,stratify=y)
    scaler=StandardScaler(); Xtr_s=scaler.fit_transform(Xtr); Xte_s=scaler.transform(Xte)
    model=RandomForestClassifier(n_estimators=400,random_state=42,class_weight='balanced',n_jobs=-1)
    model.fit(Xtr_s,ytr); pred=model.predict(Xte_s)
    report=classification_report(yte,pred,labels=['cat','dog'],output_dict=True,zero_division=0)
    perm=permutation_importance(model,Xte_s,yte,n_repeats=10,random_state=42,scoring='f1_macro',n_jobs=-1)
    return dict(model=model,scaler=scaler,accuracy=accuracy_score(yte,pred),cm=confusion_matrix(yte,pred,labels=['cat','dog']),report=report,
                rf=pd.DataFrame({'Feature':FEATURE_KEYS,'Importance':model.feature_importances_}).sort_values('Importance',ascending=False),
                perm=pd.DataFrame({'Feature':FEATURE_KEYS,'Importance':perm.importances_mean,'Std':perm.importances_std}).sort_values('Importance',ascending=False))

def local_impact(model,scaler,values,df,target):
    x=np.asarray(values,dtype=float).reshape(1,-1); xs=scaler.transform(x); classes=list(model.classes_); ti=classes.index(target)
    base=model.predict_proba(xs)[0,ti]; med=df[FEATURE_KEYS].median().values; rows=[]
    for i,name in enumerate(FEATURE_KEYS):
        xp=x.copy(); xp[0,i]=med[i]; p=model.predict_proba(scaler.transform(xp))[0,ti]
        rows.append({'Feature':name,'Original_Value':x[0,i],'Reference_Median':med[i],'Impact':base-p})
    out=pd.DataFrame(rows); out['Abs_Impact']=out.Impact.abs(); return out.sort_values('Abs_Impact',ascending=False).reset_index(drop=True)

def barplot(d,col,title,n=15):
    q=d.head(n).sort_values(col); fig,ax=plt.subplots(figsize=(9,6)); ax.barh(q.Feature,q[col]); ax.set_title(title); ax.set_xlabel(col); ax.grid(True,axis='x',alpha=.25); plt.tight_layout(); return fig

st.sidebar.header('⚙️ DSP Settings')
low=st.sidebar.number_input('Low Cut (Hz)',20,10000,300,50)
high=st.sidebar.number_input('High Cut (Hz)',100,10000,4000,100)
if low>=high: st.error('Low Cut 必須小於 High Cut。'); st.stop()

cats,dogs=locate_dataset()
if not cats or not dogs:
    st.error('找不到 cats_dogs/cats + cats_dogs/dogs Dataset。'); st.stop()
st.sidebar.success(f'Cat: {len(cats)} files\nDog: {len(dogs)} files')

with st.spinner('正在從真實 Cat/Dog 音檔建立 DSP Features...'):
    df,errors=build_dataset(str(DATASET_DIR),low,high)
if df.empty: st.error('Feature Dataset 是空的。'); st.stop()
with st.spinner('正在用真實 Dataset 訓練 Random Forest...'):
    pack=train_model(df.to_json())
model,scaler=pack['model'],pack['scaler']

st.subheader('① Real Dataset → DSP Features')
a,b,c=st.columns(3); a.metric('Total Audio',len(df)); b.metric('Cat',int((df.Label=='cat').sum())); c.metric('Dog',int((df.Label=='dog').sum()))
st.dataframe(df[['File','Label']].head(20),use_container_width=True)

st.subheader('② Model Evaluation')
a,b,c,d=st.columns(4); a.metric('Accuracy',f"{pack['accuracy']*100:.2f}%"); b.metric('Cat F1',f"{pack['report']['cat']['f1-score']*100:.2f}%"); c.metric('Dog F1',f"{pack['report']['dog']['f1-score']*100:.2f}%"); d.metric('Macro F1',f"{pack['report']['macro avg']['f1-score']*100:.2f}%")
cm=pack['cm']; fig,ax=plt.subplots(figsize=(5,4)); im=ax.imshow(cm); ax.set_xticks([0,1]); ax.set_yticks([0,1]); ax.set_xticklabels(['Cat','Dog']); ax.set_yticklabels(['Cat','Dog']); ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); ax.set_title('Confusion Matrix');
for i in range(2):
    for j in range(2): ax.text(j,i,cm[i,j],ha='center',va='center')
plt.colorbar(im,ax=ax); plt.tight_layout(); st.pyplot(fig)

st.subheader('③ Global Feature Importance')
t1,t2=st.tabs(['🌲 RF Importance','🔀 Permutation Importance'])
with t1:
    st.pyplot(barplot(pack['rf'],'Importance','Random Forest Global Feature Importance')); st.dataframe(pack['rf'],use_container_width=True)
with t2:
    st.pyplot(barplot(pack['perm'],'Importance','Permutation Importance on Test Set')); st.dataframe(pack['perm'],use_container_width=True)

st.subheader('④ Cat vs Dog Feature Distribution')
feature=st.selectbox('選擇 DSP Feature',FEATURE_KEYS,index=FEATURE_KEYS.index('Spectral_Centroid'))
catv=df.loc[df.Label=='cat',feature]; dogv=df.loc[df.Label=='dog',feature]
fig,ax=plt.subplots(figsize=(9,5)); ax.hist(catv,bins=30,alpha=.55,label='Cat'); ax.hist(dogv,bins=30,alpha=.55,label='Dog'); ax.axvline(catv.mean(),linestyle='--',label='Cat Mean'); ax.axvline(dogv.mean(),linestyle='--',label='Dog Mean'); ax.set_xlabel(feature); ax.set_ylabel('Samples'); ax.set_title(f'Cat vs Dog Distribution: {feature}'); ax.legend(); ax.grid(True,alpha=.25); plt.tight_layout(); st.pyplot(fig)
a,b,c=st.columns(3); a.metric('Cat Mean',f'{catv.mean():.4f}'); b.metric('Dog Mean',f'{dogv.mean():.4f}'); c.metric('Absolute Mean Gap',f'{abs(catv.mean()-dogv.mean()):.4f}')

st.subheader('⑤ Local Feature Impact / Misclassification Analysis')
files=cats+dogs; rel=[str(p.relative_to(DATASET_DIR)) for p in files]; preferred=[x for x in rel if 'dog_barking_75.wav' in x.lower()]; idx=rel.index(preferred[0]) if preferred else 0
uploaded = st.file_uploader(
    "📁 上傳音訊 / Browse files",
    type=["wav", "mp3", "flac", "m4a", "ogg", "aiff", "aif"]
)

if uploaded is not None:
    import tempfile
    suffix = Path(uploaded.name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        path = Path(tmp.name)
    st.audio(uploaded)
else:
    st.info("請先上傳音訊檔案，再進行 DSP 與 AI 分析。")
    st.stop()

true = None  # 上傳音檔沒有已知標籤，不從檔名推定
try:
    x,sr=librosa.load(str(path),sr=22050,mono=True); x=bandpass_filter(x,sr,low,high); feats=extract_dsp_features(x,sr); vec=np.array([feats[k] for k in FEATURE_KEYS]); xs=scaler.transform(vec.reshape(1,-1)); pred=model.predict(xs)[0]; probs=model.predict_proba(xs)[0]; pm={c:probs[i] for i,c in enumerate(model.classes_)}
    a,b=st.columns(2); a.metric('Prediction',pred.upper()); b.metric('Confidence',f"{pm[pred]*100:.1f}%")
    if pred==true: st.success('✅ Prediction 正確')
    else: st.error('❌ Misclassification：這筆資料適合做 Local Explanation')
    with st.expander('查看 22 個 DSP Features'): st.dataframe(pd.DataFrame({'Feature':FEATURE_KEYS,'Value':[feats[k] for k in FEATURE_KEYS]}),use_container_width=True)
    local=local_impact(model,scaler,vec,df,pred)
    st.markdown(f'**Local explanation:** 正值表示該 Feature 原本的數值支持目前的 `{pred.upper()}` prediction；負值表示抵抗該 prediction。')
    st.pyplot(barplot(local,'Impact',f'Why did the model predict {pred.upper()}?',10))
    disp=local.head(15).copy(); disp['Impact (%)']=(disp.Impact*100).round(2); disp['Original_Value']=disp.Original_Value.round(4); disp['Reference_Median']=disp.Reference_Median.round(4); st.dataframe(disp[['Feature','Original_Value','Reference_Median','Impact (%)']],use_container_width=True)
except Exception as e: st.error(f'無法分析音檔：{e}')

st.subheader('⑥ Engineering Interpretation')
st.markdown('''
```text
Real Cats/Dogs Audio
        ↓
DSP Feature Extraction (22 features)
        ↓
Random Forest
   ┌────┼───────────────┐
   ↓    ↓               ↓
Global  Cat-vs-Dog      Local
Importance Distribution Impact
   │    │               │
   └────┴───────┬───────┘
                 ↓
        Misclassification
                 ↓
        Root-Cause Analysis
```

**注意：Feature Importance / Local Impact 是模型解釋，不是物理因果證明。**
''')

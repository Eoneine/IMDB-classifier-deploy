import streamlit as st
import joblib
import numpy as np
 
st.set_page_config(page_title='Classifier Demo', page_icon=':mag:', layout='centered')
 
@st.cache_resource
def load_artefak():
    model = joblib.load('lr_best.pkl')
    vec   = joblib.load('vec_terpilih.pkl')
    le    = joblib.load('label_encoder.pkl')
    with open('treshold.txt') as f:
        thr = float(f.read().strip())
    return model, vec, le, thr
 
model, vec, le, threshold = load_artefak()
 
st.title(':mag: Text Classification App')
st.caption(f'Prediction threshold: {threshold:.3f}  |  Classes: {list(le.classes_)}')
st.divider()
 
teks_input = st.text_area(
    'Enter the text you want to classify:',
    height=150,
    placeholder='Type or paste your text here...'
)
 
if st.button('Classify', type='primary', use_container_width=True):
    if not teks_input.strip():
        st.warning('Please enter some text first.')
    else:
        try:
            X = vec.transform([teks_input])
            if len(le.classes_) == 2 and hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[0, 1]
                pred  = int(proba >= threshold)
                kelas_pred = le.classes_[pred]
                st.success(f'Prediction: **{kelas_pred}**')
                st.metric('Positive class probability', f'{proba:.4f}')
                st.progress(float(proba))
            else:
                pred = model.predict(X)[0]
                proba = model.predict_proba(X)[0] if hasattr(model, 'predict_proba') else None
                kelas_pred = le.classes_[pred]
                st.success(f'Prediction: **{kelas_pred}**')
                if proba is not None:
                    import pandas as pd
                    df_proba = pd.DataFrame({'Class': le.classes_, 'Probability': proba.round(4)})
                    st.dataframe(df_proba, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f'Error: {e}')
 
st.divider()
st.caption('Created for PPKD Jakarta Selatan - Data Analyst Program')

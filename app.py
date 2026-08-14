import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# ১. পেজ কনফিগারেশন
st.set_page_config(page_title="Smart Crop Recommendation", layout="wide")

# ২. ডেটা ও মডেল লোড করা (Cache ব্যবহার করা হচ্ছে যাতে অ্যাপ ফাস্ট হয়)
@st.cache_resource
def load_data_and_models():
    model = joblib.load('best_rf_model.pkl')
    scaler = joblib.load('scaler.pkl')
    df = pd.read_csv('Crop_recommendation.csv')
    return model, scaler, df

model, scaler, df = load_data_and_models()

# ৩. মূল ড্যাশবোর্ড UI
st.title("🌱 Smart Crop Recommendation Dashboard")
st.markdown("Enter the soil and climate details on the left, and view the AI's recommendation along with the analytics dashboard on the right.")

# স্ক্রিনকে দুই ভাগে ভাগ করা
col1, col2 = st.columns([1, 2])

# বাম দিকের অংশ: ইউজারের ইনপুট
with col1:
    st.header("📊 Input Parameters")
    N = st.number_input("Nitrogen (N)", value=90.0)
    P = st.number_input("Phosphorus (P)", value=42.0)
    K = st.number_input("Potassium (K)", value=43.0)
    temp = st.number_input("Temperature (°C)", value=20.8)
    humidity = st.number_input("Humidity (%)", value=82.0)
    ph = st.number_input("pH Value", value=6.5)
    rainfall = st.number_input("Rainfall (mm)", value=202.9)
    
    predict_btn = st.button("Predict Best Crop", use_container_width=True)

# ডান দিকের অংশ: রেজাল্ট এবং গ্রাফ
with col2:
    st.header("🎯 Model Recommendation")
    
    # প্রেডিক্ট বাটনে ক্লিক করলে কী হবে
    if predict_btn:
        input_data = np.array([[N, P, K, temp, humidity, ph, rainfall]])
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)[0]
        st.success(f"🌟 The most suitable crop for this condition is: **{prediction.upper()}** 🌟")
    
    st.header("📈 Analytics Dashboard")
    # গ্রাফগুলোকে আলাদা ট্যাবে দেখানোর জন্য
    tab1, tab2 = st.tabs(["Feature Correlation", "Feature Importance"])
    
    with tab1:
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        sns.heatmap(df.drop('label', axis=1).corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax1)
        st.pyplot(fig1)
        
    with tab2:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        X_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        feature_importances = pd.Series(model.feature_importances_, index=X_columns).sort_values(ascending=False)
        sns.barplot(x=feature_importances, y=feature_importances.index, palette='viridis', ax=ax2)
        st.pyplot(fig2)

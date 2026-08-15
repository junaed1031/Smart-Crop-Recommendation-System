import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ১. পেজ কনফিগারেশন
st.set_page_config(page_title="Smart Crop Recommendation", layout="wide")

# ২. ডেটা ও মডেল লোড করা
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
    
    if predict_btn:
        input_data = np.array([[N, P, K, temp, humidity, ph, rainfall]])
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)[0]
        st.success(f"🌟 The most suitable crop for this condition is: **{prediction.upper()}** 🌟")
    
    st.header("📈 Analytics Dashboard")
    # চারটি আলাদা ট্যাব তৈরি করা হলো
    tab1, tab2, tab3, tab4 = st.tabs(["Feature Correlation", "Feature Importance", "Model Performance", "Data Overview"])
    
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
        
    with tab3:
        st.subheader("Model Accuracy & Evaluation")
        
        # ব্যাকগ্রাউন্ডে দ্রুত একুরেসি ক্যালকুলেট করার লজিক
        X = df.drop('label', axis=1)
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, y_pred)
        st.metric(label="Random Forest Accuracy", value=f"{acc * 100:.2f}%")
        
        st.markdown("**Confusion Matrix:**")
        fig3, ax3 = plt.subplots(figsize=(12, 8))
        cm = confusion_matrix(y_test, y_pred)
        
        # এখানে ফসলের নামগুলো লেবেল হিসেবে দেওয়া হলো
        class_labels = model.classes_
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_labels, yticklabels=class_labels, ax=ax3)
        
        plt.xlabel("Predicted Crop")
        plt.ylabel("Actual Crop")
        plt.xticks(rotation=45, ha='right') # নামগুলো বাঁকা করে সুন্দরভাবে দেখানোর জন্য
        st.pyplot(fig3)
        
    with tab4:
        st.subheader("Dataset Quick Look")
        st.markdown("Here are the first 10 rows of the dataset used to train the model:")
        # ডেটাসেট সুন্দর একটি টেবিলে দেখাবে
        st.dataframe(df.head(10), use_container_width=True)

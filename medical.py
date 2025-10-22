import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# load the model
model = pickle.load(open('model.pkl', 'rb'))

# load dataset
data = pd.read_csv('insurance.csv')

## streamlit PAGE CONFIGURATION
st.set_page_config(page_title="Medical Diagnosis App", 
                   layout="wide",
                   page_icon="🏥",
                   initial_sidebar_state="auto")

# Sidebar : Theme + Navigation
st.sidebar.title("🏠 Menu")
page = st.sidebar.radio("Go to", ["Home", "Predict Insurance Cost"])
st.sidebar.markdown("---")

# ----------------------
# Prediction Page
# ----------------------
if page == "Predict Insurance Cost":
    st.title("💰 Medical Insurance Cost Prediction")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 18, 100, 25)
        bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
        children = st.number_input("Number of Children", 0, 10, 0)
    with col2:
        sex = st.selectbox("Gender", ("male", "female"))
        smoker = st.selectbox("Smoker", ("yes", "no"))
        region = st.selectbox("Region", ("northeast", "northwest", "southeast", "southwest"))

    if st.button("🔮 Predict"):
        sex_enc = 1 if sex=="male" else 0
        smoker_enc = 1 if smoker=="yes" else 0
        region_map = {"northeast":0, "northwest":1, "southeast":2, "southwest":3}
        region_enc = region_map[region]

        input_data = np.array([[age, sex_enc, bmi, children, smoker_enc, region_enc]])
        prediction = model.predict(input_data)[0]
        st.success(f"💵 Predicted Insurance Cost: **${prediction:,.2f}**")

# ----------------------
# Data Visualization Page
# ----------------------
elif page == 'Home':
    st.title("📊 Medical Insurance Data Visualization")
    st.write("View various statistics and distributions in the insurance dataset.")

    tabs = st.tabs(['Statistics', 'Age Distribution', 'BMI Distribution', 'Charge Distribution', 'Gender Analysis', 'Children Analysis', 'Smoker Analysis', 'Region Analysis'])
    with tabs[0]:
        st.subheader("Dataset Overview")
        st.dataframe(data)
        st.subheader("Statistical Summary")
        st.write(data.describe())
        st.write("### Categorical Value Counts")
        st.write(data['sex'].value_counts())
        st.write(data['smoker'].value_counts())
        st.write(data['region'].value_counts())
    
    with tabs[1]:
        st.subheader("Age Distribution")
        plt.figure(figsize=(5,2))
        sns.histplot(data['age'], bins = 30, kde = True)
        plt.title("Age Distribution")
        st.pyplot(plt)

    with tabs[2]:
        st.subheader("BMI Distribution")
        plt.figure(figsize=(5, 2))
        sns.histplot(data['bmi'], bins = 30, kde = True)
        plt.title("BMI Distribution")
        st.pyplot(plt)
    with tabs[3]:
        st.subheader("Charge Distribution")
        plt.figure(figsize=(5, 2))
        sns.histplot(data['charges'], bins = 30, kde = True)
        plt.title("Charge Distribution")
        st.pyplot(plt)

    with tabs[4]:
        st.subheader("Gender Analysis")
        plt.figure(figsize=(10, 12))
        plt.subplot(4, 1, 1)
        sns.countplot(x = 'sex', data = data, hue = 'smoker')
        plt.title("Count of Smokers by Gender")
        
        plt.subplot(4, 1, 2)
        sns.boxplot(x = 'sex', y = 'charges', data = data)
        plt.title("Charges by Gender")

        plt.subplot(4, 1, 3)
        sns.boxplot(x = 'sex', y = 'bmi', data = data)
        plt.title("BMI by Gender")
        plt.tight_layout()
        
        st.pyplot(plt)

    with tabs[5]:
        st.subheader("Children Analysis")
        plt.figure(figsize=(10, 12))
        plt.subplot(3, 1, 1)
        sns.countplot(x = 'children', data = data, hue = 'smoker')
        plt.title("Count of Smokers by Number of Children")
        
        plt.subplot(3, 1, 2)
        sns.boxplot(x = 'children', y = 'charges', data = data)
        plt.title("Charges by Number of Children")

        plt.subplot(3, 1, 3)
        sns.boxplot(x = 'children', y = 'bmi', data = data)
        plt.title("BMI by Number of Children")
        plt.tight_layout()
        
        st.pyplot(plt)

    with tabs[6]:
        st.subheader("Smoker Analysis")
        plt.figure(figsize=(10, 12))
        plt.subplot(3, 1, 1)
        sns.countplot(x = 'smoker', data = data, hue = 'sex')
        plt.title("Count of Smokers by Gender") 
        plt.subplot(3, 1, 2)
        sns.boxplot(x = 'smoker', y = 'charges', data = data)           
        plt.title("Charges by Smoking Status")
        plt.subplot(3, 1, 3)
        sns.boxplot(x = 'smoker', y = 'bmi', data = data)
        plt.title("BMI by Smoking Status")
        plt.tight_layout()
        st.pyplot(plt)
    with tabs[7]:
        st.subheader("Region Analysis")
        plt.figure(figsize=(10, 12))
        plt.subplot(3, 1, 1)
        sns.countplot(x = 'region', data = data, hue = 'smoker')
        plt.title("Count of Smokers by Region") 
        plt.subplot(3, 1, 2)
        sns.boxplot(x = 'region', y = 'charges', data = data)           
        plt.title("Charges by Region")
        plt.subplot(3, 1, 3)
        sns.boxplot(x = 'region', y = 'bmi', data = data)
        plt.title("BMI by Region")
        plt.tight_layout()
        st.pyplot(plt)
    
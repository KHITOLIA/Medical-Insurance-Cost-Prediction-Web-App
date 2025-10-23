import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# ------------------------------------------------
# Load model and dataset
# ------------------------------------------------
model = pickle.load(open('model.pkl', 'rb'))
data = pd.read_csv('insurance.csv')

# ------------------------------------------------
# Initialize users.csv
# ------------------------------------------------
if not os.path.exists('users.csv'):
    user_df = pd.DataFrame(columns=['username', 'email_id', 'password'])
    user_df.to_csv('users.csv', index=False)
else:
    user_df = pd.read_csv('users.csv')

# ------------------------------------------------
# Streamlit page setup
# ------------------------------------------------
st.set_page_config(
    page_title="Medical Insurance Predictor",
    page_icon="🏥",
    layout="wide"
)

# ------------------------------------------------
# Custom Background and Styling
# ------------------------------------------------
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://media.istockphoto.com/id/1136667774/vector/health-insurance-vector-illustration-medical-protection-medical-insurance-concepts-flat.jpg?s=1024x1024&w=is&k=20&c=HtgV0pY1OT6wGyYxJPzlciFEapuORFKDpIdnQe3wlO4=");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
[data-testid="stSidebar"] {
    background-color: black;
}
.stButton>button {
    background-color: #00adb5;
    color: white;
    border-radius: 10px;
    height: 2.5em;
    width: 100%;
    font-weight: bold;
}
.stTextInput>div>div>input {
    border-radius: 10px;
}
h1, h2, h3 {
    color: white;
}
div.block-container {
    background-color: rgba(0,0,0,0.6);
    padding: 2em;
    border-radius: 15px;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# ------------------------------------------------
# Initialize session state
# ------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "users_df" not in st.session_state:
    st.session_state.users_df = user_df

# ------------------------------------------------
# LOGIN / SIGNUP PAGE
# ------------------------------------------------
def login_page():
    st.markdown("<h1 style='text-align:center;'>🏥 Medical Insurance Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center;'>Login or Create an Account</h3>", unsafe_allow_html=True)

    # Center layout
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        choice = st.radio("Select Option", ["Login", "Sign Up"], horizontal=True)

        if choice == "Sign Up":
            st.subheader("Create Account")
            new_user = st.text_input("Enter Email ID")
            new_user_name = st.text_input("Full Name")
            new_pass = st.text_input("Create Password", type="password")

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Sign Up"):
                if new_user in st.session_state.users_df['email_id'].values:
                    st.error("Email ID already exists! Please choose another.")
                else:
                    new_row = pd.DataFrame({"username": [new_user_name],
                                            "email_id": [new_user],
                                            "password": [new_pass]})
                    st.session_state.users_df = pd.concat([st.session_state.users_df, new_row], ignore_index=True)
                    st.session_state.users_df.to_csv("users.csv", index=False)
                    st.success("✅ Account created successfully! You can now log in.")

        elif choice == "Login":
            st.subheader("Login to Continue")
            email_id = st.text_input("Email ID")
            password = st.text_input("Password", type="password")

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Login"):
                df = st.session_state.users_df
                user = df[(df['email_id'] == email_id) & (df['password'] == password)]
                if not user.empty:
                    st.session_state.logged_in = True
                    st.session_state.username = user.iloc[0]['username']
                    st.success(f"🎉 Welcome {st.session_state.username}!")
                    st.rerun()
                else:
                    st.error("❌ Invalid email or password!")

# ------------------------------------------------
# MAIN DASHBOARD (after login)
# ------------------------------------------------
def dashboard():
    st.sidebar.title(f"👋 Hello, {st.session_state.username}")
    st.sidebar.markdown("---")
    page = st.sidebar.radio("📂 Navigation", ["🏠 Home", "💰 Predict Insurance Cost"])

    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.experimental_rerun()

    # ---------------- HOME PAGE ----------------
    if page == "🏠 Home":
        st.title("📊 Medical Insurance Data Visualization Dashboard")
        st.write("Explore insights and distributions from the dataset below 👇")

        tabs = st.tabs(['Statistics', 'Age Distribution', 'BMI Distribution', 'Charge Distribution', 'Gender Analysis', 'Children Analysis', 
                        'Smoker Analysis', 'Region Analysis', 'Average Charges by Region', 'Smokers Pie Chart', 'Pair Plot', 'Correlation Heatmap',
                        'Report Summary'])
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

        with tabs[8]:
            st.subheader("Average Charges by Region")
            region_avg = data.groupby('region')['charges'].mean().reset_index()
            plt.figure(figsize=(6,4))
            sns.barplot(x= data['region'].unique(), y='charges', data=region_avg, palette="Set2", )
            plt.title("Average Charges by Region")
            st.pyplot(plt)

        with tabs[9]:
            st.subheader("Smokers Pie Chart")
            smoker_counts = data['smoker'].value_counts()
            plt.figure(figsize=(6,6))
            plt.pie(smoker_counts, labels=smoker_counts.index, autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff'])
            plt.title("Proportion of Smokers vs Non-Smokers")
            st.pyplot(plt)
        
        with tabs[10]:
            st.subheader('Pair Plot of Features')
            sns.pairplot(data, hue = 'smoker', diag_kind='kde')
            plt.title("Pair Plot of Medical Insurance Features")
            st.pyplot(plt)
        
        with tabs[11]:
            st.subheader('Correlation Heatmap')
            data['region'] = le.fit_transform(data['region'])
            data['sex'] = le.fit_transform(data['sex'])
            data['smoker'] = le.fit_transform(data['smoker'])
            corr = data.corr()
            plt.figure(figsize=(10, 6))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title("Correlation Heatmap of Features")    
            st.pyplot(plt)
        
        with tabs[12]:
            st.subheader('Report Summary')
            st.write(""" 

#### **Medical Insurance Data Analysis Report**

#### **1. Dataset Overview**

* Total data points: `1338` rows (patients).
* Features: `age`, `sex`, `bmi`, `children`, `smoker`, `region`, `charges`.
* No missing values found — dataset is clean and ready for analysis.
* Feature types:

  * Numerical: `age`, `bmi`, `children`, `charges`.
  * Categorical: `sex`, `smoker`, `region`.

---

#### **2. Age Analysis**

* Age range: 18–64 years.
* Most patients are between 18–40 years.
* Distribution is right-skewed — fewer older adults.

---

#### **3. Gender Analysis**

* Almost equal proportion of males and females.
* Among smokers, males slightly outnumber females.
* Charges and BMI show minor differences between genders, but smoking heavily impacts charges.

---

#### **4. BMI (Body Mass Index) Analysis**

* BMI range: ~15–53.
* Majority of patients have BMI between 20–35.
* Higher BMI often corresponds to slightly higher insurance charges, but not as strongly as smoking status.

---

#### **5. Children Feature**

* Number of children ranges from 0 to 5.
* Most patients have 0 or 1 child.
* Families with more children show slightly higher average charges.

---

#### **6. Smoker Analysis**

* Smokers: ~20% of the dataset.
* Smoking status is the **strongest factor** influencing insurance charges.
* Smokers consistently have significantly higher charges than non-smokers.
* Smoking is more common among males in this dataset.

---

#### **7. Region Analysis**

* Dataset divided into 4 regions: `northeast`, `northwest`, `southeast`, `southwest`.
* Distribution is roughly even across regions.
* Average charges slightly higher in the `southeast` and `northeast`.

---

#### **8. Charges Analysis**

* Charges range: ~$1121 to ~$63770.
* Right-skewed distribution (few extremely high charges).
* Strong correlation observed:

  * **Smoker → Higher charges**
  * **Age → Moderate positive correlation with charges**
  * **BMI → Positive correlation with charges**
* Children and gender have weaker correlations.

---

#### **9. Correlation Highlights**

* `charges` most correlated with:

  * `smoker` (strong positive correlation)
  * `age` (moderate positive correlation)
  * `bmi` (moderate positive correlation)
* `children` and `region` have weak correlation with `charges`.

---

#### **10. Visual Insights**

* **Histograms**: Age, BMI, and Charges distributions help identify population clusters.
* **Boxplots**: Charges vary significantly for smokers; BMI also shows variation across genders.
* **Bar plots**: Average charges highest in `southeast` and `northeast` regions.
* **Pie chart**: Non-smokers dominate the dataset (~80%).

---

#### **11. Key Takeaways**

* Smoking is the **primary driver of insurance costs**.
* Older age and higher BMI also increase charges but to a lesser extent.
* Gender has minimal impact on charges, but males smoke slightly more than females.
* Region and number of children have minor effect on costs.
* Target audience for high insurance charges: **older, overweight, smoker individuals**.

---


""")
              
        # ---------------- PREDICTION PAGE ----------------
    elif page == "💰 Predict Insurance Cost":
        st.title("💰 Predict Your Medical Insurance Cost")

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
            sex_enc = 1 if sex == "male" else 0
            smoker_enc = 1 if smoker == "yes" else 0
            region_map = {"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3}
            region_enc = region_map[region]

            input_data = np.array([[age, sex_enc, bmi, children, smoker_enc, region_enc]])
            prediction = model.predict(input_data)[0]
            st.success(f"💵 Estimated Insurance Cost: **${prediction:,.2f}**")
# ------------------------------------------------
# MAIN APP CONTROLLER
# ------------------------------------------------
if not st.session_state.logged_in:
    login_page()
else:
    dashboard()
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('diabetes.csv')

# Set up the app title and sidebar
st.title('Diabetes Checker')
st.video("video.mp4")
st.sidebar.header('Patient Data')
st.subheader('Training Data')
st.write(df.describe())

# Prepare the data for training
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Function to get user input data
def user_report():
    Pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    Glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    BloodPressure = st.sidebar.slider('Blood Pressure', 0, 122, 70)
    SkinThickness = st.sidebar.slider('Skin Thickness', 0, 100, 20)
    Insulin = st.sidebar.slider('Insulin', 0, 846, 79)
    BMI = st.sidebar.slider('BMI', 0, 67, 20)
    DiabetesPedigreeFunction = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.47)
    Age = st.sidebar.slider('Age', 21, 88, 33)

    user_report_data = {
        'Pregnancies': Pregnancies,
        'Glucose': Glucose,
        'BloodPressure': BloodPressure,
        'SkinThickness': SkinThickness,
        'Insulin': Insulin,
        'BMI': BMI,
        'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
        'Age': Age
    }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data

# Get user input data
user_data = user_report()

st.subheader('Patient Data')
st.write(user_data)

st.subheader('Upload your disease photo')
st.file_uploader('Upload a photo')

# Train the model
rf = RandomForestClassifier()
rf.fit(x_train, y_train)

# Predict and display results when the user clicks the submit button
if st.sidebar.button('Submit'):
    user_result = rf.predict(user_data)

    st.title('Visualised Patient Report')

    color = 'blue' if user_result[0] == 0 else 'red'

    st.header('Pregnancy count (Others vs Yours)')
    fig_preg = plt.figure()
    sns.scatterplot(x='Age', y='Pregnancies', data=df, hue='Outcome', palette='Greens')
    sns.scatterplot(x=user_data['Age'], y=user_data['Pregnancies'], s=150, color=color)
    plt.xticks(np.arange(10, 100, 5))
    plt.yticks(np.arange(0, 20, 2))
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_preg)

    st.header('Glucose Value (Others vs Yours)')
    fig_glucose = plt.figure()
    sns.scatterplot(x='Age', y='Glucose', data=df, hue='Outcome', palette='magma')
    sns.scatterplot(x=user_data['Age'], y=user_data['Glucose'], s=150, color=color)
    plt.xticks(np.arange(10, 100, 5))
    plt.yticks(np.arange(0, 220, 10))
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_glucose)

    st.header('Blood Pressure Value (Others vs Yours)')
    fig_bp = plt.figure()
    sns.scatterplot(x='Age', y='BloodPressure', data=df, hue='Outcome', palette='Reds')
    sns.scatterplot(x=user_data['Age'], y=user_data['BloodPressure'], s=150, color=color)
    plt.xticks(np.arange(10, 100, 5))
    plt.yticks(np.arange(0, 130, 10))
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_bp)

    st.header('Skin Thickness Value (Others vs Yours)')
    fig_st = plt.figure()
    sns.scatterplot(x='Age', y='SkinThickness', data=df, hue='Outcome', palette='Blues')
    sns.scatterplot(x=user_data['Age'], y=user_data['SkinThickness'], s=150, color=color)
    plt.xticks(np.arange(10, 100, 5))
    plt.yticks(np.arange(0, 110, 10))
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_st)

    st.header('Insulin Value (Others vs Yours)')
    fig_i = plt.figure()
    sns.scatterplot(x='Age', y='Insulin', data=df, hue='Outcome', palette='rocket')
    sns.scatterplot(x=user_data['Age'], y=user_data['Insulin'], s=150, color=color)
    plt.xticks(np.arange(10, 100, 5))
    plt.yticks(np.arange(0, 900, 50))
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_i)

    st.header('BMI Value (Others vs Yours)')
    fig_bmi = plt.figure()
    sns.scatterplot(x='Age', y='BMI', data=df, hue='Outcome', palette='rainbow')
    sns.scatterplot(x=user_data['Age'], y=user_data['BMI'], s=150, color=color)
    plt.xticks(np.arange(10, 100, 5))
    plt.yticks(np.arange(0, 70, 5))
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_bmi)

    st.header('DPF Value (Others vs Yours)')
    fig_dpf = plt.figure()
    sns.scatterplot(x='Age', y='DiabetesPedigreeFunction', data=df, hue='Outcome', palette='YlOrBr')
    sns.scatterplot(x=user_data['Age'], y=user_data['DiabetesPedigreeFunction'], s=150, color=color)
    plt.xticks(np.arange(10, 100, 5))
    plt.yticks(np.arange(0, 3, 0.2))
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_dpf)

    st.subheader('Final Report:')
    output = 'You are not Diabetic ❤' if user_result[0] == 0 else 'You are Diabetic 😥'
    st.title(output)

    st.subheader('Percentage:')
    st.write(str(accuracy_score(y_test, rf.predict(x_test)) * 100) + '%')

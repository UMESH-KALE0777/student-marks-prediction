import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt

st.set_page_config(page_title="Student Marks Prediction", page_icon="📘", layout="centered")

# Load model & metrics
model = pickle.load(open("models/linear_model.pkl", "rb"))
metrics = json.load(open("models/metrics.json", "r"))

df = pd.read_csv("data/student_marks.csv")

# Title
st.title("📘 Student Marks Prediction Dashboard")
st.write("A machine learning model that predicts student marks based on study habits.")

# Show accuracy
st.metric(label="📊 Model Accuracy (R² Score)", value=f"{metrics['accuracy']:.2f}")

st.divider()

# Input section
st.subheader("🧮 Enter Student Inputs")
hours = st.slider("⏳ Hours Studied", 1, 10, 4)
attendance = st.slider("📚 Attendance (%)", 50, 100, 75)
assignments = st.slider("📝 Assignments Submitted", 1, 10, 5)

st.divider()

# Prediction
if st.button("🔮 Predict Marks"):
    input_data = np.array([[hours, attendance, assignments]])
    prediction = model.predict(input_data)[0]
    st.success(f"🎯 **Predicted Marks: {round(prediction, 2)} / 100**")

    # Explain prediction
    st.subheader("📉 Contribution of Each Factor")
    fig, ax = plt.subplots()
    features = ["Hours", "Attendance", "Assignments"]
    values = [hours, attendance, assignments]
    ax.bar(features, values)
    plt.title("Input Feature Impact")
    st.pyplot(fig)

st.divider()

# Graph Section
st.subheader("📈 Data Visualization")

col1, col2 = st.columns(2)

with col1:
    st.write("### Hours vs Marks")
    fig, ax = plt.subplots()
    ax.scatter(df['Hours_Studied'], df['Marks'])
    ax.set_xlabel("Hours Studied")
    ax.set_ylabel("Marks")
    st.pyplot(fig)

with col2:
    st.write("### Attendance vs Marks")
    fig, ax = plt.subplots()
    ax.scatter(df['Attendance'], df['Marks'])
    ax.set_xlabel("Attendance (%)")
    ax.set_ylabel("Marks")
    st.pyplot(fig)

st.write("### Assignments vs Marks")
fig, ax = plt.subplots()
ax.scatter(df['Assignments_Submitted'], df['Marks'])
ax.set_xlabel("Assignments Submitted")
ax.set_ylabel("Marks")
st.pyplot(fig)

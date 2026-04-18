import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Habit Trend Dashboard",
    page_icon="🏃",
    layout="centered"
)

# -------------------------------
# Title
# -------------------------------
st.title("🏃 Habit Trend Dashboard")
st.write("AI-powered daily activity trend prediction")

# -------------------------------
# Model Selection
# -------------------------------
model_option = st.selectbox(
    "🔍 Select Model",
    ["RandomForest", "LogisticRegression", "SVM"]
)

model = pickle.load(open(f"models/{model_option}.pkl", "rb"))

# -------------------------------
# Inputs
# -------------------------------
st.subheader("📊 Enter Activity Details")

col1, col2 = st.columns(2)

with col1:
    steps = st.number_input("Total Steps", min_value=0, value=5000)
    very_active = st.number_input("Very Active Minutes", min_value=0, value=30)

with col2:
    calories = st.number_input("Calories Burned", min_value=0, value=2000)
    sedentary = st.number_input("Sedentary Minutes", min_value=0, value=600)

# Feature Engineering
activity_ratio = very_active / (sedentary + 1)

st.write(f"⚙️ Activity Ratio: {activity_ratio:.2f}")

# -------------------------------
# 📊 Chart 1: Input Overview
# -------------------------------
st.subheader("📊 Activity Overview")

labels = ["Steps", "Calories", "Active", "Sedentary"]
values = [steps, calories, very_active, sedentary]

plt.figure()
plt.bar(labels, values)
plt.title("Daily Activity Summary")
st.pyplot(plt)

# -------------------------------
# 📈 Chart 2: Activity Distribution
# -------------------------------
st.subheader("📈 Activity Distribution")

plt.figure()
plt.pie(
    [very_active, sedentary],
    labels=["Active", "Sedentary"],
    autopct="%1.1f%%"
)
plt.title("Active vs Sedentary Time")
st.pyplot(plt)

# -------------------------------
# Prediction
# -------------------------------
if st.button("🚀 Predict Trend"):

    features = np.array([[steps, calories, very_active, sedentary, activity_ratio]])
    prediction = model.predict(features)[0]

    st.subheader("📈 Prediction Result")

    if prediction == "Declining":
        st.error("📉 Declining Activity Trend")
        st.write("💡 Increase steps and reduce sedentary time")

    elif prediction == "Improving":
        st.success("📈 Improving Activity Trend")
        st.write("💪 Keep going! You're doing great")

    else:
        st.info("⚖️ Stable Activity Trend")
        st.write("👍 Try pushing slightly for improvement")

    # -------------------------------
    # 📊 Chart 3: Prediction Confidence
    # -------------------------------
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[0]

        st.subheader("🎯 Prediction Confidence")

        classes = ["Declining", "Improving", "Stable"]

        plt.figure()
        plt.bar(classes, probs)
        plt.title("Prediction Probabilities")
        st.pyplot(plt)

    # -------------------------------
    # 📉 Chart 4: Simulated Trend
    # -------------------------------
    st.subheader("📉 Weekly Trend Simulation")

    simulated_steps = [steps + i * 500 for i in range(5)]

    plt.figure()
    plt.plot(simulated_steps, marker='o')
    plt.title("Simulated Future Steps Trend")
    plt.xlabel("Days Ahead")
    plt.ylabel("Steps")
    st.pyplot(plt)
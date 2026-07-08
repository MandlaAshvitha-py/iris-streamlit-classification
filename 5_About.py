import streamlit as st

# --------------------------
# Page Configuration
# --------------------------
st.set_page_config(
    page_title="About",
    page_icon="ℹ️",
    layout="wide"
)

# --------------------------
# Title
# --------------------------
st.title("ℹ️ About This Project")

st.markdown("---")

# --------------------------
# Project Description
# --------------------------
st.header("📖 Project Description")

st.write("""
The Iris Flower Classification project is a Machine Learning application
developed using Streamlit.

It predicts the species of an Iris flower based on four input features.
""")

# --------------------------
# Objective
# --------------------------
st.header("🎯 Project Objective")

st.write("""
The main objective of this project is to demonstrate the complete
Machine Learning workflow:

• Data Collection

• Data Exploration

• Data Visualization

• Model Training

• Model Evaluation

• Prediction using a trained model
""")

# --------------------------
# Technologies
# --------------------------
st.header("🛠 Technologies Used")

st.write("""
- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
""")

# --------------------------
# Machine Learning Algorithm
# --------------------------
st.header("🤖 Machine Learning Algorithm")

st.write("""
This project uses the Random Forest Classifier algorithm for
classification.
""")

# --------------------------
# Dataset
# --------------------------
st.header("📊 Dataset")

st.write("""
Dataset Name: Iris Dataset

Number of Samples: 150

Number of Features: 4

Target Classes: 3
""")

# --------------------------
# Skills
# --------------------------
st.header("💡 Skills Demonstrated")

st.write("""
✔ Data Analysis

✔ Data Visualization

✔ Machine Learning

✔ Model Training

✔ Streamlit Development

✔ Python Programming
""")

st.markdown("---")

st.success("Thank you for exploring this project! 🌸")
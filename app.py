import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon="🌸",
    layout="wide"
)

# Sidebar
st.sidebar.title("🌸 Navigation")
st.sidebar.success("Select a page from the sidebar.")

# Main Title
st.title("🌸 Iris Flower Classification ")

st.image("images/iris.jpg", width=300)

st.markdown("---")

# Introduction
st.header("📖 Project Overview")

st.write("""
The Iris Flower Classification project uses Machine Learning to predict
the species of an Iris flower based on four flower measurements.

This application is built using:

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
""")

# Dataset Information
st.header("📊 Dataset Information")

st.write("""
The Iris dataset contains **150 flower samples**.

Each flower has four features:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The target variable contains three flower species:

- Setosa
- Versicolor
- Virginica
""")

# Machine Learning Model
st.header("🤖 Machine Learning Model")

st.write("""
This project uses the **Random Forest Classifier** algorithm.

The model learns patterns from the Iris dataset and predicts
the flower species based on user input.
""")

# Features
st.header("✨ Project Features")

st.write("""
✔ Dataset Exploration

✔ Data Visualization

✔ Model Training

✔ Flower Prediction

✔ Interactive User Interface
""")

st.markdown("---")

st.success("👈 Use the left sidebar to navigate through different pages.")
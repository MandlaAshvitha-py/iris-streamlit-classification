import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris

# Page Configuration
st.set_page_config(
    page_title="Dataset",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Iris Dataset")

st.write("Explore the Iris dataset used for training the Machine Learning model.")

# Load Dataset
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# Add Target Column
df["Species"] = iris.target

# Convert Target Numbers to Names
species_names = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}

df["Species"] = df["Species"].map(species_names)

# Display Dataset
st.header("📋 Complete Dataset")

st.dataframe(df, use_container_width=True)

# Dataset Shape
st.header("📐 Dataset Shape")

rows, columns = df.shape

col1, col2 = st.columns(2)

with col1:
    st.metric("Rows", rows)

with col2:
    st.metric("Columns", columns)

# Column Names
st.header("🏷 Column Names")

st.write(df.columns.tolist())

# First Five Rows
st.header("👀 First 5 Rows")

st.dataframe(df.head(), use_container_width=True)

# Last Five Rows
st.header("🔚 Last 5 Rows")

st.dataframe(df.tail(), use_container_width=True)

# Statistical Summary
st.header("📊 Statistical Summary")

st.dataframe(df.describe(), use_container_width=True)

# Missing Values
st.header("❓ Missing Values")

st.dataframe(df.isnull().sum().to_frame("Missing Values"))

# Class Distribution
st.header("🌸 Flower Species Distribution")

st.dataframe(df["Species"].value_counts().to_frame("Count"))
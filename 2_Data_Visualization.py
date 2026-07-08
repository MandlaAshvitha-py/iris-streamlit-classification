import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Page Configuration
st.set_page_config(
    page_title="Data Visualization",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Iris Data Visualization")

st.write("Visualize the Iris dataset using different charts.")

# Load Dataset
iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["Species"] = iris.target

# Convert numbers to flower names
species = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}

df["Species"] = df["Species"].map(species)

# -----------------------------
# Histogram
# -----------------------------
st.header("1️⃣ Histogram")

feature = st.selectbox(
    "Select Feature",
    iris.feature_names
)

fig, ax = plt.subplots(figsize=(8,4))

sns.histplot(
    data=df,
    x=feature,
    hue="Species",
    kde=True,
    ax=ax
)

st.pyplot(fig)

# -----------------------------
# Scatter Plot
# -----------------------------
st.header("2️⃣ Scatter Plot")

x_feature = st.selectbox(
    "X-axis",
    iris.feature_names,
    key="x"
)

y_feature = st.selectbox(
    "Y-axis",
    iris.feature_names,
    index=1,
    key="y"
)

fig, ax = plt.subplots(figsize=(7,5))

sns.scatterplot(
    data=df,
    x=x_feature,
    y=y_feature,
    hue="Species",
    s=80,
    ax=ax
)

st.pyplot(fig)

# -----------------------------
# Box Plot
# -----------------------------
st.header("3️⃣ Box Plot")

feature_box = st.selectbox(
    "Select Feature for Box Plot",
    iris.feature_names,
    key="box"
)

fig, ax = plt.subplots(figsize=(7,5))

sns.boxplot(
    data=df,
    x="Species",
    y=feature_box,
    ax=ax
)

st.pyplot(fig)

# -----------------------------
# Correlation Heatmap
# -----------------------------
st.header("4️⃣ Correlation Heatmap")

fig, ax = plt.subplots(figsize=(7,5))

sns.heatmap(
    df.iloc[:,0:4].corr(),
    annot=True,
    cmap="Blues",
    ax=ax
)

st.pyplot(fig)

# -----------------------------
# Pair Plot
# -----------------------------
st.header("5️⃣ Pair Plot")

pair = sns.pairplot(
    df,
    hue="Species"
)

st.pyplot(pair.figure)
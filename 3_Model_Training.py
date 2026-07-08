import streamlit as st
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Page Configuration
st.set_page_config(
    page_title="Model Training",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Model Training")

st.write("Train different Machine Learning models and compare their performance.")

# Load Dataset
iris = load_iris()

X = iris.data
y = iris.target

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Sidebar Algorithm Selection
algorithm = st.selectbox(
    "Select Machine Learning Algorithm",
    (
        "Logistic Regression",
        "K-Nearest Neighbors",
        "Decision Tree",
        "Random Forest"
    )
)

# Create Model
if algorithm == "Logistic Regression":
    model = LogisticRegression(max_iter=200)

elif algorithm == "K-Nearest Neighbors":
    model = KNeighborsClassifier()

elif algorithm == "Decision Tree":
    model = DecisionTreeClassifier(random_state=42)

else:
    model = RandomForestClassifier(random_state=42)

# Train Button
if st.button("Train Model"):

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    st.success(f"Model Trained Successfully!")

    st.subheader("Accuracy")

    st.metric(
        label="Accuracy",
        value=f"{accuracy*100:.2f}%"
    )

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, predictions)

    cm_df = pd.DataFrame(
        cm,
        index=iris.target_names,
        columns=iris.target_names
    )

    st.dataframe(cm_df)

    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        predictions,
        target_names=iris.target_names,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(report_df)
import streamlit as st
import pickle
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="Prediction",
    page_icon="🌸",
    layout="wide"
)

# Load Trained Model
with open("iris_model.pkl", "rb") as file:
    model = pickle.load(file)

species = [
    "Setosa",
    "Versicolor",
    "Virginica"
]

descriptions = {
    "Setosa": "Setosa is a small Iris flower with short petals. It is easily distinguishable from the other two species.",
    "Versicolor": "Versicolor is a medium-sized Iris flower with moderate petal and sepal dimensions.",
    "Virginica": "Virginica is the largest Iris flower with long petals and sepals."
}

st.title("🌸 Iris Flower Prediction")

st.write("Enter the flower measurements below to predict the Iris species.")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider(
        "Sepal Length (cm)",
        4.0,
        8.0,
        5.1
    )

    sepal_width = st.slider(
        "Sepal Width (cm)",
        2.0,
        4.5,
        3.5
    )

with col2:
    petal_length = st.slider(
        "Petal Length (cm)",
        1.0,
        7.0,
        1.4
    )

    petal_width = st.slider(
        "Petal Width (cm)",
        0.1,
        2.5,
        0.2
    )

st.markdown("---")

if st.button("🔍 Predict"):

    input_data = [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]]

    # Predict
    prediction = model.predict(input_data)
    probabilities = model.predict_proba(input_data)

    predicted_species = species[prediction[0]]
    confidence = probabilities.max() * 100

    # Display Prediction
    st.success(f"🌸 Predicted Flower: {predicted_species}")

    # Display Flower Image
    if predicted_species == "Setosa":
        st.image("images/setosa.jpg", width=300)

    elif predicted_species == "Versicolor":
        st.image("images/versicolor.jpg", width=300)

    elif predicted_species == "Virginica":
        st.image("images/virginica.jpg", width=300)

    # Display Confidence
    st.metric(
        label="Confidence",
        value=f"{confidence:.2f}%"
    )

    # Display Prediction Probabilities
    st.subheader("Prediction Probabilities")

    probability_df = pd.DataFrame({
        "Species": species,
        "Probability (%)": (probabilities[0] * 100).round(2)
    })

    st.dataframe(
        probability_df,
        use_container_width=True
    )

    # Display Description
    st.subheader("Flower Description")

    st.info(descriptions[predicted_species])
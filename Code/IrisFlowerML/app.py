import streamlit as streamlit
import pandas as panda
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# --- Page Configuration ---
streamlit.set_page_config(page_title="Iris Classifier")

# --- Header ---
streamlit.title('Iris Species Classification')
streamlit.markdown("""
This app utilizes a **Logistic Regression** machine learning model to predict the species of Iris flowers 
based on their sepal and petal measurements.
""")

# --- Sidebar for the Inputs ---
streamlit.sidebar.header('Input Parameters')

def user_input_features():
    streamlit.sidebar.markdown("**Adjust the sliders below:**")
    
    # Sliders allow user to select values within the known range of the dataset
    sepal_length = streamlit.sidebar.slider('Sepal length (cm)', 4.3, 7.9, 5.4)
    sepal_width = streamlit.sidebar.slider('Sepal width (cm)', 2.0, 4.4, 3.4)
    petal_length = streamlit.sidebar.slider('Petal length (cm)', 1.0, 6.9, 1.3)
    petal_width = streamlit.sidebar.slider('Petal width (cm)', 0.1, 2.5, 0.2)
    
    data = {'sepal length (cm)': sepal_length,
            'sepal width (cm)': sepal_width,
            'petal length (cm)': petal_length,
            'petal width (cm)': petal_width}
    
    features = panda.DataFrame(data, index=[0])
    return features

# Get user input
dataFrame = user_input_features()

# --- Display Input ---
streamlit.subheader('1. Current Flower Measurements')
streamlit.dataframe(dataFrame)

# --- Model Training (On the Fly) ---
# For this portfolio demonstreamlitration, load for simplicity.
iris = load_iris()
X = iris.data
y = iris.target

clf = LogisticRegression(max_iter=200)
clf.fit(X, y)

# --- Prediction ---
prediction = clf.predict(dataFrame)
prediction_proba = clf.predict_proba(dataFrame)

streamlit.subheader('2. Classification Result')
species_type = iris.target_names[prediction][0]
streamlit.success(f"Predicted Species: **{species_type.title()}**")

# --- Probability Breakdown ---
streamlit.subheader('3. Prediction Confidence')
streamlit.write("Probability calculated for each class:")

proba_dataFrame = panda.DataFrame(prediction_proba, columns=iris.target_names)
streamlit.bar_chart(proba_dataFrame.T)
import streamlit as st
from fairlearn.metrics import MetricFrame, demographic_parity_difference
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Title and Sidebar
st.title("Responsible AI Dashboard")
st.sidebar.title("Navigation")
option = st.sidebar.selectbox("Choose a section",
                              ["Home", "Fairness Analysis", "Model Explainability", "Transparency", "User Feedback"])

if option == "Home":
    st.write("Welcome to the Responsible AI Dashboard")
elif option == "Fairness Analysis":
    st.write("Fairness Analysis Section")
elif option == "Model Explainability":
    st.write("Model Explainability Section")
elif option == "Transparency":
    st.write("Transparency Section")
elif option == "User Feedback":
    st.write("User Feedback Section")


# Load your dataset here
@st.cache_data
def load_data():
    # Example using COMPAS dataset
    url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
    data = pd.read_csv(url)
    # Preprocess the data as needed
    return data


def fairness_analysis(data):
    X = data[['age', 'priors_count', 'juv_fel_count', 'juv_misd_count', 'juv_other_count']]
    y = data['two_year_recid']
    sensitive_feature = data['race']
    X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
        X, y, sensitive_feature, test_size=0.3, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    fairness_metric = MetricFrame(metrics=accuracy_score, y_true=y_test, y_pred=y_pred,
                                  sensitive_features=sensitive_test)
    dp_difference = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_test)

    st.write("Model Accuracy: ", accuracy)
    st.write("Demographic Parity Difference: ", dp_difference)
    st.write("Fairness Metric: ", fairness_metric.by_group)


def explainability_analysis(data):
    X = data[['age', 'priors_count', 'juv_fel_count', 'juv_misd_count', 'juv_other_count']]
    y = data['two_year_recid']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    explainer = shap.Explainer(model, X_test)
    shap_values = explainer(X_test)

    st.write("Feature Importance")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    st.pyplot(fig)


if option == "Fairness Analysis":
    data = load_data()
    fairness_analysis(data)

if option == "Model Explainability":
    data = load_data()
    explainability_analysis(data)
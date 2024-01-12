# Import necessary libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model import get_f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import shap
from streamlit_shap import st_shap
df = pd.read_csv('../data/heart.csv')

page = st.sidebar.selectbox("Choose a page", ["Home", "Model Configuration"])
st.set_option('deprecation.showPyplotGlobalUse', False)
if page == "Home":
    st.title("Dataset Exploration App")
    st.write("Welcome to the Dataset Exploration App! Choose a page from the sidebar.")

    st.markdown("## Dataset Description")
    st.title("About Set:")
    st.write("This set consists of data about different patients, and the purpose of the set is to predict whether they have a heart disease or not")

    st.title("Features:")
    st.write("- age: The person’s age in years")
    st.write("- sex: The person’s sex (1 = male, 0 = female)")
    st.write("- cp: chest pain type— Value 0: asymptomatic— Value 1: atypical angina— Value 2: non-anginal pain— Value 3: typical angina")
    st.write("- trestbps: The person’s resting blood pressure (mm Hg on admission to the hospital)")
    st.write("- chol: The person’s cholesterol measurement in mg/dl")
    st.write("- fbs: The person’s fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false)")
    st.write("- restecg: resting electrocardiographic results— Value 0: showing probable or definite left ventricular hypertrophy by Estes’ criteria— Value 1: normal— Value 2: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)")
    st.write("- thalach: The person’s maximum heart rate achieved")
    st.write("- exang: Exercise induced angina (1 = yes; 0 = no)")
    st.write("- oldpeak: ST depression induced by exercise relative to rest (‘ST’ relates to positions on the ECG plot. See more here)")
    st.write("- slope: the slope of the peak exercise ST segment — 0: downsloping; 1: flat; 2: upsloping0: downsloping; 1: flat; 2: upsloping")
    st.write("- ca: The number of major vessels (0–3)")
    st.write("- thal: A blood disorder called thalassemia Value 0: NULL (dropped from the dataset previouslyValue 1: fixed defect (no blood flow in some part of the heart)Value 2: normal blood flowValue 3: reversible defect (a blood flow is observed but it is not normal)")
    st.write("- target: Heart disease (1 = no, 0= yes)")

    st.title("Target column:")
    st.write("- target")

    st.title("ML problems it covers:")
    st.write("- Classification")
    st.write("- Missing values")
    st.title("Data Card:")
    st.write(df.head())

elif page == "Model Configuration":
    st.title("Model Configuration")
    st.markdown("## Current Head of the Dataset")
    st.write(df.head())
    st.markdown("## Input Parameters")
    
    
    param1_value = st.slider("n_estimators", min_value=20, max_value=100, value=50)
    param2_value = st.slider("max_depth", min_value=10, max_value=15, value=30)
    param3_value = st.selectbox("max_features", options=["sqrt", "log2"])
    param4_value = st.selectbox("criterion", options=["gini", "entropy"])
 
    if st.button("Train Model"):
        with st.spinner('Training the RandomForestClassifier Model...'):
            f1_score, X, y, X_train, X_test, y_train, y_test, rf_classifier = get_f1_score(param1_value, param2_value, param3_value, param4_value)
            st.success("Model trained successfully!")
            st.write("F1 Score:", f1_score)
       
        st.markdown("## Selected Parameters")
        st.write(f"n_estimators: {param1_value}")
        st.write(f"max_depth: {param2_value}")
        st.write(f"max_features: {param3_value}")
        st.write(f"criterion: {param3_value}")
      
        st.markdown("## Data Visualization")
        
        
        st.subheader("Feature Importance Bar Chart")
        feature_importances = pd.Series(rf_classifier.feature_importances_, index=X.columns)
        feature_importances = feature_importances.sort_values(ascending=False)
        st.bar_chart(feature_importances)
        
        
        st.subheader("Actual vs. Predicted Categories Bar Chart")
        predictions = rf_classifier.predict(X_test)
        result_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
        st.bar_chart(result_df)

      
        st.subheader("Class balance")
        st.bar_chart(df['target'].value_counts())
        
        
        st.subheader('SHAP Summary Plot')
        with st.spinner('Generating SHAP values and plots...'):
            explainer = shap.Explainer(rf_classifier.predict, X_train, feature_names=X.columns)
            shap_values = explainer(X_test)

        
        st_shap(shap.plots.bar(shap_values), height=900)
        
        st.subheader('Confusion Matrix')
        
        predictions = rf_classifier.predict(X_test)
        cm = confusion_matrix(y_test, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=df['target'].unique())
        disp.plot(cmap='Blues', values_format='d')
        st.pyplot()
        
        st.subheader("Correlation Heatmap")
        correlation_matrix = df.corr()
        st.write(sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f").figure)


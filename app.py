import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import (accuracy_score, classification_report, 
                           confusion_matrix, precision_score, 
                           recall_score, f1_score)
import seaborn as sns
import matplotlib.pyplot as plt
import time
from io import BytesIO

# Set page config
st.set_page_config(
    page_title="Credit Score Predictor",
    page_icon="💰",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stAlert {
        padding: 20px;
    }
    .metric-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .model-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def display_model_metrics(y_true, y_pred, model_name):
    """Display comprehensive metrics for a single model"""
    with st.expander(f"🔍 Detailed {model_name} Metrics", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.2%}")
        with col2:
            st.metric("Precision", f"{precision_score(y_true, y_pred, average='weighted'):.2%}")
        with col3:
            st.metric("Recall", f"{recall_score(y_true, y_pred, average='weighted'):.2%}")
        
        st.write("#### Classification Report")
        report = classification_report(y_true, y_pred, target_names=label_encoder.classes_, output_dict=True)
        st.json(report)
        
        # Confusion Matrix
        st.write("#### Confusion Matrix")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            confusion_matrix(y_true, y_pred),
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
            ax=ax
        )
        ax.set_title(f'{model_name} Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)

# Load models and preprocessing
@st.cache_resource
def load_models():
    try:
        models = {
            "ann": load_model("credit_model.h5"),
            "logistic": joblib.load("logistic_model.pkl"),
            "random_forest": joblib.load("rf_model.pkl"),
            "scaler": joblib.load("scaler.pkl"),
            "label_encoder": joblib.load("label_encoder.pkl"),
            "gender_encoder": joblib.load("Gender_encoder.pkl"),
            "education_encoder": joblib.load("Education_encoder.pkl"),
            "marital_encoder": joblib.load("Marital_Status_encoder.pkl"),
            "home_encoder": joblib.load("Home_Ownership_encoder.pkl")
        }
        return models
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()

models = load_models()

# Assign loaded models to variables
ann_model = models["ann"]
logistic_model = models["logistic"]
rf_model = models["random_forest"]
scaler = models["scaler"]
label_encoder = models["label_encoder"]
gender_encoder = models["gender_encoder"]
education_encoder = models["education_encoder"]
marital_encoder = models["marital_encoder"]
home_encoder = models["home_encoder"]

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Credit Score Prediction", "Model Performance Comparison", "Data Analysis"])

if page == "Credit Score Prediction":
    st.title("🔍 Credit Score Prediction")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            gender = st.selectbox("Gender", gender_encoder.classes_)
            income = st.number_input("Annual Income (₹)", min_value=0, value=500000, step=1000)
            
        with col2:
            education = st.selectbox("Education", education_encoder.classes_)
            marital_status = st.selectbox("Marital Status", marital_encoder.classes_)
            num_children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
            home_ownership = st.selectbox("Home Ownership", home_encoder.classes_)
        
        model_choice = st.radio(
            "Select Model",
            ["Neural Network", "Logistic Regression", "Random Forest"],
            horizontal=True
        )
        
        submitted = st.form_submit_button("Predict Credit Score")
        
        if submitted:
            with st.spinner('Making prediction...'):
                try:
                    # Encode categorical inputs
                    input_data = {
                        "Age": [age],
                        "Gender": [gender_encoder.transform([gender])[0]],
                        "Income": [income],
                        "Education": [education_encoder.transform([education])[0]],
                        "Marital Status": [marital_encoder.transform([marital_status])[0]],
                        "Number of Children": [num_children],
                        "Home Ownership": [home_encoder.transform([home_ownership])[0]]
                    }
                    
                    # Create DataFrame with correct column order
                    input_df = pd.DataFrame(input_data, columns=scaler.feature_names_in_)
                    input_scaled = scaler.transform(input_df)
                    
                    # Measure prediction time
                    start_time = time.time()
                    if model_choice == "Neural Network":
                        pred_proba = ann_model.predict(input_scaled)[0]
                        pred = np.argmax(pred_proba)
                        confidence = pred_proba[pred]
                    elif model_choice == "Logistic Regression":
                        pred = logistic_model.predict(input_scaled)[0]
                        pred_proba = logistic_model.predict_proba(input_scaled)[0]
                        confidence = pred_proba[pred]
                    else:
                        pred = rf_model.predict(input_df)[0]
                        pred_proba = rf_model.predict_proba(input_df)[0]
                        confidence = pred_proba[pred]
                    
                    prediction_time = (time.time() - start_time) * 1000  # in ms
                    
                    result = label_encoder.inverse_transform([pred])[0]
                    
                    # Display results
                    st.success("### Prediction Results")
                    
                    col_res1, col_res2 = st.columns(2)
                    with col_res1:
                        st.metric("Predicted Credit Score", result)
                        st.metric("Confidence", f"{confidence:.1%}")
                    
                    with col_res2:
                        st.metric("Prediction Time", f"{prediction_time:.2f} ms")
                        
                        # Show probability distribution
                        fig_prob, ax_prob = plt.subplots(figsize=(6, 3))
                        sns.barplot(
                            x=label_encoder.classes_,
                            y=pred_proba,
                            palette="viridis",
                            ax=ax_prob
                        )
                        ax_prob.set_title("Class Probabilities")
                        ax_prob.set_ylim(0, 1)
                        ax_prob.tick_params(axis='x', rotation=45)
                        st.pyplot(fig_prob)
                    
                    # Recommendations based on prediction
                    st.write("### Recommendations")
                    if result == "Low":
                        st.error("""
                        - Pay your bills on time
                        - Reduce your credit utilization ratio
                        - Avoid applying for new credit frequently
                        - Check your credit report for errors
                        """)
                    elif result == "Medium":
                        st.warning("""
                        - Continue good payment habits
                        - Keep credit utilization below 30%
                        - Maintain a mix of credit types
                        - Avoid closing old credit accounts
                        """)
                    else:
                        st.success("""
                        - Maintain your current good habits
                        - Consider premium credit cards with better rewards
                        - You may qualify for lower interest rates
                        """)
                    
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")

elif page == "Model Performance Comparison":
    st.title("📊 Model Performance Comparison")
    
    try:
        # Load test data
        @st.cache_data
        def load_test_data():
            data = pd.read_csv("./credit_score_data/Credit Score Classification Dataset.csv")
            return data
        
        data = load_test_data()
        
        with st.expander("View Test Data Sample"):
            st.dataframe(data.head())
        
        # Preprocess data
        data_encoded = data.copy()
        data_encoded["Gender"] = gender_encoder.transform(data["Gender"])
        data_encoded["Education"] = education_encoder.transform(data["Education"])
        data_encoded["Marital Status"] = marital_encoder.transform(data["Marital Status"])
        data_encoded["Home Ownership"] = home_encoder.transform(data["Home Ownership"])
        data_encoded["Credit Score"] = label_encoder.transform(data["Credit Score"])
        
        X = data_encoded[scaler.feature_names_in_]
        y = data_encoded["Credit Score"]
        X_scaled = scaler.transform(X)
        
        # Get predictions
        with st.spinner('Evaluating models...'):
            start_time_ann = time.time()
            y_pred_ann = np.argmax(ann_model.predict(X_scaled), axis=1)
            ann_time = time.time() - start_time_ann
            
            start_time_log = time.time()
            y_pred_log = logistic_model.predict(X_scaled)
            log_time = time.time() - start_time_log
            
            start_time_rf = time.time()
            y_pred_rf = rf_model.predict(X)
            rf_time = time.time() - start_time_rf
        
        models = {
            "Neural Network": {"preds": y_pred_ann, "time": ann_time},
            "Logistic Regression": {"preds": y_pred_log, "time": log_time},
            "Random Forest": {"preds": y_pred_rf, "time": rf_time}
        }
        
        # Individual Model Evaluation
        st.header("Individual Model Performance")
        tab1, tab2, tab3 = st.tabs(["Neural Network", "Logistic Regression", "Random Forest"])
        
        with tab1:
            display_model_metrics(y, models["Neural Network"]["preds"], "Neural Network")
        
        with tab2:
            display_model_metrics(y, models["Logistic Regression"]["preds"], "Logistic Regression")
        
        with tab3:
            display_model_metrics(y, models["Random Forest"]["preds"], "Random Forest")
        
        # Comprehensive Comparison
        st.header("📈 Model Comparison Dashboard")
        
        # Metrics calculation
        comparison_data = []
        for name, model in models.items():
            preds = model["preds"]
            comparison_data.append({
                "Model": name,
                "Accuracy": accuracy_score(y, preds),
                "Precision": precision_score(y, preds, average='weighted'),
                "Recall": recall_score(y, preds, average='weighted'),
                "F1 Score": f1_score(y, preds, average='weighted'),
                "Prediction Time (s)": model["time"],
                "Prediction Speed (samples/s)": len(y) / model["time"]
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Metrics Table
        st.subheader("Performance Metrics")
        st.dataframe(
            comparison_df.style
            .background_gradient(cmap='Blues', subset=['Accuracy', 'Precision', 'Recall', 'F1 Score'])
            .format({
                "Accuracy": "{:.2%}",
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1 Score": "{:.2%}",
                "Prediction Time (s)": "{:.4f}",
                "Prediction Speed (samples/s)": "{:.0f}"
            }),
            use_container_width=True,
            height=150
        )
        
        # Visualizations
        st.subheader("Performance Visualization")
        
        # Metric Comparison
        fig_metrics, ax_metrics = plt.subplots(2, 2, figsize=(15, 10))
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        for i, metric in enumerate(metrics):
            row = i // 2
            col = i % 2
            sns.barplot(
                x='Model',
                y=metric,
                data=comparison_df,
                ax=ax_metrics[row, col],
                palette='viridis'
            )
            ax_metrics[row, col].set_title(metric)
            ax_metrics[row, col].set_ylim(0, 1)
            ax_metrics[row, col].bar_label(ax_metrics[row, col].containers[0], fmt='%.2f')
        
        plt.tight_layout()
        st.pyplot(fig_metrics)
        
        # Speed Comparison
        st.write("#### Speed Comparison")
        col_speed1, col_speed2 = st.columns(2)
        
        with col_speed1:
            fig_time = plt.figure(figsize=(6, 4))
            sns.barplot(
                x='Model',
                y='Prediction Time (s)',
                data=comparison_df,
                palette='rocket'
            )
            plt.title('Total Prediction Time')
            st.pyplot(fig_time)
        
        with col_speed2:
            fig_speed = plt.figure(figsize=(6, 4))
            sns.barplot(
                x='Model',
                y='Prediction Speed (samples/s)',
                data=comparison_df,
                palette='mako'
            )
            plt.title('Prediction Throughput')
            st.pyplot(fig_speed)
        
        # Confusion Matrices
        st.write("#### Confusion Matrices Comparison")
        fig_conf, ax_conf = plt.subplots(1, 3, figsize=(18, 5))
        for i, (name, model) in enumerate(models.items()):
            sns.heatmap(
                confusion_matrix(y, model["preds"]),
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_,
                ax=ax_conf[i]
            )
            ax_conf[i].set_title(name)
            ax_conf[i].set_xlabel('Predicted')
            ax_conf[i].set_ylabel('Actual')
        
        plt.tight_layout()
        st.pyplot(fig_conf)
        
        # Model Recommendations
        st.subheader("Model Selection Guidance")
        
        best_acc = comparison_df.loc[comparison_df['Accuracy'].idxmax()]
        best_speed = comparison_df.loc[comparison_df['Prediction Speed (samples/s)'].idxmax()]
        
        st.success(f"🏆 **Best Accuracy**: {best_acc['Model']} ({best_acc['Accuracy']:.2%})")
        st.info(f"⚡ **Fastest Prediction**: {best_speed['Model']} ({best_speed['Prediction Speed (samples/s)']:.0f} samples/sec)")
        
        st.markdown("""
        - **For highest accuracy**: Choose the model with best accuracy score
        - **For production deployment**: Consider speed-accuracy tradeoff
        - **For interpretability**: Random Forest provides feature importance
        - **For complex patterns**: Neural Network may capture non-linear relationships
        """)
        
    except Exception as e:
        st.error(f"Error during model comparison: {str(e)}")

elif page == "Data Analysis":
    st.title("🔎 Data Analysis")
    
    try:
        data = pd.read_csv("./credit_score_data/Credit Score Classification Dataset.csv")
        
        st.subheader("Dataset Overview")
        st.write(f"Total records: {len(data)}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("#### First 5 Records")
            st.dataframe(data.head())
        with col2:
            st.write("#### Data Summary")
            st.dataframe(data.describe())
        
        st.subheader("Credit Score Distribution")
        fig_dist = plt.figure(figsize=(8, 4))
        sns.countplot(data=data, x='Credit Score', order=['Low', 'Medium', 'High'])
        st.pyplot(fig_dist)
        
        st.subheader("Feature Relationships")
        feature = st.selectbox("Select feature to analyze", 
                             ['Age', 'Income', 'Number of Children'])
        
        fig_rel = plt.figure(figsize=(10, 5))
        sns.boxplot(data=data, x='Credit Score', y=feature, 
                   order=['Low', 'Medium', 'High'])
        st.pyplot(fig_rel)
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
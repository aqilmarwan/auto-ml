import os
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocess import transform_new_data  # noqa: E402

def load_preprocessors():
    """Load saved preprocessors."""
    try:
        project_root = get_project_root()
        preprocessors_path = os.path.join(project_root, 'models', 'preprocessors.pkl')
        preprocessors = joblib.load(preprocessors_path)
        return preprocessors
    except FileNotFoundError:
        st.error("Preprocessors not found. Please run training first.")
        return None

def preprocess_uploaded_data(df, preprocessors):
    """
    Apply the same preprocessing pipeline to uploaded data.
    """
    return transform_new_data(df.copy(), preprocessors)

def load_models():
    """Load trained models."""
    try:
        project_root = get_project_root()
        models_path = os.path.join(project_root, 'models', 'saved_models.pkl')
        models = joblib.load(models_path)
        return models
    except FileNotFoundError:
        st.error("Models not found. Please run training first.")
        return None

def predict_fraud(model, X, model_name):
    """
    Make predictions using the selected model.
    
    Args:
        model: Trained model
        X: Preprocessed features
        model_name: Name of the model
    
    Returns:
        Predictions array
    """
    predictions = model.predict(X)
    return predictions

def main():
    st.set_page_config(page_title="Fraud Insurance Detection", page_icon="🚗", layout="wide")
    
    st.title("🚗 Fraudulent Vehicle Insurance Claims Detection")
    st.markdown("---")
    
    # Load models and preprocessors
    models = load_models()
    preprocessors = load_preprocessors()
    
    if models is None or preprocessors is None:
        st.stop()
    
    # File upload
    st.header("📤 Upload CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
            
            # Dataset preview
            st.subheader("📊 Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Model selection
            st.subheader("🤖 Model Selection")
            available_models = sorted([m for m in models.keys() if "AdaBoost" in m or "XGBoost" in m])
            selected_model_name = st.selectbox(
                "Select a model for prediction:",
                available_models
            )
            
            if st.button("🔍 Predict Fraud", type="primary"):
                with st.spinner("Preprocessing data and making predictions..."):
                    # Preprocess data
                    X_processed = preprocess_uploaded_data(df.copy(), preprocessors)
                    
                    # Get selected model
                    selected_model = models[selected_model_name]
                    
                    # Make predictions
                    predictions = predict_fraud(selected_model, X_processed, selected_model_name)
                    
                    # Add predictions to original dataframe
                    df['fraud_prediction'] = predictions
                    df['fraud_prediction_label'] = df['fraud_prediction'].map({0: 'Genuine', 1: 'Fraud'})
                    
                    # Display results
                    st.subheader("📈 Prediction Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fraud_count = int(np.sum(predictions == 1))
                        genuine_count = int(np.sum(predictions == 0))
                        total = len(predictions)
                        
                        st.metric("Total Claims", total)
                        st.metric("Fraudulent Claims", fraud_count, delta=f"{fraud_count/total*100:.1f}%")
                        st.metric("Genuine Claims", genuine_count, delta=f"{genuine_count/total*100:.1f}%")
                    
                    with col2:
                        # Bar chart
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(figsize=(6, 4))
                        counts = [genuine_count, fraud_count]
                        labels = ['Genuine', 'Fraud']
                        colors = ['#2ecc71', '#e74c3c']
                        ax.bar(labels, counts, color=colors)
                        ax.set_ylabel('Count')
                        ax.set_title('Fraud vs Genuine Claims')
                        for i, v in enumerate(counts):
                            ax.text(i, v + max(counts)*0.01, str(v), ha='center', va='bottom')
                        st.pyplot(fig)
                    
                    # Display predictions
                    st.subheader("🔍 Predictions Table")
                    display_cols = ['fraud_prediction_label'] + [col for col in df.columns if col not in ['fraud_prediction', 'fraud_prediction_label']]
                    st.dataframe(df[display_cols], use_container_width=True)
                    
                    # Download button
                    st.subheader("💾 Download Results")
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV with Predictions",
                        data=csv,
                        file_name="fraud_predictions.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.exception(e)
    
    else:
        st.info("👆 Please upload a CSV file to get started.")
        
        # Instructions
        st.markdown("---")
        st.subheader("📋 Instructions")
        st.markdown("""
        1. **Upload CSV**: Use the Kaggle auto insurance claims data (or schema-compatible file)
        2. **Select Model**: Choose between AdaBoost or XGBoost (calibrated option if available)
        3. **Predict**: Click the "Predict Fraud" button
        4. **Review**: View fraud vs genuine counts and detailed predictions
        5. **Download**: Download the results as a CSV file
        
        **Expected columns (examples):**
        policy_bind_date, incident_date, policy_csl, policy_annual_premium, policy_deductable/deductible,
        months_as_customer, umbrella_limit, insured_* fields, incident_hour_of_the_day, incident_state,
        property_damage, police_report_available, injury_claim, property_claim, vehicle_claim, fraud_reported (optional for scoring)
        """)

if __name__ == '__main__':
    main()


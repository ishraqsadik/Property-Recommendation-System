import streamlit as st

# Page configuration - MUST be first
st.set_page_config(
    page_title="Property Test",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Property System Test")

try:
    # Test basic imports
    st.write("Testing basic imports...")
    import pandas as pd
    import numpy as np
    st.success("✅ Basic imports successful")
    
    # Test data file existence
    st.write("Testing data file...")
    import os
    if os.path.exists("appraisals_dataset.json"):
        st.success("✅ Data file found")
    else:
        st.error("❌ Data file not found")
    
    # Test ML module imports
    st.write("Testing ML module imports...")
    from app.ml.data_processing import load_and_process_data
    st.success("✅ Data processing import successful")
    
    from app.ml import initialize_model_from_data
    st.success("✅ Model import successful")
    
    # Test data loading
    if st.button("Test Data Loading"):
        with st.spinner("Loading data..."):
            try:
                subjects_df, comps_df, properties_df = load_and_process_data("appraisals_dataset.json")
                st.success(f"✅ Data loaded: {len(subjects_df)} subjects, {len(comps_df)} comps, {len(properties_df)} properties")
                
                # Show sample data
                st.subheader("Sample Subjects")
                st.dataframe(subjects_df.head())
                
                st.subheader("Sample Properties")
                st.dataframe(properties_df.head())
                
            except Exception as e:
                st.error(f"❌ Data loading failed: {str(e)}")
                st.exception(e)
    
    # Test model loading
    if st.button("Test Model Loading"):
        with st.spinner("Loading model..."):
            try:
                model = initialize_model_from_data(
                    json_path="appraisals_dataset.json",
                    model_path="data/models/recommendation_model.pkl"
                )
                st.success("✅ Model loaded successfully")
                st.write(f"Model type: {type(model)}")
                
                if hasattr(model, 'test_property_ids'):
                    st.write(f"Test property IDs: {len(model.test_property_ids)} found")
                else:
                    st.warning("No test_property_ids found in model")
                    
            except Exception as e:
                st.error(f"❌ Model loading failed: {str(e)}")
                st.exception(e)

except Exception as e:
    st.error(f"❌ Import error: {str(e)}")
    st.exception(e) 
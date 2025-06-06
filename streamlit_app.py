import streamlit as st

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Property Recommendation System",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import joblib
import os
from datetime import datetime
import warnings
import json
warnings.filterwarnings('ignore')

# Import your existing ML modules
from app.ml import initialize_model_from_data, initialize_feedback_system
from app.ml.data_processing import load_and_process_data, deduplicate_properties
from app.ml.utils import format_property_record
from app.ml.feature_engineering import FEATURES
from app.explainability import PropertyExplainer, create_explainer_for_streamlit, display_property_explanation

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        border: 1px solid #e1e5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .property-card {
        background-color: #ffffff;
        border: 1px solid #e1e5e9;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .subject-property {
        background-color: #e8f4f8;
        border-left: 4px solid #1f77b4;
    }
    .true-comp {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .recommendation-score {
        font-size: 1.2em;
        font-weight: bold;
        color: #1f77b4;
    }
    
    /* Hide sidebar completely */
    .css-1d391kg {
        display: none;
    }
    
    /* Hide sidebar toggle button */
    .css-1rs6os {
        display: none;
    }
    
    /* Hide sidebar collapse button */
    .css-1lcbmhc {
        display: none;
    }
    
    /* Ensure main content takes full width */
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: none;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to make data JSON serializable
def make_json_serializable(obj):
    """Convert pandas objects to JSON serializable format"""
    if hasattr(obj, 'to_dict'):
        # It's a pandas Series/DataFrame row
        data = obj.to_dict()
    elif isinstance(obj, dict):
        data = obj.copy()
    else:
        return obj
    
    # Convert any pandas Timestamps to strings
    for key, value in data.items():
        if hasattr(value, 'strftime'):  # It's a datetime/timestamp
            data[key] = value.strftime('%Y-%m-%d') if value else None
        elif pd.isna(value):  # Handle NaN values
            data[key] = None
        elif hasattr(value, 'item'):  # NumPy scalar
            data[key] = value.item()
    
    return data

class PropertyRecommendationApp:
    def __init__(self):
        self.initialize_session_state()
        # Automatically load model and data on app start
        if not st.session_state.model_loaded:
            self.auto_load_on_startup()
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'selected_subject' not in st.session_state:
            st.session_state.selected_subject = None
        if 'recommendations' not in st.session_state:
            st.session_state.recommendations = None
        if 'subject_property' not in st.session_state:
            st.session_state.subject_property = None
        # Add data storage in session state
        if 'subjects_df' not in st.session_state:
            st.session_state.subjects_df = None
        if 'comps_df' not in st.session_state:
            st.session_state.comps_df = None
        if 'properties_df' not in st.session_state:
            st.session_state.properties_df = None
        if 'model' not in st.session_state:
            st.session_state.model = None
        if 'explainer' not in st.session_state:
            st.session_state.explainer = None
        # Add feedback system storage
        if 'feedback_system' not in st.session_state:
            st.session_state.feedback_system = None
        if 'feedback_stats' not in st.session_state:
            st.session_state.feedback_stats = None
        # Track loading state
        if 'loading_error' not in st.session_state:
            st.session_state.loading_error = None
        # Add custom form state
        if 'show_custom_form' not in st.session_state:
            st.session_state.show_custom_form = False
    
    def auto_load_on_startup(self):
        """Automatically load model and data when app starts"""
        try:
            # Load data first
            data_path = "appraisals_dataset.json"
            if not os.path.exists(data_path):
                st.session_state.loading_error = "❌ Property data file not found!"
                return False
            
            # Load and process data - store in session state
            subjects_df, comps_df, properties_df = load_and_process_data(data_path)
            st.session_state.subjects_df = subjects_df
            st.session_state.comps_df = comps_df
            st.session_state.properties_df = deduplicate_properties(properties_df)
            
            # Ensure property IDs are strings
            if 'id' in st.session_state.properties_df.columns:
                st.session_state.properties_df['id'] = st.session_state.properties_df['id'].astype(str)
            
            # Initialize model properly - store in session state
            model_path = "data/models/recommendation_model.pkl"
            
            # Use initialize_model_from_data which returns a PropertyRecommendationModel
            st.session_state.model = initialize_model_from_data(
                json_path=data_path,
                model_path=model_path
            )
            
            # Initialize explainer - store in session state
            st.session_state.explainer = create_explainer_for_streamlit(st.session_state.model)
            
            # Initialize feedback system - store in session state
            st.session_state.feedback_system = initialize_feedback_system(
                model=st.session_state.model,
                feedback_db_path="data/feedback/feedback_database.json",
                training_data_path="data/processed/training_data.pkl",
                model_save_path="data/models/recommendation_model.pkl",
                retrain_threshold=5  # Lower threshold for demo purposes
            )
            if st.session_state.feedback_system:
                # Load current feedback stats
                st.session_state.feedback_stats = st.session_state.feedback_system.get_feedback_stats()
            
            st.session_state.model_loaded = True
            st.session_state.data_loaded = True
            st.session_state.loading_error = None
            
        except Exception as e:
            error_msg = f"❌ Error during initialization: {str(e)}"
            st.session_state.loading_error = error_msg
            return False
        return True
    
    def get_test_subjects(self, limit: int = 20):
        """Get test subjects similar to the API endpoint"""
        try:
            # Use session state data instead of instance variables
            subjects_df = st.session_state.get('subjects_df')
            properties_df = st.session_state.get('properties_df')
            model = st.session_state.get('model')
            
            # Check if data is loaded
            if subjects_df is None or subjects_df.empty:
                return []
            
            # Find matching test subjects based on model test property IDs
            if hasattr(model, 'test_property_ids') and model.test_property_ids:
                # Find matching test subjects
                test_subjects = []
                
                for order_id in subjects_df['order_id'].unique():
                    # Check if any property in this order is in the test set
                    order_props = properties_df[properties_df['order_id'] == order_id]
                    if any(str(prop_id) in model.test_property_ids for prop_id in order_props['id'].astype(str)):
                        test_subjects.append(order_id)
                
                if not test_subjects:
                    test_subjects = subjects_df['order_id'].unique()[:limit]
            else:
                test_subjects = subjects_df['order_id'].unique()[:limit]
            
            # Get subject properties
            subjects = []
            for order_id in test_subjects[:limit]:
                subject_row = subjects_df[subjects_df['order_id'] == order_id]
                if not subject_row.empty:
                    prop_dict = subject_row.iloc[0].to_dict()
                    formatted_prop = format_property_record(prop_dict)
                    formatted_prop.pop('sale_price', None)  # Remove sale_price from subject
                    formatted_prop['id'] = order_id  # Use order_id as id
                    subjects.append(formatted_prop)
            
            return subjects
            
        except Exception as e:
            return []
    
    def get_property_recommendations(self, subject_property_dict):
        """Get property recommendations for a subject property"""
        try:
            # Use session state data instead of instance variables
            properties_df = st.session_state.get('properties_df')
            comps_df = st.session_state.get('comps_df')
            model = st.session_state.get('model')
            
            order_id = subject_property_dict.get('order_id')
            subject_address = subject_property_dict.get('address', '')
            
            # Get properties for this subject's order_id
            order_properties = properties_df[properties_df['order_id'] == order_id].copy()
            
            # Exclude the subject property itself
            if subject_address:
                order_properties = order_properties[
                    order_properties['address'].str.lower() != subject_address.lower()
                ]
            
            if order_properties.empty:
                order_properties = properties_df.copy()
            
            # Get comp data for this order
            order_comps = comps_df[comps_df['order_id'] == order_id].copy()
            
            if not order_comps.empty:
                # Join comp information
                order_properties = order_properties.merge(
                    order_comps[['address', 'comp_rank', 'distance_to_subject']],
                    on='address',
                    how='left'
                )
            
            # Create mappings for true comp injection
            comp_mapping = {
                'sale_price': 'sale_price',
                'sale_date': 'sale_date',
                'age': 'age',
                'gla': 'gla',
                'structure_type': 'structure_type',
                'stories': 'stories',
                'bedrooms': 'bedrooms',
                'bed_count': 'bedrooms',
                'full_baths': 'full_baths',
                'bath_count': 'full_baths',
                'half_baths': 'half_baths',
                'room_count': 'room_count',
                'address': 'address',
                'city': 'city',
                'comp_rank': 'comp_rank'
            }
            
            # Prepare true comps for injection
            true_comps_for_injection = None
            if not order_comps.empty:
                true_comps_for_injection = order_comps.rename(columns=comp_mapping)
                true_comps_for_injection['is_true_comp'] = 1
                
                # Set IDs and order_id
                for i, row in true_comps_for_injection.iterrows():
                    if pd.isna(row.get('id')):
                        true_comps_for_injection.at[i, 'id'] = f"{order_id}_true_comp_{i}"
                    true_comps_for_injection.at[i, 'order_id'] = order_id
                    
                    # Ensure required fields are not NaN
                    for field in ['city', 'province']:
                        if pd.isna(row.get(field)):
                            true_comps_for_injection.at[i, field] = ""
            
            # Generate recommendations using model
            recommendations = model.recommend_comps(
                order_properties,
                subject_data=subject_property_dict,
                n_recommendations=10,
                threshold=0.1,
                true_comps_df=true_comps_for_injection
            )
            
            return recommendations
            
        except Exception as e:
            st.error(f"Error getting recommendations: {str(e)}")
            st.exception(e)
            return pd.DataFrame()
    
    def display_subject_property(self, subject_dict):
        """Display the selected subject property"""
        st.subheader("🏠 Subject Property")
        
        with st.container():
            st.markdown('<div class="property-card subject-property">', unsafe_allow_html=True)
            
            # Large, bold property address
            address = subject_dict.get('address', 'Address Not Available')
            st.markdown(f"### **{address}**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Property Details**")
                st.write(f"**City:** {subject_dict.get('city', 'N/A')}")
                st.write(f"**Order ID:** {subject_dict.get('order_id', 'N/A')}")
            
            with col2:
                st.markdown("**Features**")
                st.write(f"**GLA:** {subject_dict.get('gla', 'N/A')} sq ft")
                st.write(f"**Bedrooms:** {subject_dict.get('bedrooms', 'N/A')}")
                st.write(f"**Age:** {subject_dict.get('age', 'N/A')} years")
            
            with col3:
                st.markdown("**Property Type**")
                st.write(f"**Structure:** {subject_dict.get('structure_type', 'N/A')}")
                st.write(f"**Stories:** {subject_dict.get('stories', 'N/A')}")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    def display_recommendations(self, recommendations):
        """Display property recommendations"""
        if recommendations.empty:
            st.info("No recommendations found for this property.")
            return
        
        st.subheader(f"🎯 Top {len(recommendations)} Property Recommendations")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_score = recommendations['score'].mean() if 'score' in recommendations.columns else 0
            st.metric("Average Score", f"{avg_score:.3f}")
        
        with col2:
            true_comps = recommendations['is_true_comp'].sum() if 'is_true_comp' in recommendations.columns else 0
            st.metric("True Comps", f"{true_comps}")
        
        with col3:
            avg_price = recommendations['sale_price'].mean() if 'sale_price' in recommendations.columns else 0
            st.metric("Average Price", f"${avg_price:,.0f}" if avg_price > 0 else "N/A")
        
        with col4:
            avg_gla = recommendations['gla'].mean() if 'gla' in recommendations.columns else 0
            st.metric("Average GLA", f"{avg_gla:,.0f} sq ft" if avg_gla > 0 else "N/A")
        
        # Display each recommendation
        for idx, (_, property_data) in enumerate(recommendations.iterrows()):
            is_true_comp = property_data.get('is_true_comp', 0)
            card_class = "property-card true-comp" if is_true_comp else "property-card"
            
            with st.expander(
                f"{'🟢' if is_true_comp else '🔵'} Property {idx + 1}: {property_data.get('address', 'Address Not Available')} "
                f"{'(TRUE COMP)' if is_true_comp else ''}", 
                expanded=(idx < 3)
            ):
                st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
                
                # Large, bold property address
                address = property_data.get('address', 'Address Not Available')
                st.markdown(f"### **{address}**")
                
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    st.markdown("**Property Details**")
                    st.write(f"**City:** {property_data.get('city', 'N/A')}")
                    st.write(f"**Price:** ${property_data.get('sale_price', 0):,.0f}")
                    st.write(f"**GLA:** {property_data.get('gla', 'N/A')} sq ft")
                    st.write(f"**Bedrooms:** {property_data.get('bedrooms', 'N/A')}")
                    st.write(f"**Age:** {property_data.get('age', 'N/A')} years")
                
                with col2:
                    st.markdown("**Recommendation Analysis**")
                    score = property_data.get('score', 0)
                    st.markdown(f'<div class="recommendation-score">Match Score: {score:.3f}</div>', unsafe_allow_html=True)
                    
                    if is_true_comp:
                        st.success("✅ This is a TRUE COMPARABLE from the appraisal")
                        comp_rank = property_data.get('comp_rank', 'N/A')
                        st.write(f"**Comp Rank:** {comp_rank}")
                    
                    # SHAP explanation button
                    if st.button(f"🔍 Explain Why This Property Was Recommended", key=f"explain_{idx}"):
                        st.session_state[f'show_explanation_{idx}'] = True
                
                with col3:
                    st.markdown("**Property Type**")
                    if property_data.get('structure_type'):
                        st.write(f"**Type:** {property_data['structure_type']}")
                    if property_data.get('stories'):
                        st.write(f"**Stories:** {property_data['stories']}")
                    if property_data.get('comp_rank'):
                        st.write(f"**Comp Rank:** {property_data['comp_rank']}")
                
                # Show SHAP explanation if requested
                if st.session_state.get(f'show_explanation_{idx}', False):
                    st.markdown("---")
                    explainer = st.session_state.get('explainer')
                    if explainer:
                        # Create a single-row dataframe for this property
                        single_property_df = pd.DataFrame([property_data])
                        display_property_explanation(
                            explainer, 
                            single_property_df, 
                            0,  # First (and only) row
                            property_data.get('address', f'Property {idx + 1}')
                        )
                    else:
                        st.error("Explainer not available. Please reload the model.")
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    def create_feature_importance_dashboard(self, recommendations):
        """Create a dashboard showing overall feature importance"""
        explainer = st.session_state.get('explainer')
        if recommendations.empty or not explainer:
            return
        
        st.subheader("📊 Feature Importance Analysis")
        
        try:
            # Calculate SHAP values for top recommendations
            all_explanations = []
            
            for idx in range(min(len(recommendations), 5)):  # Limit to top 5 for performance
                single_property_df = pd.DataFrame([recommendations.iloc[idx]])
                explanation = explainer.explain_property_recommendation(single_property_df, 0, top_features=10)
                if explanation is not None:
                    all_explanations.append(explanation)
            
            if all_explanations:
                # Combine all explanations
                combined_df = pd.concat(all_explanations, ignore_index=True)
                
                # Calculate average importance per feature
                feature_importance = combined_df.groupby('feature')['abs_shap'].mean().sort_values(ascending=False).head(10)
                
                # Create visualization
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=feature_importance.values,
                    y=feature_importance.index,
                    orientation='h',
                    marker_color='#1f77b4',
                    text=[f"{val:.3f}" for val in feature_importance.values],
                    textposition='outside'
                ))
                
                fig.update_layout(
                    title="Average Feature Importance Across Top Properties",
                    xaxis_title="Average |SHAP| Value",
                    yaxis_title="Features",
                    height=400,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show interpretation
                st.markdown("#### 💡 Key Insights:")
                top_3_features = feature_importance.head(3)
                for feature, importance in top_3_features.items():
                    clean_feature = feature.replace('cat__', '').replace('num__', '')
                    st.markdown(f"• **{clean_feature}** is a key factor in recommendations (importance: {importance:.3f})")
        
        except Exception as e:
            st.error(f"Error creating feature importance dashboard: {str(e)}")
    
    def display_feedback_collection(self, subject_property, recommendations):
        """Display feedback collection interface"""
        feedback_system = st.session_state.get('feedback_system')
        if not feedback_system or recommendations.empty:
            return
        
        st.subheader("🎯 Help Improve Our Recommendations")
        st.markdown("Your feedback helps us learn and provide better property recommendations!")
        
        # Feedback form
        with st.form("feedback_form"):
            st.markdown("#### Rate Our Top 3 Recommendations")
            st.markdown("*Check the properties you think are good comparables:*")
            
            # Show top 3 recommendations with checkboxes
            approved_properties = []
            top_3 = recommendations.head(3)
            
            for idx, (_, property_data) in enumerate(top_3.iterrows()):
                address = property_data.get('address', f'Property {idx + 1}')
                price = property_data.get('sale_price', 0)
                gla = property_data.get('gla', 'N/A')
                score = property_data.get('score', 0)
                is_true_comp = property_data.get('is_true_comp', 0)
                
                # Create property display with true comp indicator
                true_comp_indicator = " 🟢 (TRUE COMP)" if is_true_comp else ""
                property_display = f"**Property {idx + 1}**: {address}{true_comp_indicator}"
                property_details = f"${price:,.0f} | {gla} sq ft | Score: {score:.3f}"
                
                col1, col2 = st.columns([1, 4])
                with col1:
                    is_approved = st.checkbox("✅ Good", key=f"approve_top_{idx}")
                with col2:
                    st.markdown(property_display)
                    st.caption(property_details)
                
                if is_approved:
                    approved_properties.append(property_data)
            
            st.markdown("---")
            
            # Overall assessment based on selections
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Overall Assessment")
                if len(approved_properties) == 3:
                    st.success("👍 All recommendations approved")
                    overall_approved = True
                elif len(approved_properties) > 0:
                    st.warning(f"👌 {len(approved_properties)}/3 recommendations approved")
                    overall_approved = False
                else:
                    st.error("👎 No recommendations approved")
                    overall_approved = False
            
            with col2:
                feedback_comments = st.text_area(
                    "Additional comments (optional):",
                    placeholder="Any specific feedback about the recommendations...",
                    max_chars=500,
                    key="feedback_comments"
                )
            
            # Submit feedback
            submitted = st.form_submit_button("📤 Submit Feedback", type="primary")
            
            if submitted:
                if len(approved_properties) == 0:
                    st.warning("⚠️ Please select at least one property or provide feedback!")
                    return
                
                # Prepare recommendation data (all top 3)
                recommended_properties = [make_json_serializable(row) for _, row in top_3.iterrows()]
                
                # Prepare approved properties (convert to JSON serializable)
                approved_properties_serializable = [make_json_serializable(prop) for prop in approved_properties]
                
                # Prepare subject data (convert to JSON serializable)
                subject_data_serializable = make_json_serializable(subject_property)
                
                # Submit feedback
                try:
                    result = feedback_system.add_feedback(
                        subject_id=str(subject_property.get('order_id', 'unknown')),
                        subject_data=subject_data_serializable,
                        recommended_properties=recommended_properties,
                        is_approved=overall_approved,
                        selected_properties=approved_properties_serializable if not overall_approved else None,
                        comments=feedback_comments
                    )
                    
                    # Update feedback stats
                    st.session_state.feedback_stats = feedback_system.get_feedback_stats()
                    
                    # Show success message
                    st.success("✅ Thank you for your feedback!")
                    
                    # Show retraining info
                    new_feedback_count = result.get('new_feedback_count', 0)
                    threshold = result.get('retrain_threshold', 5)
                    
                    if new_feedback_count >= threshold:
                        st.info(f"🔄 Model will be retrained with {new_feedback_count} new feedback entries!")
                    else:
                        st.info(f"📊 Feedback collected ({new_feedback_count}/{threshold} needed for retraining)")
                    
                    # Clear form by rerunning
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error submitting feedback: {str(e)}")
    
    def display_feedback_stats(self):
        """Display feedback learning statistics"""
        feedback_stats = st.session_state.get('feedback_stats')
        if not feedback_stats:
            return
        
        st.subheader("📈 Learning Progress")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Feedback", 
                feedback_stats.get('total_count', 0),
                help="Total feedback entries collected"
            )
        
        with col2:
            approval_rate = feedback_stats.get('approval_rate', 0) * 100
            st.metric(
                "Approval Rate", 
                f"{approval_rate:.1f}%",
                help="Percentage of recommendations that were approved"
            )
        
        with col3:
            st.metric(
                "Processed", 
                feedback_stats.get('processed_count', 0),
                help="Feedback entries used for model training"
            )
        
        with col4:
            unprocessed = feedback_stats.get('unprocessed_count', 0)
            threshold = 5  # Match the threshold from initialization
            st.metric(
                "Pending", 
                f"{unprocessed}/{threshold}",
                help="New feedback pending model retraining"
            )
        
        # Show last retrain info
        last_retrain = feedback_stats.get('last_retrain')
        if last_retrain:
            st.caption(f"Last model update: {last_retrain[:19].replace('T', ' ')}")
        else:
            st.caption("Model has not been retrained with feedback yet")
    
    def display_manual_retraining(self):
        """Display manual retraining interface"""
        feedback_system = st.session_state.get('feedback_system')
        if not feedback_system:
            return
        
        st.subheader("🔧 Manual Model Retraining")
        
        feedback_stats = st.session_state.get('feedback_stats', {})
        unprocessed = feedback_stats.get('unprocessed_count', 0)
        
        if unprocessed > 0:
            st.info(f"There are {unprocessed} new feedback entries available for training.")
            
            if st.button("🚀 Retrain Model Now", type="primary"):
                with st.spinner("Retraining model with feedback data..."):
                    try:
                        result = feedback_system.retrain_model(force=True)
                        
                        if result.get('status') == 'success':
                            st.success("✅ Model retrained successfully!")
                            
                            # Show metrics if available
                            metrics = result.get('metrics', {})
                            if metrics:
                                st.json(metrics)
                            
                            # Update stats
                            st.session_state.feedback_stats = feedback_system.get_feedback_stats()
                            
                            # Reload model in session state
                            st.info("🔄 Reloading updated model...")
                            st.session_state.model_loaded = False
                            st.rerun()
                            
                        else:
                            st.warning(f"Retraining skipped: {result.get('reason', 'Unknown reason')}")
                            
                    except Exception as e:
                        st.error(f"Error during retraining: {str(e)}")
                        # Show more details for debugging
                        st.exception(e)
        else:
            st.info("No new feedback available for retraining.")
            
            # Still allow retraining with all feedback
            if st.button("🔄 Retrain with All Feedback", type="secondary"):
                with st.spinner("Retraining model with all feedback..."):
                    try:
                        result = feedback_system.retrain_model(force=True)
                        if result.get('status') == 'success':
                            st.success("✅ Model retrained successfully!")
                            
                            # Update stats and reload
                            st.session_state.feedback_stats = feedback_system.get_feedback_stats()
                            st.info("🔄 Reloading updated model...")
                            st.session_state.model_loaded = False
                            st.rerun()
                        else:
                            st.info(f"Retraining result: {result.get('reason', 'No feedback available')}")
                    except Exception as e:
                        st.error(f"Error during retraining: {str(e)}")
                        st.exception(e)
        
        # Export feedback option
        st.markdown("---")
        if st.button("📊 Export Feedback Data"):
            try:
                export_path = feedback_system.export_feedback_to_csv("data/feedback/exported_feedback.csv")
                st.success(f"✅ Feedback data exported to: {export_path}")
                
                # Show download link
                with open(export_path, 'rb') as f:
                    st.download_button(
                        "📥 Download CSV",
                        f.read(),
                        file_name="feedback_data.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"Error exporting feedback: {str(e)}")
    
    def run(self):
        """Main application runner"""
        # Header
        st.title("🏠 Property Recommendation System")
        st.markdown("AI-powered property recommendations with explainable insights")
        
        # Quick OpenAI status check (collapsed by default)
        env_openai_key = os.getenv("OPENAI_API_KEY")
        if env_openai_key:
            st.success("🤖 Enhanced AI explanations are enabled")
        else:
            with st.expander("ℹ️ Optional: Enhanced AI explanations available with OpenAI API key"):
                st.info("Enhanced explanations are available if you configure an OpenAI API key in your Streamlit secrets.")
        
        # Check if there was a loading error
        if st.session_state.loading_error:
            st.error(st.session_state.loading_error)
            st.markdown("### Troubleshooting:")
            st.markdown("- Ensure `appraisals_dataset.json` exists in the project root")
            st.markdown("- Check that all required dependencies are installed")
            st.markdown("- Verify the model files exist in `data/models/`")
            
            # Retry button
            if st.button("🔄 Retry Initialization", type="primary"):
                st.session_state.model_loaded = False
                st.session_state.data_loaded = False
                st.session_state.loading_error = None
                st.rerun()
            return
        
        # Main content
        if not st.session_state.model_loaded:
            # Show simple loading message
            st.markdown("### 🏠 Property Recommendation System")
            st.markdown("*Loading AI model and property data...*")
            return
        
        # Test property selection
        st.subheader("📋 Select a Test Property")
        
        # Get test subjects
        test_subjects = self.get_test_subjects()
        
        if not test_subjects:
            st.error("No test subjects available. Please check your data.")
            return
        
        # Create dropdown options
        subject_options = {}
        for subject in test_subjects:
            address = subject.get('address', 'Unknown Address')
            order_id = subject.get('order_id', 'Unknown ID')
            city = subject.get('city', 'Unknown City')
            display_name = f"{address} ({city}) - ID: {order_id}"
            subject_options[display_name] = subject
        
        # Subject selection
        selected_display = st.selectbox(
            "Choose a test property to analyze:",
            list(subject_options.keys()),
            index=0 if subject_options else None
        )
        
        # Button for custom property entry
        st.markdown("---")
        if st.button("📝 Enter New Test Property", type="secondary"):
            st.session_state.show_custom_form = True
        
        # Custom property form
        if st.session_state.get('show_custom_form', False):
            st.subheader("📝 Enter Custom Property Details")
            
            with st.form("custom_property_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Property Details**")
                    custom_address = st.text_input("Address*", placeholder="123 Main St")
                    custom_city = st.text_input("City*", placeholder="Tampa")
                    custom_gla = st.number_input("GLA (sq ft)*", min_value=500, max_value=10000, value=1500, step=50)
                    custom_bedrooms = st.number_input("Bedrooms*", min_value=1, max_value=10, value=3)
                    custom_age = st.number_input("Age (years)*", min_value=0, max_value=100, value=10)
                
                with col2:
                    st.markdown("**Property Features**")
                    custom_structure = st.selectbox("Structure Type", 
                        ["Single Family Detached", "Townhouse", "Condominium", "Semi-Detached", "Other"])
                    custom_stories = st.number_input("Stories", min_value=1, max_value=4, value=1)
                    custom_full_baths = st.number_input("Full Bathrooms", min_value=1, max_value=10, value=2)
                    custom_half_baths = st.number_input("Half Bathrooms", min_value=0, max_value=5, value=0)
                    custom_room_count = st.number_input("Total Rooms", min_value=3, max_value=20, value=7)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    submit_custom = st.form_submit_button("🔍 Get Recommendations", type="primary")
                with col2:
                    cancel_custom = st.form_submit_button("❌ Cancel")
                
                if submit_custom:
                    if custom_address and custom_city and custom_gla:
                        # Create custom property dict
                        custom_property = {
                            'address': custom_address,
                            'city': custom_city,
                            'gla': custom_gla,
                            'bedrooms': custom_bedrooms,
                            'age': custom_age,
                            'structure_type': custom_structure,
                            'stories': custom_stories,
                            'full_baths': custom_full_baths,
                            'half_baths': custom_half_baths,
                            'room_count': custom_room_count,
                            'order_id': f"custom_{hash(custom_address)}",  # Generate unique ID
                            'id': f"custom_{hash(custom_address)}"
                        }
                        
                        with st.spinner("🔍 Analyzing your custom property and generating recommendations..."):
                            recommendations = self.get_property_recommendations(custom_property)
                            st.session_state.selected_subject = custom_property
                            st.session_state.recommendations = recommendations
                            st.session_state.show_custom_form = False
                            st.rerun()
                    else:
                        st.error("Please fill in all required fields marked with *")
                
                if cancel_custom:
                    st.session_state.show_custom_form = False
                    st.rerun()
        
        # Regular property selection (only show if custom form is not displayed)
        elif selected_display and st.button("🔍 Get Recommendations", type="primary"):
            selected_subject = subject_options[selected_display]
            
            with st.spinner("🔍 Analyzing property and generating recommendations..."):
                recommendations = self.get_property_recommendations(selected_subject)
                st.session_state.selected_subject = selected_subject
                st.session_state.recommendations = recommendations
        
        # Display results
        if st.session_state.selected_subject is not None:
            # Display subject property
            self.display_subject_property(st.session_state.selected_subject)
            
            if st.session_state.recommendations is not None and not st.session_state.recommendations.empty:
                # Feature importance dashboard
                self.create_feature_importance_dashboard(st.session_state.recommendations)
                
                # Display recommendations
                self.display_recommendations(st.session_state.recommendations)
                
                # Display feedback collection
                self.display_feedback_collection(st.session_state.selected_subject, st.session_state.recommendations)
                
                # Display feedback stats
                self.display_feedback_stats()
                
                # Display manual retraining
                self.display_manual_retraining()
            else:
                st.warning("No recommendations were generated for this property.")
        
        # Footer
        st.markdown("---")

# Run the application
if __name__ == "__main__":
    app = PropertyRecommendationApp()
    app.run() 
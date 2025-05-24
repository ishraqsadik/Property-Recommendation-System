import streamlit as st

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Property Recommendation System",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
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
            
            # Debug: Check if data is loaded
            if subjects_df is None:
                st.error("Subjects data is None. Please reload the model and data.")
                return []
            
            if subjects_df.empty:
                st.error("Subjects data is empty. Please check your data file.")
                return []
            
            # Debug: Show model test property IDs info
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
            st.error(f"Error getting test subjects: {str(e)}")
            st.exception(e)
            # Debug: Show what data we have
            st.write(f"Debug info:")
            st.write(f"- subjects_df is None: {st.session_state.get('subjects_df') is None}")
            st.write(f"- properties_df is None: {st.session_state.get('properties_df') is None}")
            st.write(f"- model is None: {st.session_state.get('model') is None}")
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
                st.warning(f"No candidates found for order_id {order_id}. Using all properties as fallback.")
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
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Property Details**")
                st.write(f"**Address:** {subject_dict.get('address', 'N/A')}")
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
                
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    st.markdown("**Property Details**")
                    st.write(f"**Address:** {property_data.get('address', 'N/A')}")
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
        
        # Sidebar
        with st.sidebar:
            st.header("⚙️ System Status")
            
            # System status display
            if st.session_state.model_loaded:
                st.write("Model Ready")
                st.write("Data Loaded")
                
                # Show system info
                model = st.session_state.get('model')
                if model and hasattr(model, 'test_property_ids'):
                    st.write(f"📋 {len(model.test_property_ids)} test properties available")
                
                # Manual reload option (advanced users)
                with st.expander("🔧 Advanced Options"):
                    if st.button("🔄 Reload System"):
                        st.session_state.model_loaded = False
                        st.session_state.data_loaded = False
                        st.session_state.loading_error = None
                        st.rerun()
            else:
                st.write("System is initializing...")
            
            st.markdown("---")
            st.markdown("### 📊 Features")
            st.markdown("• Test property selection")
            st.markdown("• AI-powered recommendations") 
            st.markdown("• SHAP explainability")
            st.markdown("• **Enhanced AI explanations** (requires OpenAI API)")
            st.markdown("• True comparables identification")
            st.markdown("• Feature importance analysis")
            st.markdown("• **Feedback collection**")
            st.markdown("• **Self-improving model**")
            st.markdown("• **Automatic retraining**")
            
            # OpenAI status
            st.markdown("---")
            st.markdown("### 🤖 AI Features")
            
            # OpenAI API Key Input Section
            st.markdown("#### OpenAI API Key Setup")
            
            # Check for environment variable first
            env_openai_key = os.getenv("OPENAI_API_KEY")
            
            # Add session state for user-provided API key
            if 'user_openai_key' not in st.session_state:
                st.session_state.user_openai_key = ""
            
            # User input for API key
            if not env_openai_key:
                st.markdown("**Enter your OpenAI API Key for enhanced explanations:**")
                user_key = st.text_input(
                    "OpenAI API Key",
                    type="password",
                    value=st.session_state.user_openai_key,
                    placeholder="sk-...",
                    help="Your API key is stored securely in this session only"
                )
                
                if user_key != st.session_state.user_openai_key:
                    st.session_state.user_openai_key = user_key
                    # Update the environment variable for this session
                    if user_key:
                        os.environ["OPENAI_API_KEY"] = user_key
                        st.rerun()
                
                if not user_key:
                    st.info("💡 **How to get an OpenAI API Key:**")
                    st.markdown("""
                    1. Go to [platform.openai.com](https://platform.openai.com/)
                    2. Sign up or log in
                    3. Navigate to API Keys section
                    4. Create a new API key
                    5. Copy and paste it above
                    """)
                    st.warning("⚠️ Enhanced AI explanations require an OpenAI API key")
                else:
                    st.write("✅ OpenAI API Key entered!")
            else:
                st.write("✅ OpenAI API Key found in environment")
            
            # Check OpenAI status
            current_openai_key = st.session_state.user_openai_key or env_openai_key
            if current_openai_key:
                # Show partial key for confirmation (only if it looks valid)
                if len(current_openai_key) > 20 and current_openai_key.startswith('sk-'):
                    masked_key = f"{current_openai_key[:10]}...{current_openai_key[-4:]}"
                    st.caption(f"🔑 Key: {masked_key}")
                
                # Check if explainer has OpenAI client
                explainer = st.session_state.get('explainer')
                if explainer and hasattr(explainer, 'openai_client') and explainer.openai_client:
                    st.write("✅ Enhanced Explanations Ready")
                else:
                    st.write("⚠️ Explainer needs refresh - restart app if issues persist")
                    
            else:
                st.write("❌ No OpenAI API Key")
                st.caption("Enhanced AI explanations unavailable")
            
            # Feedback system status
            st.markdown("---")
            st.markdown("### 🎯 Self-Learning System")
            feedback_system = st.session_state.get('feedback_system')
            if feedback_system:
                st.write("✅ Feedback Learning Active")
                
                # Show quick stats
                feedback_stats = st.session_state.get('feedback_stats')
                if feedback_stats:
                    total = feedback_stats.get('total_count', 0)
                    approval_rate = feedback_stats.get('approval_rate', 0) * 100
                    st.caption(f"📊 {total} feedback entries")
                    st.caption(f"👍 {approval_rate:.0f}% approval rate")
            else:
                st.write("⚠️ Feedback System Not Ready")
        
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
        
        if selected_display and st.button("🔍 Get Recommendations", type="primary"):
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
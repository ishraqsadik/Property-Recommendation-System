import pandas as pd
import numpy as np
import shap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from typing import Optional, Dict, List, Tuple
import warnings
import os
warnings.filterwarnings('ignore')

# OpenAI integration
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    # Note: Warning will be shown in UI when explainer is initialized

class PropertyExplainer:
    """
    Property recommendation explainability using SHAP values.
    Based on the explainability code from the Jupyter notebook.
    """
    
    def __init__(self, model, feature_names: Optional[List[str]] = None):
        """
        Initialize the explainer.
        
        Args:
            model: Trained ML model (XGBoost or sklearn pipeline)
            feature_names: List of feature names for interpretation
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.preprocessor = None
        self.openai_client = None
        self._initialize_explainer()
        self._initialize_openai()
    
    def _initialize_explainer(self):
        """Initialize SHAP explainer based on model type"""
        try:
            # Handle PropertyRecommendationModel
            if hasattr(self.model, 'pipeline'):
                # This is a PropertyRecommendationModel with a pipeline
                pipeline = self.model.pipeline
                self.preprocessor = pipeline.named_steps.get('prep')
                xgb_model = pipeline.named_steps.get('xgb')
                
                if xgb_model is not None:
                    self.explainer = shap.TreeExplainer(xgb_model)
                    if self.preprocessor and hasattr(self.preprocessor, 'get_feature_names_out'):
                        self.feature_names = self.preprocessor.get_feature_names_out()
                    print("✅ SHAP explainer initialized successfully for PropertyRecommendationModel")
                else:
                    raise ValueError("Could not find XGBoost model in PropertyRecommendationModel pipeline")
            
            elif hasattr(self.model, 'named_steps'):
                # This is a sklearn Pipeline directly
                self.preprocessor = self.model.named_steps.get('prep') or self.model.named_steps.get('preprocessor')
                xgb_model = self.model.named_steps.get('xgb') or self.model.named_steps.get('model')
                
                if xgb_model is not None:
                    self.explainer = shap.TreeExplainer(xgb_model)
                    if self.preprocessor and hasattr(self.preprocessor, 'get_feature_names_out'):
                        self.feature_names = self.preprocessor.get_feature_names_out()
                    print("✅ SHAP explainer initialized successfully for sklearn Pipeline")
                else:
                    raise ValueError("Could not find XGBoost model in pipeline")
            
            elif hasattr(self.model, 'predict_proba'):
                # This might be a direct XGBoost model
                self.explainer = shap.TreeExplainer(self.model)
                print("✅ SHAP explainer initialized successfully for direct model")
            
            else:
                raise ValueError(f"Unsupported model type: {type(self.model)}. Expected PropertyRecommendationModel, sklearn Pipeline, or XGBoost model.")
            
        except Exception as e:
            print(f"❌ Error initializing SHAP explainer: {str(e)}")
            print(f"Model type: {type(self.model)}")
            if hasattr(self.model, '__dict__'):
                print(f"Model attributes: {list(self.model.__dict__.keys())}")
            self.explainer = None
    
    def _initialize_openai(self):
        """Initialize OpenAI client if API key is available"""
        if not OPENAI_AVAILABLE:
            print("❌ OpenAI package not installed. Enhanced explanations unavailable.")
            return
            
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                self.openai_client = OpenAI(api_key=api_key)
                print("✅ OpenAI client initialized successfully - Enhanced explanations ready!")
            except Exception as e:
                print(f"❌ Error initializing OpenAI client: {str(e)}")
                self.openai_client = None
        else:
            print("❌ No OpenAI API key found. Enhanced explanations unavailable.")
    
    def explain_property_recommendation(self, 
                                        property_data: pd.DataFrame, 
                                        property_index: int = 0,
                                        top_features: int = 10) -> Optional[pd.DataFrame]:
        """
        Generate SHAP explanation for a specific property recommendation.
        
        Args:
            property_data: DataFrame containing property features
            property_index: Index of the property to explain
            top_features: Number of top features to include in explanation
            
        Returns:
            DataFrame with feature importances and SHAP values
        """
        if self.explainer is None:
            st.error("SHAP explainer not initialized")
            return None
        
        try:
            # Select the property
            if property_index >= len(property_data):
                st.error(f"Property index {property_index} out of range")
                return None
            
            property_row = property_data.iloc[[property_index]]
            
            # Prepare features for SHAP analysis
            X_for_shap = self._prepare_features_for_shap(property_row)
            if X_for_shap is None:
                return None
            
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(X_for_shap)
            
            # Create explanation DataFrame
            explanation_df = self._create_explanation_dataframe(
                shap_values[0], 
                property_row, 
                top_features
            )
            
            return explanation_df
            
        except Exception as e:
            st.error(f"Error generating explanation: {str(e)}")
            return None
    
    def _prepare_features_for_shap(self, property_data: pd.DataFrame) -> Optional[np.ndarray]:
        """Prepare property features for SHAP analysis"""
        try:
            if self.preprocessor is not None:
                # Use preprocessor from pipeline
                X_transformed = self.preprocessor.transform(property_data)
            else:
                # Handle features manually - use the model's expected features
                if hasattr(self.model, 'features'):
                    feature_columns = self.model.features
                else:
                    feature_columns = ['gla', 'bedrooms', 'age', 'structure_type', 'stories', 'bathrooms', 'room_count']
                
                available_columns = [col for col in feature_columns if col in property_data.columns]
                
                if len(available_columns) < 3:
                    st.warning("Insufficient features for SHAP analysis")
                    return None
                
                X = property_data[available_columns].copy()
                
                # Handle missing values
                for col in X.columns:
                    if X[col].dtype == 'object':
                        X[col] = X[col].fillna('Unknown')
                    else:
                        X[col] = X[col].fillna(X[col].median())
                
                X_transformed = X.values
            
            return X_transformed
            
        except Exception as e:
            st.error(f"Error preparing features: {str(e)}")
            print(f"Property data columns: {list(property_data.columns)}")
            if hasattr(self.model, 'features'):
                print(f"Model expected features: {self.model.features}")
            return None
    
    def _create_explanation_dataframe(self, 
                                      shap_values: np.ndarray, 
                                      property_data: pd.DataFrame,
                                      top_features: int) -> pd.DataFrame:
        """Create a DataFrame with SHAP explanations"""
        # Determine feature names
        if self.feature_names is not None and len(self.feature_names) == len(shap_values):
            feature_names = self.feature_names
        else:
            # Fallback to generic names
            feature_names = [f"feature_{i}" for i in range(len(shap_values))]
        
        # Get actual feature values for context
        feature_values = self._get_feature_values(property_data, feature_names)
        
        # Create explanation DataFrame
        explanation_df = pd.DataFrame({
            'feature': feature_names,
            'shap_value': shap_values,
            'abs_shap': np.abs(shap_values),
            'feature_value': feature_values
        }).sort_values('abs_shap', ascending=False).head(top_features)
        
        return explanation_df.reset_index(drop=True)
    
    def _get_feature_values(self, property_data: pd.DataFrame, feature_names: List[str]) -> List:
        """Extract feature values for display"""
        feature_values = []
        
        for feature_name in feature_names:
            try:
                # Handle encoded categorical features
                if 'cat__' in feature_name:
                    # Extract original column and value
                    parts = feature_name.split('__')
                    if len(parts) >= 2:
                        col_name = parts[1].split('_')[0]
                        if col_name in property_data.columns:
                            feature_values.append(str(property_data[col_name].iloc[0]))
                        else:
                            feature_values.append("Unknown")
                    else:
                        feature_values.append("Unknown")
                
                elif 'num__' in feature_name:
                    # Extract numerical feature
                    col_name = feature_name.replace('num__', '')
                    if col_name in property_data.columns:
                        feature_values.append(property_data[col_name].iloc[0])
                    else:
                        feature_values.append("Unknown")
                
                else:
                    # Direct feature name
                    if feature_name in property_data.columns:
                        feature_values.append(property_data[feature_name].iloc[0])
                    else:
                        feature_values.append("Unknown")
                        
            except:
                feature_values.append("Unknown")
        
        return feature_values
    
    def create_shap_waterfall_plot(self, explanation_df: pd.DataFrame, title: str = "Feature Impact") -> go.Figure:
        """Create a waterfall plot showing SHAP values"""
        # Sort by SHAP value for better visualization
        sorted_df = explanation_df.sort_values('shap_value', ascending=True)
        
        # Create colors based on positive/negative impact
        colors = ['#d62728' if val < 0 else '#2ca02c' for val in sorted_df['shap_value']]
        
        # Create the plot
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=sorted_df['shap_value'],
            y=sorted_df['feature'],
            orientation='h',
            marker_color=colors,
            text=[f"{val:+.3f}" for val in sorted_df['shap_value']],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>SHAP Value: %{x:.3f}<br>Feature Value: %{customdata}<extra></extra>',
            customdata=sorted_df['feature_value']
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="SHAP Value (Impact on Recommendation)",
            yaxis_title="Features",
            height=max(400, len(sorted_df) * 30),
            showlegend=False,
            template="plotly_white",
            margin=dict(l=150, r=50, t=50, b=50)
        )
        
        # Add vertical line at x=0
        fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="black")
        
        return fig
    
    def create_shap_summary_plot(self, shap_values_multiple: np.ndarray, features_df: pd.DataFrame) -> go.Figure:
        """Create a summary plot for multiple properties"""
        try:
            # Calculate mean absolute SHAP values
            mean_abs_shap = np.mean(np.abs(shap_values_multiple), axis=0)
            
            # Get feature names
            if self.feature_names and len(self.feature_names) == len(mean_abs_shap):
                feature_names = self.feature_names
            else:
                feature_names = [f"feature_{i}" for i in range(len(mean_abs_shap))]
            
            # Create summary DataFrame
            summary_df = pd.DataFrame({
                'feature': feature_names,
                'mean_abs_shap': mean_abs_shap
            }).sort_values('mean_abs_shap', ascending=True)
            
            # Create bar plot
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=summary_df['mean_abs_shap'],
                y=summary_df['feature'],
                orientation='h',
                marker_color='#1f77b4',
                text=[f"{val:.3f}" for val in summary_df['mean_abs_shap']],
                textposition='outside'
            ))
            
            fig.update_layout(
                title="Feature Importance Summary (Mean |SHAP|)",
                xaxis_title="Mean Absolute SHAP Value",
                yaxis_title="Features",
                height=max(400, len(summary_df) * 25),
                showlegend=False,
                template="plotly_white"
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating summary plot: {str(e)}")
            return go.Figure()
    
    def generate_openai_explanation(self, explanation_df: pd.DataFrame, property_address: str = "") -> Optional[str]:
        """
        Generate natural language explanation using OpenAI GPT.
        Based on the LLM explanation function from the notebook.
        """
        if not self.openai_client or explanation_df.empty:
            return None
        
        try:
            # Prepare feature bullets for prompt - matching notebook format
            bullets = "\n".join(
                f"- {row['feature']}: {row['feature_value']} (impact {row['shap_value']:+.3f})"
                for _, row in explanation_df.head(5).iterrows()
            )
            
            # Create prompt exactly like in notebook
            prompt = f"""
You are an expert real-estate appraisal assistant. Below are key features and their influence on this property's suitability as a comparable:

{bullets}

Write a concise, 2–3 sentence explanation of why this property is a strong or weak comparable for the subject property.
"""
            
            # Call OpenAI API - using gpt-4 instead of gpt-4.1 for compatibility
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Generate appraisal-style explanations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return None
    
    def generate_natural_language_explanation(self, explanation_df: pd.DataFrame, property_address: str = "") -> str:
        """
        Generate human-readable explanation using OpenAI only.
        No fallback to rule-based explanations.
        """
        if explanation_df.empty:
            return "No explanation available for this property."
        
        # Use OpenAI only
        openai_explanation = self.generate_openai_explanation(explanation_df, property_address)
        if openai_explanation:
            return f"🤖 **AI-Powered Explanation:**\n\n{openai_explanation}"
        else:
            return "❌ **OpenAI explanation unavailable.** Please check your API key and internet connection."

# Integration functions for the Streamlit app
def create_explainer_for_streamlit(model) -> Optional[PropertyExplainer]:
    """Create and return a PropertyExplainer instance for Streamlit app"""
    try:
        explainer = PropertyExplainer(model)
        if explainer.explainer is None:
            st.error("Failed to initialize SHAP explainer")
            return None
        return explainer
    except Exception as e:
        st.error(f"Error creating explainer: {str(e)}")
        return None

def display_property_explanation(explainer: PropertyExplainer, 
                                 property_data: pd.DataFrame, 
                                 property_index: int,
                                 property_address: str = "") -> None:
    """Display SHAP explanation in Streamlit"""
    if explainer is None:
        st.error("Explainer not available")
        return
    
    explanation_df = explainer.explain_property_recommendation(property_data, property_index)
    
    if explanation_df is not None:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # SHAP waterfall plot
            fig = explainer.create_shap_waterfall_plot(explanation_df)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Feature details table
            st.markdown("#### 📊 Feature Details")
            display_df = explanation_df[['feature', 'feature_value', 'shap_value']].copy()
            display_df['shap_value'] = display_df['shap_value'].round(3)
            display_df.columns = ['Feature', 'Value', 'Impact']
            st.dataframe(display_df, use_container_width=True)
        
        # Natural language explanation (with OpenAI if available)
        st.markdown("#### 💬 Explanation")
        explanation_text = explainer.generate_natural_language_explanation(
            explanation_df, property_address
        )
        st.markdown(explanation_text)
    
    else:
        st.error("Could not generate explanation for this property") 
# Property Recommendation System

🏠 **AI-Powered Property Recommendations with Explainable Insights**

An intelligent property recommendation system designed for real estate appraisers, featuring self-learning capabilities through user feedback and comprehensive AI explainability.

**🚀 Try the Live App: [https://compfinder.streamlit.app/](https://compfinder.streamlit.app/)**

## ✨ Key Features

### 🎯 Core Functionality
- **Smart Property Matching**: AI-powered similarity scoring using XGBoost
- **True Comparable Identification**: Highlights actual comparables used by appraisers
- **Interactive Web Interface**: Clean, user-friendly Streamlit interface
- **Self-Learning System**: Model improves automatically based on user feedback

### 🔍 Explainable AI
- **SHAP Integration**: Visual explanations for each recommendation
- **OpenAI-Enhanced Explanations**: Natural language insights (when API key provided)
- **Feature Importance Analysis**: Understand which property characteristics matter most
- **Interactive Visualizations**: Waterfall plots and feature impact charts

### 📈 Learning & Analytics
- **Feedback Collection**: Rate recommendations and guide model improvement
- **Automatic Retraining**: Model updates when sufficient feedback is collected
- **Progress Tracking**: Monitor approval rates and learning metrics
- **Data Export**: CSV export for feedback analysis

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **ML Pipeline**: XGBoost with scikit-learn preprocessing
- **Explainability**: SHAP for interpretable AI
- **Enhanced AI**: OpenAI GPT-4 for natural language explanations
- **Visualization**: Plotly for interactive charts
- **Data Processing**: Pandas for efficient data manipulation

## 🚀 Getting Started

### Option 1: Use the Deployed App (Recommended)

**Visit: [https://compfinder.streamlit.app/](https://compfinder.streamlit.app/)**

- ✅ No installation required
- ✅ Always up-to-date
- ✅ Ready to use immediately
- 🔑 Optional: Add your OpenAI API key in the sidebar for enhanced explanations

### Option 2: Run Locally

#### Prerequisites
- Python 3.8+
- OpenAI API Key (optional, for enhanced explanations)

#### Installation

```bash
# Clone the repository
git clone https://github.com/ishraqsadik/Property-Recommendation-System.git
cd Property-Recommendation-System

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run streamlit_app.py
```

#### Configuration (Optional)

```bash
# Set OpenAI API key for enhanced explanations
export OPENAI_API_KEY=your_api_key_here

# Or create a .env file
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

## 📦 Project Structure

```
property-recommendation-system/
├── streamlit_app.py                # Main Streamlit application
├── app/
│   ├── explainability.py          # SHAP & OpenAI explainability
│   ├── ml/                         # Machine learning components
│   │   ├── model.py               # Property recommendation model
│   │   ├── feature_engineering.py # Feature extraction & processing
│   │   ├── data_processing.py     # Data loading & cleaning
│   │   ├── feedback_learning.py   # Self-learning feedback system
│   │   └── utils.py               # Utility functions
│   └── api/                        # API models (for data structures)
├── data/                          # Data storage
│   ├── models/                    # Trained model files
│   ├── feedback/                  # User feedback database
│   └── processed/                 # Processed training data
├── appraisals_dataset.json        # Property dataset
└── requirements.txt               # Python dependencies
```

## 📋 How to Use

### 1. Access the Application
**Visit: [https://compfinder.streamlit.app/](https://compfinder.streamlit.app/)**
- System automatically initializes when you open the app
- Model, data, and AI components load seamlessly in the background
- No manual setup required!

### 2. Select a Test Property
- Choose from real properties in the appraisal dataset
- Properties are from the model's test set (unseen during training)
- Click "🔍 Get Recommendations" to analyze

### 3. Review Recommendations
- View top property matches with similarity scores
- Green indicators show "True Comparables" (actual appraisal comps)
- Detailed property information and metrics displayed

### 4. Understand the AI Decisions
- Click "🔍 Explain Why This Property Was Recommended"
- Interactive SHAP waterfall plots show feature contributions
- Natural language explanations (with OpenAI integration)
- Feature importance dashboard across all recommendations

### 5. Provide Feedback
- Rate the top 3 recommendations using checkboxes
- Add optional comments for additional context
- Submit feedback to improve future recommendations

### 6. Monitor Learning Progress
- View feedback statistics and approval rates
- Track processed vs pending feedback
- Manual retraining controls available
- Export feedback data for analysis

## 🔧 Advanced Configuration

### OpenAI API Key Setup

For enhanced AI explanations, you can provide your OpenAI API key:

**In the Deployed App:**
- Visit [https://compfinder.streamlit.app/](https://compfinder.streamlit.app/)
- Look for "OpenAI API Key Setup" in the sidebar
- Enter your API key securely (stored only in your session)

**For Local Development:**

```bash
# Environment Variable
export OPENAI_API_KEY=your_api_key_here

# Or .env file  
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

### Custom Model Training

The system can retrain automatically or manually:

**Via Web Interface:**
- Navigate to "Manual Model Retraining" section in the app
- Click "🚀 Retrain Model Now"

**Local Development:**
```bash
# Force retrain with existing feedback
python force_retrain.py
```

## 🔬 Model & Data Science

### Train/Test Split Validation
- System maintains strict separation between training and test data
- Only test properties are available for recommendations
- Prevents data leakage and ensures valid model evaluation
- Test property IDs preserved during model retraining

### Self-Learning System
- Collects user feedback on recommendation quality
- Automatically triggers retraining when threshold reached
- Incorporates both positive and negative feedback
- Preserves original training data while adding feedback examples

### Feature Engineering
Key features used for property matching:
- **Physical**: GLA, bedrooms, bathrooms, age, stories
- **Location**: City, distance metrics, neighborhood characteristics  
- **Property Type**: Structure type, heating/cooling systems
- **Market**: Recent sale prices, days on market

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

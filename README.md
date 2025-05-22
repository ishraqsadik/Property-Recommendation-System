# Property Recommendation System

A property recommendation system with human feedback learning capabilities, designed for real estate appraisers to find comparable properties.

## Features

- Property recommendation based on similarity metrics
- Human feedback learning to improve recommendations over time
- API for integration with other systems
- Exportable feedback data for analysis

## Project Structure

```
property-recommendation-system/
├── app/                    # Main application code
│   ├── api/                # API components
│   │   ├── models/         # Pydantic models for API
│   │   └── routes/         # API endpoints
│   └── ml/                 # Machine learning components
│       ├── data_processing.py      # Data cleaning and processing
│       ├── feature_engineering.py  # Feature extraction
│       ├── model.py               # Recommendation model
│       ├── feedback_learning.py   # Feedback learning system
│       └── utils.py               # Utility functions
├── data/                   # Data storage
│   ├── raw/                # Raw data files
│   ├── processed/          # Processed data files
│   ├── models/             # Saved models
│   └── feedback/           # Feedback database
```

## Requirements

- Python 3.8+
- FastAPI
- Pandas
- XGBoost
- Scikit-learn
- Uvicorn (for running the API server)

## Installation

1. Clone the repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Application

Start the API server:

```bash
python -m app.main
```

The API will be available at http://localhost:8000

## API Documentation

Once the server is running, you can access the API documentation at:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Train/Test Split Validation

The system ensures that only properties from the test split are used for recommendations:

- During model training, the system stores the IDs of all test properties
- When recommendations are requested, the model validates that only properties from the test set are considered
- This validation prevents "data leakage" where training data might be used for recommendations
- When the model is retrained with feedback data, the system preserves the original test property IDs

This approach ensures that the frontend only receives recommendations for properties that weren't used to train the original model, maintaining a clean separation between training and testing data.

## Workflow

1. Submit a subject property for recommendation
2. Receive comparable property recommendations
3. Provide feedback on the recommendations
4. The system learns from feedback and improves future recommendations 
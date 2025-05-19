# Comparable Property Recommendation System

This repository contains a **Comparable Property Recommendation System** that:

* **Loads** and cleans a dataset of real estate appraisals (subjects, comps, and neighborhood properties).
* **Trains** an XGBoost-based ranking model to predict the top 3 comps for a subject property.
* Incorporates **explainability** via SHAP and optional LLM-generated prose explanations.

## 🚀 Features

* **Data Ingestion & Cleaning**: Parses nested JSON into pandas DataFrames, normalizes addresses, handles missing values.
* **Machine Learning Pipeline**: ColumnTransformer + XGBoost ranking model, with group-wise train/test split and early stopping.
* **Explainability**:

  * **SHAP** to extract top feature drivers per recommended comp.
  * **LLM Integration**: uses GPT (gpt-4.1) to convert bullet-point SHAP insights into natural language.

## 💡 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your_org>/<your_repo>.git
cd <your_repo>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` should include at least:

```
pandas
scikit-learn
xgboost
shap
openai          # for LLM integration (optional)
python-dotenv   # if using .env files
```

### 3. Configure API keys

* **OpenAI** (for LLM explanations):

  1. Add your key to a `.env` file (gitignored):

     ```
     OPENAI_API_KEY=sk-...
     ```
  2. Or set via Colab / environment:

     ```python
     import os
     os.environ['OPENAI_API_KEY'] = 'sk-...'
     ```

### 4. Prepare the dataset

Place your appraisal JSON (`appraisals.json`) at the repository root. It should follow the structure:

```json
{
  "appraisals": [
     { "orderID": ..., "subject": {...}, "comps": [...], "properties": [...] },
     ...
  ]
}
```





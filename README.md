# 🏠 California House Price Prediction

A machine learning project that predicts California housing prices using a Random Forest model, complete with a Streamlit web app.

## 🌐 Web App
Run the app locally to predict house prices instantly by entering house details.

## 📁 Project Structure
- `data/` - California housing dataset
- `notebooks/` - step by step exploration notebooks
- `main.py` - training + inference pipeline
- `app.py` - Streamlit web app

## 🚀 How to Run

**1. Install dependencies:**
pip install -r requirements.txt

**2. Train the model:**
python main.py

**3. Run the web app:**
streamlit run app.py

> Note: Run `main.py` first — it generates `model.pkl` and `pipeline.pkl` which the app needs.

## 📚 Topics Covered
- Exploratory data analysis
- Data visualization
- Handling missing values
- Feature scaling & categorical encoding
- Sklearn Pipelines
- Random Forest Regression
- Model saving with joblib
- Streamlit web app

## 🛠️ Libraries Used
pandas, numpy, scikit-learn, streamlit, joblib

## 📖 Note
Learning project based on Hands-On Machine Learning by Aurélien Géron.
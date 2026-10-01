# crop-prediction-flask-ML
A full-stack crop recommendation web application that predicts suitable crops using soil nutrients, weather conditions, and a scikit-learn Random Forest model. Built with Python, Flask, SQLite, HTML, CSS, and JavaScript.
# CropWise

A Flask crop-recommendation web app using Python, Pandas/NumPy, scikit-learn, Joblib and SQLite.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Visit `http://127.0.0.1:5000`. The model is generated automatically on first prediction and stored in `models/crop_model.joblib`; predictions are saved to `data/predictions.db`.

## Deployment

The included `Procfile` supports Render or Railway. Set the start command to `gunicorn app:app`. For persistent prediction history in production, use a managed database because local SQLite storage is often ephemeral.

## Important

This demo trains from representative crop profiles generated in `model.py`. Replace `create_training_data()` with validated regional soil, weather, yield, and crop data before using it to make real farming decisions.

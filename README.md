Crypto Predictor 💰
This project uses MLflow to track models and Streamlit to run the app interface.
1. Start the MLflow Server
MLflow UI lets you view and manage experiments. Run:
 *"C:\Program Files\Python310\python.exe" -m mlflow ui --backend-store-uri <your-uri>
Replace <your-uri> with your MLflow backend path or database.

2. Launch the App
Run the Streamlit app to interact with the predictor:
 *streamlit run app.py

3. If Streamlit is not installed or not found
Run Streamlit via Python module:
 *python -m streamlit run app.py


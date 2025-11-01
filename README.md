# Hybrid-Renewable-Energy-Prediction-Optimization
AI-BASED HYBRID RENEWABLE ENERGY PREDICTION USING ML
⚡ Project Overview

This project focuses on predicting the energy output from hybrid renewable sources (solar + wind) using machine learning models trained on environmental and weather data.
It uses simulated or real-world datasets (temperature, humidity, solar irradiance, wind speed) to predict power generation.
The model helps improve energy management, forecasting accuracy, and grid reliability by combining solar and wind prediction in a single system.

📁 Repository Structure
Hybrid-Energy-Prediction/
│
├── data/
│   └── hybrid_energy_dataset.csv           # Generated or collected dataset
│
├── notebooks/
│   └── 01_data_preprocessing.ipynb         # Data cleaning & feature scaling
│   └── 02_model_training.ipynb             # ML model training & evaluation
│   └── 03_prediction_visualization.ipynb   # Graphs and result visualization
│
├── src/
│   ├── data_generator.py                   # Code to generate synthetic dataset
│   ├── model_train.py                      # Train regression/ML model
│   ├── predict.py                          # Predict output for new data
│   └── utils.py                            # Helper functions
│
├── results/
│   ├── model_performance.csv               # Metrics summary
│   └── prediction_plot.png                 # Visualization of results
│
├── requirements.txt                        # Python dependencies
├── README.md                               # Project overview & setup
└── app.py                                  # Optional Streamlit or Flask dashboard

🔄 Workflow

Data Generation / Collection:
Use Python or IoT sensors to collect weather parameters — temperature, humidity, solar irradiance, and wind speed.

Data Preprocessing:
Handle missing values, normalize data, and create derived features like “hour of day” or “daylight duration.”

Model Training:
Train ML models (e.g., Linear Regression, Random Forest, or LSTM) to predict solar, wind, and total energy output.

Model Evaluation:
Compute metrics such as MAE, RMSE, and R² Score to assess performance.

Prediction & Visualization:
Plot predicted vs. actual power output and visualize energy patterns across time.

Deployment (Optional):
Use Streamlit or Flask for an interactive dashboard showing real-time predictions.

📊 Key Results

Prediction Accuracy: 90–95% (depending on model & tuning).

RMSE: Around 0.05–0.1 kW for total output.

Visualization: Clear time-series comparison of predicted vs. actual energy generation.

Outcome: Reliable short-term forecasting of hybrid renewable energy production, suitable for smart-grid integration.

▶️ How to Run
# 1. Clone the repository
git clone https://github.com/yourusername/Hybrid-Energy-Prediction.git
cd Hybrid-Energy-Prediction

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate dataset
python src/data_generator.py

# 5. Train the model
python src/model_train.py

# 6. Make predictions
python src/predict.py

# 7. (Optional) Run dashboard
streamlit run app.py

🧩 Requirements (requirements.txt)
pandas
numpy
scikit-learn
matplotlib
seaborn
streamlit
tensorflow

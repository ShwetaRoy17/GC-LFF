# 🌍 GC-LFF: GAN-CNN-LSTM Federated Framework for AQI Forecasting

A comprehensive Python-based framework for **Air Quality Index (AQI)** forecasting in Indian cities, using a hybrid deep learning model (**GAN + CNN + LSTM**) integrated with **Federated Learning** for privacy-preserving, distributed model training.

---

## 📂 Dataset

We use the publicly available dataset from Kaggle:

> 🔗 [Air Quality Data in India – Kaggle](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)

The dataset includes pollutant readings from multiple Indian cities, including PM2.5, PM10, CO, NO2, SO2, O3, and other sensor data, along with timestamps.

---

## 🧠 Notebook Structure

The Python Notebook is organized into the following sections:

### 1. 📦 Import
Load required libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `tensorflow`, `sklearn`, etc.

### 2. 📥 Data Loading
Load raw AQI and pollutant data CSVs into pandas DataFrames.

### 3. 🧹 Data Preprocessing
- Time alignment
- Data cleaning
- Categorical encoding
- Feature normalization

### 4. ❓ Missing Data
- Visualize missing data
- Prepare for imputation using advanced methods

### 5. ✂️ Splitting Data for Hyperparameter Tuning
- Use `train_test_split` to split data for validation
- Ensures unbiased model evaluation

### 6. 📊 Evaluation Metric Function
Define MAE, RMSE, and MSE for performance measurement.

### 7. 🤖 GAIN – Generative Imputation
- Impute missing values using GAIN (Generative Adversarial Imputation Networks)
- Compare with KNN, Mean, and MICE

### 8. 🛠️ Hyperparameter Tuning
- Tune CNN-LSTM layers
- Kernel sizes, LSTM units, dropout, learning rate

### 9. 🏋️ Final Training on Full Dataset
- Train optimized model on full imputed dataset

### 10. 📈 Plot
Visualize training/validation loss and predictions over time.

### 11. 📉 Plot of Missing Data
Heatmap of missing sensor readings across time and cities.

### 12. 🔄 Comparison with Traditional Methods
Compare GAIN-imputed model with Mean, KNN, and MICE:
- Evaluate statistical significance (paired t-test)

---

## 🤝 Federated Learning Section

### 13. 🧱 Federated Learning Pre-Req
- Define federated learning workflow using `FedAvg`
- Load base CNN-LSTM model for each client

### 14. 🏙️ Preprocessing City-wise Data
- Slice AQI data for each city
- Prepare local train/test splits

### 15. 📉 FL: Error Evaluation
- Evaluate federated model across all clients
- Compute average loss per round

### 16. 🆕 NEW FL CODE
- Updated implementation using efficient client model sync
- Parallel local training per round

### 17. 📊 FL: Error Plot
- Visualize performance (MSE/RMSE) per city and per round

---

## 🧪 Centralized vs Federated Learning

### 18. 🏢 Centralized Learning
- Train the same CNN-LSTM on the entire dataset centrally

### 19. ⚖️ CL vs FL City-wise Prediction Graph
- Compare prediction error of centralized vs federated model for each city
- Highlight improvements in data privacy and generalization

---

## 📎 Requirements

Install required dependencies via pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow

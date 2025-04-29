# credit-card-fraud-detection
A machine learning project to detect credit card fraud using LightGBM and XGBoost, with performance evaluation, hyperparameter tuning, and SHAP-based model explainability.


# 💳 Credit Card Fraud Detection using Machine Learning

This AI-driven project focuses on detecting fraudulent credit card transactions using advanced machine learning techniques such as **LightGBM** and **XGBoost**. The dataset includes transactional, user, and merchant-level information, and the models are trained to distinguish between legitimate and fraudulent activity.

> ⚠️ **Note**: This project is AI-focused only — no frontend or backend has been developed. It's built purely for ML modeling, evaluation, and interpretability.

---

## 🧠 Project Objective

- Build a machine learning model that accurately classifies transactions as **fraudulent** or **legitimate**.
- Minimize **false positives** while maximizing **fraud detection**.
- Understand model predictions using **SHAP (SHapley Additive exPlanations)**.


## 🔍 Dataset Summary

The dataset includes transaction details such as:

- `trans_date_trans_time`, `amt`, `merchant`, `category`
- User details like `gender`, `job`, `city_pop`, `dob`
- Merchant geolocation: `merch_lat`, `merch_long`
- Target variable: `is_fraud` (0 = legitimate, 1 = fraud)

---

## 🛠️ Notebook Flow

1. **Import Libraries**  
2. **Reading the Dataset**  
3. **Data Preprocessing**  
   - Feature Engineering (e.g., time, distance, age)
   - Label encoding & null handling  
4. **Model Development**  
   - XGBoost & LightGBM with class imbalance techniques  
5. **Visualization**  
   - Feature importance, SHAP plots  
6. **Model Evaluation**  
   - Confusion Matrix, ROC-AUC, Precision/Recall  
7. **Model Improvement**  
   - Hyperparameter Tuning using GridSearchCV  

---

## ⚙️ Model Performance

### ✅ Best Model: LightGBM (After Grid Search)

- **Accuracy**: 0.9989  
- **ROC-AUC Score**: 0.9944  
- **False Positives**: 716  
- **False Negatives**: 14  
- **Precision (fraud)**: 0.94  
- **Recall (fraud)**: 0.76  

### 🔎 SHAP Explainability
Used `shap.Explainer()` for:
- False Positives
- False Negatives
- Top feature drivers for fraud detection

---

## 📈 Evaluation Visuals

- 📊 Confusion Matrix
- 📉 ROC Curve
- 🔀 Precision-Recall Curve
- 🎯 SHAP Force & Summary plots

---

## 🧪 Tech Stack

- Python 3.x
- Pandas, NumPy
- Scikit-learn
- LightGBM, XGBoost
- SHAP
- Matplotlib, Seaborn

install individually:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost shap
```

---

## 📜 License

MIT License

---

## 🙋‍♂️ Contribution

Pull requests are welcome. If you find bugs or have improvements, feel free to open an issue.

---

## 📢 Contact

For queries, feel free to reach out via GitHub Issues or Discussions.
Email : shivaprasad20005@gmail.com

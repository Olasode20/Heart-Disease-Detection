# 🩺 Heart Disease Prediction Model

This project aims to predict the likelihood of heart disease in patients using various machine learning models.  
It involves data preprocessing, exploratory data analysis (EDA), model training, evaluation, and optimization through hyperparameter tuning.

---

## 📊 Dataset

**Source:** [Heart Disease Train-Test Dataset - Kaggle](https://www.kaggle.com/)  
The dataset contains key medical attributes such as age, cholesterol levels, resting blood pressure, maximum heart rate, and more.

---

## ⚙️ Models Trained

| Model               | AUC Score                      |
| ------------------- | ------------------------------ |
| Logistic Regression | 0.90                           |
| Decision Tree       | 0.99                           |
| Random Forest       | **1.00**                       |
| Gradient Boosting   | 0.98 → **0.99 (after tuning)** |
| K-Nearest Neighbors | 0.95                           |

---

## 🧠 Approach

1. **Data Preprocessing**

   - Handled missing values
   - Encoded categorical features using `pd.get_dummies()`
   - Scaled numerical features for distance-based models (e.g., KNN)

2. **Exploratory Data Analysis**

   - Visualized feature relationships and target distribution using `seaborn`
   - Created **pairplots**, **heatmaps**,**countplot** and correlation matrices to understand patterns

3. **Model Training**

   - Split data into training and test sets using `train_test_split`
   - Trained multiple models using `scikit-learn`

4. **Model Evaluation**

   - Evaluated using **ROC-AUC** as the main performance metric
   - Compared models using AUC and ROC curves

5. **Hyperparameter Tuning**
   - Performed fine-tuning on the **Gradient Boosting Classifier**
   - Used `GridSearchCV` to identify the best combination of parameters  
     (learning rate, number of estimators, and tree depth)
   - Improved AUC from **0.98 → 0.99**

---

## 📈 Results Visualization

- **ROC Curves:** Compared classifier performance visually
- **Correlation Heatmap:** Showed feature relationships
- **Pairplot:** Displayed class separation among important features

---

🧩 Tech Stack

Python

Pandas

NumPy

Scikit-learn

Matplotlib / Seaborn

💡 Insights

The Random Forest model achieved the highest AUC of 1.0, showing perfect classification for this dataset.

Gradient Boosting improved significantly after hyperparameter tuning, demonstrating the impact of model optimization.

📬 Connect with Me

Muiz Olasode
📧 Email: Olasodemuiz@gmail.com

```

```

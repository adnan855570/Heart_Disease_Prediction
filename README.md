# ❤️ Heart Disease Prediction Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-brightgreen)
![Model](https://img.shields.io/badge/ML%20Models-5-blueviolet)
![Status](https://img.shields.io/badge/Project-Complete-success)

---

## 🧠 Overview

This project builds a heart disease prediction system using various machine learning models. The goal is to compare models, understand which features are most important, and visualize the process of predicting heart disease using real-world patient data.

---

## 📦 Installation

Clone the repo and install dependencies.

```bash
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction
pip install -r requirements.txt
```

Or install dependencies manually:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## 📁 Dataset

The dataset contains anonymized health metrics related to heart health collected from clinical examinations.

### 🔑 Key Features:

- **Age** – Age of the patient  
- **Sex** – Gender (Male/Female)  
- **ChestPainType** – Type of chest pain experienced  
- **RestingBP** – Resting blood pressure (in mm Hg)  
- **Cholesterol** – Serum cholesterol (in mg/dl)  
- **FastingBS** – Fasting blood sugar (> 120 mg/dl) (1 = True, 0 = False)  
- **RestingECG** – Resting electrocardiographic results  
- **MaxHR** – Maximum heart rate achieved  
- **ExerciseAngina** – Exercise-induced angina (1 = Yes, 0 = No)  
- **Oldpeak** – ST depression induced by exercise relative to rest  
- **ST_Slope** – The slope of the peak exercise ST segment  

### 🎯 Target Variable:
- **HeartDisease**:  
  - `0` = No Heart Disease  
  - `1` = Heart Disease Present

### 📚 Source:
[UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)

## 📊 Exploratory Data Analysis (EDA)

### 👥 Age Distribution

This plot shows how age is distributed across the dataset. Most patients fall within the 40–60 age range, which aligns with common risk ages for heart-related conditions.

![Age Distribution](./2.png)

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(df['Age'], bins=30, kde=True, color='blue')
plt.title("Distribution of Age")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()
```

### 🧪 Cholesterol Distribution

This histogram visualizes the distribution of cholesterol levels across patients. It helps identify outliers or skewness in cholesterol data, which is a critical risk factor for heart disease.

![Cholesterol Distribution](./3.png)

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(df['Cholesterol'], bins=30, kde=True, color='red')
plt.title("Distribution of Cholesterol")
plt.xlabel("Cholesterol")
plt.ylabel("Frequency")
plt.show()
```

### 🔥 Correlation Heatmap

The heatmap below shows the **Pearson correlation coefficients** between all numerical features in the dataset. This helps identify:
- Multicollinearity between predictors
- Key relationships with the target variable `HeartDisease`

Notably:
- `ST_Slope` and `ExerciseAngina` show strong correlations with heart disease.
- `Cholesterol` and `RestingBP` have weaker but still relevant associations.

![Correlation Heatmap](./4.png)

```python
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Encode categorical features before computing correlations
for col in ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]:
    df[col] = LabelEncoder().fit_transform(df[col])

# Generate correlation heatmap
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()
```

## ⚙️ Data Preprocessing

Before training machine learning models, the dataset must be cleaned and prepared:

- ✅ Categorical features are **Label Encoded**
- ✅ Data is split into **training and testing sets**
- ✅ Features are **standardized** using `StandardScaler`

```python
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Encode categorical features
for col in ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]:
    df[col] = LabelEncoder().fit_transform(df[col])

# Separate features and target
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Standardize the feature values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## 🤖 Model Training

We trained and evaluated multiple machine learning models to classify whether a patient has heart disease based on clinical features.

### 🧠 Models Used:
- **Decision Tree**
- **Random Forest**
- **K-Nearest Neighbors (KNN)**
- **Logistic Regression**
- **Support Vector Machine (SVM)**

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Initialize ML models
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(probability=True)
}

# Train each model
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
```

## 📈 ROC Curve Comparison

The **ROC (Receiver Operating Characteristic) curve** illustrates the diagnostic ability of each model by plotting the **True Positive Rate vs. False Positive Rate**.

> ✅ A higher Area Under the Curve (AUC) indicates better performance.  
> ✅ Ideal curves hug the top-left corner of the graph.

![ROC Curve](./1.png)

```python
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

for name, model in models.items():
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.2f})")

# Baseline
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.title("ROC Curves for Heart Disease Prediction Models")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()
```

## 🧠 Feature Importance (Random Forest)

The **Random Forest** model allows us to examine which features are most important in predicting heart disease.

> 📌 Higher importance scores indicate stronger influence on the model’s decision-making process.

![Feature Importance](./5.png)

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Get feature importances from the trained Random Forest model
importances = models["Random Forest"].feature_importances_
features = X.columns

# Visualize
sns.barplot(x=importances, y=features)
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()
```

## 🔧 Hyperparameter Tuning

We used **GridSearchCV** to optimize the hyperparameters for both **Random Forest** and **Support Vector Machine (SVM)** models.

> 🔍 Grid Search performs an exhaustive search over a range of parameters to find the best combination based on cross-validation accuracy.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Random Forest hyperparameter grid
params_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20]
}

grid_rf = GridSearchCV(RandomForestClassifier(), params_rf, cv=5)
grid_rf.fit(X_train_scaled, y_train)

# Support Vector Machine hyperparameter grid
params_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}

grid_svm = GridSearchCV(SVC(probability=True), params_svm, cv=5)
grid_svm.fit(X_train_scaled, y_train)

# Print the best parameters
print("Best RF Parameters:", grid_rf.best_params_)
print("Best SVM Parameters:", grid_svm.best_params_)
```

## 🧪 Evaluation Metrics

After training and tuning, we evaluate all models using standard classification metrics:

- **Accuracy** – Overall correctness of the model  
- **Precision** – How many predicted positives are true positives  
- **Recall** – How many actual positives were captured  
- **F1-score** – Harmonic mean of precision and recall  
- **Confusion Matrix** – Breakdown of predictions by class

```python
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Evaluate all models on the test set
for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    print(f"Model: {name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("="*60)
```

## 🪪 License

This project is licensed under the **MIT License**.  
See the [LICENSE](./LICENSE) file for full details.

---

## 🙌 Acknowledgments

Special thanks to the following resources and tools that made this project possible:

- 🎓 [UCI Machine Learning Repository](https://archive.ics.uci.edu/)
- 🧪 [scikit-learn](https://scikit-learn.org/stable/)
- 📊 [Matplotlib](https://matplotlib.org/)
- 📈 [Seaborn](https://seaborn.pydata.org/)

> 🚀 Built with ❤️ using open-source tools to help detect heart disease early through machine learning.




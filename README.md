# Medical-Insurance-ML

This is a Machine Learning project that predicts the charges (price) insurance companies will charge an individual. These charges are based on factors such as age, BMI (Body Mass Index), smoking status, and more. The project involves data cleaning, exploratory data analysis, feature engineering, and machine learning using Python in a Jupyter Notebook environment.

---

### **Key Features**
- **Objective:** Predict insurance charges using machine learning regression models.
- **Dataset:** Obtained from Kaggle (linked below).
- **Models Used:**
  - Linear Regression
  - Ridge and Lasso Regression
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - XGBoost Regressor
  - Support Vector Regressor (SVR)
- **Evaluation Metrics:** 
  - Mean Squared Error (MSE)
  - R² Score
- **Best Performing Model:** Gradient Boosting Regressor.

---

### **Libraries and Tools Used**
- **Python Libraries:**
  - `pandas` and `numpy` for data manipulation
  - `matplotlib` and `seaborn` for data visualization
  - `scikit-learn` for model building and evaluation
  - `xgboost` for advanced gradient boosting techniques
- **Development Environment:**
  - Jupyter Notebook

---

### **Project Workflow**
1. **Data Preprocessing:**
   - Handling missing values
   - Encoding categorical variables
2. **Exploratory Data Analysis:**
   - Visualizing relationships between features and target variable (`charges`)
   - Identifying trends like the impact of smoking or BMI on charges
3. **Model Building:**
   - Training and evaluating multiple machine learning models
   - Hyperparameter tuning for optimization
4. **Model Comparison:**
   - Comparing performance using MSE and R² to select the best model
5. **Conclusion:**
   - Gradient Boosting Regressor provided the lowest error and highest predictive accuracy.

---

### **Dataset**
- **Source:** [Medical Cost Personal Dataset (Kaggle)](https://www.kaggle.com/datasets/mirichoi0218/insurance?resource=download)
- **Files:**
  - `insurance.csv` - Medical insurance dataset containing features and target variable
  - `InsuranceML.ipynb` - Jupyter notebook with the complete code for the project

---

### **Future Improvements**
- Implement feature selection techniques to improve model interpretability.
- Explore deep learning models for prediction.
- Deploy the model using Flask or Streamlit for real-world usability.

---

### **Why This Project is Valuable**
This project demonstrates end-to-end implementation of a regression-based machine learning workflow. It highlights the use of Python libraries, best practices in data analysis, and model tuning. The skills demonstrated align well with real-world tasks in data science and machine learning, making it a strong portfolio addition.


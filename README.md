# HR_Employee_Attrition
This project focuses on building an end-to-end Machine Learning classification system to predict whether an employee is likely to leave an organization. Employee attrition is a major challenge for companies as it increases hiring costs and affects productivity.

 Employee Attrition Prediction using Machine Learning

 Problem Statement
  
 Employee attrition is a critical issue for organizations as it leads to increased recruitment costs, loss of experienced talent, and reduced productivity.  
 The goal of this project is to build a Machine Learning model that can predict employee attrition based on historical HR data and identify the key factors responsible   for employee      turnover.


 Dataset 
 
 The dataset contains employee-related information such as:
 
- Demographics
- Job role and department
- Salary and compensation
- Work experience
- Work-life balance
- Overtime and performance metrics

 Target Variable:

 Attrition` (Yes / No)

 This is a supervised binary classification problem

 Machine Learning Approach

Type of Learning
  Supervised Learning**
  Classification Problem**

Workflow

1. Data Understanding
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Encoding
5. Feature Scaling
6. Train-Test Split
7. Model Training
8. Model Evaluation
9. Business Insights


 Tools & Technologies Used
 
- Python
- Pandas
- NumPy
- Matplotlib & Seaborn
- Scikit-learn
- Jupyter Notebook


  Data Preprocessing Steps
  
- Removed unnecessary or non-informative columns
- Handled categorical variables using Label Encoding / One-Hot Encoding
- Scaled numerical features using StandardScaler
- Split dataset into training and testing sets


 Models Used
 
- Logistic Regression (Baseline Model)
- Random Forest Classifier (Final Model)

 Random Forest was selected due to its better performance and ability to explain feature importance.


 Model Evaluation Metrics
 
The following metrics were used to evaluate model performance:
- Accuracy
- Precision
- Recall
- F1-Score

Recall was prioritized because predicting employees likely to leave is more important than false alarms in HR decision-making.



 Key Insights
 
- Employees working overtime are more likely to leave
- Monthly income significantly impacts attrition
- Job role and years at the company influence employee retention
- Work-life balance plays a crucial role in employee satisfaction



 Business Impact
 
This model can help HR teams:

- Identify high-risk employees
- Design targeted retention strategies
- Reduce attrition-related costs
- Improve employee satisfaction and workforce stability


Future Improvements

- Handle class imbalance using SMOTE
- Hyperparameter tuning with GridSearchCV
- Add Explainable AI (SHAP)
- Deploy the model using Streamlit


Conclusion

This project demonstrates a complete Machine Learning pipeline applied to a real-world HR analytics problem. It highlights how data-driven decision-making can improve employee retention and organizational efficiency.


 Author
Vishal Kumar 81080

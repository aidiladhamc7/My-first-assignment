# Employee Attrition Prediction - Machine Learning Project

## Overview

Welcome to the **Employee Attrition Prediction** project! This project marks my first experience applying machine learning algorithms in a real-world context, specifically focusing on predicting employee attrition using **logistic regression**. This is a classification problem where the goal is to predict whether an employee will leave the company (attrition) based on various features such as age, job satisfaction, years at the company, and more.

I recently completed my **Data Science Certificate**, and this project is one of the first significant applications of my learning. It demonstrates my ability to work with real-world datasets, perform data preprocessing, apply machine learning models, and assess their performance using key evaluation metrics.

## Objective

The primary goal of this project is to predict whether an employee will leave the company based on historical employee data. By doing so, companies can take proactive measures to reduce attrition and improve employee retention, which is a key factor in maintaining organizational stability.

## Dataset

The dataset used in this project is based on historical employee data with various features. Key features include:

- **Age**: The age of the employee.
- **BusinessTravel**: Whether the employee travels frequently, rarely, or not at all.
- **MonthlyIncome**: The monthly salary of the employee.
- **JobSatisfaction**: Employee satisfaction with their job (on a scale of 1-4).
- **Bonus**: Bonus received by the employee.
- **Department**: Department to which the employee belongs.
- **DistanceFromHome**: Distance of the employee’s home from the office.
- **EducationField**: The field of education the employee has.
- **EnvSatisfaction**: Satisfaction with the work environment.
- **JobRole**: The employee's job role.
- **MaritalStatus**: Marital status of the employee.
- **TrainingTimesLastYear**: Number of training sessions the employee attended in the last year.
- **YearsAtCompany**: Number of years the employee has worked at the company.
- **OverTime**: Whether the employee works overtime.
- **Attrition**: The target variable - whether the employee has left the company (Yes/No).

## Methodology

### 1. Data Preprocessing

The first step in this project was to clean and preprocess the data to make it suitable for machine learning. This involved:

- **Variable Selection**: I selected only the relevant variables related to employee attrition and removed unnecessary ones.
- **Handling Categorical Variables**: Since some features are categorical (e.g., BusinessTravel, JobRole, MaritalStatus), I used **One-Hot Encoding** (via `pd.get_dummies()`) to convert these categorical variables into a format suitable for machine learning algorithms.
- **Scaling**: I standardized the features using **StandardScaler** to ensure that each feature has the same scale, which is crucial for models like logistic regression.

### 2. Model Selection

After preprocessing, I used two machine learning algorithms:

- **Logistic Regression**: A linear model used for binary classification. This model is a great choice for predicting categorical outcomes like employee attrition.
- **Naive Bayes**: A probabilistic classifier based on Bayes’ theorem, which was also trained and evaluated to compare performance.

### 3. Data Splitting

To evaluate the performance of the models, I split the dataset into **training** and **test** sets using **stratified sampling**, which ensures that the class distribution of the target variable (attrition) is preserved in both the training and testing sets.

### 4. Model Training & Prediction

Once the data was preprocessed and split, I trained both the **logistic regression** and **Naive Bayes** models using the training set. I then used the trained models to predict employee attrition on the test set.

### 5. Model Evaluation

After making predictions, I evaluated the models using the following metrics:

- **Accuracy**: The percentage of correct predictions.
- **Precision**: How many of the predicted positives were actually positive (useful for detecting attrition).
- **Recall**: How many of the actual positives were correctly predicted (important for catching attrition cases).
- **F1-Score**: A balanced metric that combines precision and recall.

I used **confusion matrices** to better understand the performance of each model.

## Results

- **Logistic Regression**:
  - **Accuracy**: 85%
  - **Precision (Yes)**: 67%
  - **Precision (No)**: 86%
  - The logistic regression model showed a solid performance with high accuracy and precision for predicting "No Attrition." However, the model struggled with predicting "Yes Attrition."

- **Naive Bayes**:
  - **Accuracy**: 70%
  - **Precision (Yes)**: 27%
  - **Precision (No)**: 89%
  - Naive Bayes performed with lower accuracy and precision for predicting "Yes Attrition" compared to logistic regression.

## Conclusion

This project was a valuable learning experience in applying machine learning techniques to a real-world problem. While both models performed well overall, logistic regression provided better results for this specific problem. It was an exciting first step in my journey into data science, and I look forward to continuing to develop my skills and applying them to more complex projects.

## Future Work

- **Improvement of Model Performance**: I could further tune the logistic regression model using hyperparameter optimization (e.g., grid search or random search) to improve its performance, particularly for predicting "Yes Attrition."
- **Feature Engineering**: Additional features or transformations of existing features could be explored to improve model performance.
- **Advanced Algorithms**: Exploring more complex algorithms such as **Random Forest**, **Support Vector Machines (SVM)**, or **Neural Networks** could help improve prediction accuracy.

## Technologies Used

- **Python**: The primary programming language used for the project.
- **Pandas**: For data manipulation and preprocessing.
- **NumPy**: For numerical operations.
- **Scikit-Learn**: For machine learning algorithms, model training, and evaluation.
- **Matplotlib/Seaborn**: For data visualization (not included in the code but can be added for further analysis).

## How to Run the Project

1. Clone the repository to your local machine.

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt

3. Run the Jupyter notebook to execute the code and see the results:
   ```bash
   jupyter notebook Logistic_regression_project.ipynb

   Alternatively, if you want to run the Python script:
   ```bash
   python Logistic_regression_project.ipynb





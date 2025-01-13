# Loan Approval Prediction

## Introduction
This project aims to predict loan approval status using various machine learning algorithms. The dataset has undergone preprocessing to ensure the accuracy and efficiency of the models. The project demonstrates the application of supervised learning techniques for binary classification.

---

## Team Members

- **Gopala Ram Jiyani** (B22ME075)
- **Hetarth Jodha** (B22MT021)
- **Pushkin Dugam** (B22ME052)
- **Vinit Thakur** (B22ES026)
- **Wasim Akram** (B22CI048)

---

## Data Preprocessing

### Techniques Used
- **Normalization (Min-Max Scaling):** Rescales the data to lie between 0 and 1. Ensures uniform scaling, critical for distance-based algorithms like KNN and SVM.
- **Label Encoding:** Converts categorical data into numerical format by assigning a unique integer value to each category.
- **Null Value Replacement:** Ensures completeness of the dataset by handling missing values.

---

## Machine Learning Models Used

### 1. Logistic Regression
- **Purpose:** Designed for binary classification problems.
- **How it works:**
  - Predicts probabilities using the sigmoid function.
  - Optimized using Gradient Descent to minimize the cost function.

### 2. K-Nearest Neighbors (KNN)
- **Purpose:** Classification and regression tasks.
- **Key Features:**
  - Non-parametric and instance-based.
  - Predictions based on the majority vote or averaging.

### 3. Decision Tree
- **Purpose:** Classification and regression tasks.
- **Key Features:**
  - Recursive splitting based on feature values.
  - Gini impurity or entropy used for splitting criteria.

### 4. Support Vector Machines (SVM)
- **Purpose:** Classification by finding the optimal hyperplane.
- **Key Features:**
  - Uses kernel functions like Linear, RBF, and Polynomial.
  - Optimized with Grid Search and Cross-Validation.

### 5. Additional Models (Planned):
- XGBoost
- Random Forest
- Voting Classifier

---

## Results
- Models will be evaluated on a custom dataset.
- Each model will predict loan approval status and calculate accuracy.
- Scores out of 5 will be assigned (1 point for each approved loan).

---

## Future Enhancements
- Incorporate advanced models like XGBoost and Random Forest.
- Experiment with ensemble methods for improved accuracy.

---

## License
This project is open-sourced under the MIT License.

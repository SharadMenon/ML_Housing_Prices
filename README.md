🏠 ML_Housing_Prices
This is my first machine learning project, focused on predicting housing prices using various regression models. The dataset used is from Kaggle and contains a variety of features such as number of bedrooms, bathrooms, furnishing status, etc.

📌 Project Overview
The goal of this project was to:

Understand the relationship between different house features and their price.

Build and evaluate machine learning models for regression.

Learn and apply feature preprocessing, model optimization, and evaluation techniques.

📂 Dataset
Source: Kaggle Housing Price Dataset (Uploaded)

Total records: 545 entries

Features include both numerical and categorical data (like location, furnishing status, presence of amenities, etc.)

🧠 Steps Followed
Data Preprocessing

Loaded dataset using Pandas.

Assigned the target variable (house price) to y and features to X.

Encoded categorical variables using Label Encoding and One Hot Encoding.

Handled the dummy variable trap by removing one dummy column for each categorical variable.

Feature Selection

Used Backward Elimination (based on p-values) with StatsModels to select significant features affecting the house price.

Train-Test Split

Split the dataset into 80% training and 20% testing using train_test_split.

Feature Scaling

Applied StandardScaler to handle feature ranges and outliers.

Model Building

Built and evaluated the following models:

🔹 Multiple Linear Regression

🌳 Decision Tree Regressor

🌲 Random Forest Regressor

Model Evaluation

Used metrics to evaluate model performance:

R² Score

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

Visualized Actual vs Predicted Price for both Training and Testing sets.

📊 Results & Visualizations
Below are some sample visualizations showing actual vs predicted prices:
![Figure 2025-04-14 220226 (5)](https://github.com/user-attachments/assets/a1f181ac-015f-4fce-90ba-4b13c882b9c1)
![Figure 2025-04-14 220226 (4)](https://github.com/user-attachments/assets/078b0d09-5e87-4b85-b5ff-00922d5857fc)
![Figure 2025-04-14 220226 (3)](https://github.com/user-attachments/assets/11d94f90-3181-48c2-b4fb-ebb6280fff07)
![Figure 2025-04-14 220226 (2)](https://github.com/user-attachments/assets/629e0cee-6fda-48f9-be9b-a02bf230a2e1)
![Figure 2025-04-14 220226 (1)](https://github.com/user-attachments/assets/e43ed6cd-3d0c-4618-abee-eea9652b14c6)
![Figure 2025-04-14 220226 (0)](https://github.com/user-attachments/assets/295d7692-209c-45cf-827e-c646e9837dfa)


🔍 Observations
Despite using advanced models like Random Forest, the R² score was relatively low.

This might be due to:

Limited size or quality of the dataset.

Missing important features (like location-specific factors).

Noise in data or outliers not captured fully.

💡 Learnings
Learned how to preprocess and encode real-world data.

Understood the importance of feature selection and avoiding data leakage.

Gained hands-on experience with model evaluation metrics.

Realized the importance of visual analysis alongside numerical metrics.


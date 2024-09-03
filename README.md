# MLproject2
This Python script focuses on processing and analyzing hotel review data to predict reviewer scores using machine learning models, particularly K-Nearest Neighbors (KNN) and Logistic Regression. Here's a breakdown of the workflow and functions used in the script:

Data Reading and Preprocessing
read_data(path): Reads the CSV file and splits the data into features (x) and target variable (y).

handle_nulls(X_train, y_train): Handles missing values in the training data by filling them with the mode or using forward-fill for certain columns.

find_outliers_IQR(df) and impute_outliers_IQR(df): Detect and impute outliers in the data using the Interquartile Range (IQR) method.

outlier(col): Applies the outlier detection and imputation functions to specific columns in the data.

Feature_Encoder(X): Encodes categorical features in the data using LabelEncoder.

scale(encoded_Data): Scales the features to a range between 0 and 1 using MinMaxScaler.

encode_y(y): Encodes the target variable y into numerical categories.

Feature Selection and Model Training
select_features(K, X, Y, X_test): Selects the top K features using the SelectKBest method with the Chi-square test.
Model Training and Evaluation
K-Nearest Neighbors (KNN): The script trains the KNN model using different hyperparameters (n_neighbors, metric) and evaluates it using cross-validation. The best model is selected based on accuracy.

Logistic Regression: Similarly, Logistic Regression is trained with different penalties (l1, l2) and regularization strengths (C). The best model is chosen based on cross-validation accuracy.

Model Evaluation: Both models are evaluated on the test set, and metrics such as accuracy, training time, and test time are recorded.

Visualization
Scatter Plot: A scatter plot visualizes the relationship between predicted values and certain features.
Bar Graph: A bar graph is used to compare accuracy, training time, and test time.

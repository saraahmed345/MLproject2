#!/usr/bin/env python
# coding: utf-8

# In[51]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from pickle import dump
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import OneHotEncoder
import pickle


def read_data(path):
    data = pd.read_csv(path)
   
    x = data.loc[:, data.columns != 'Reviewer_Score']
    y = data['Reviewer_Score']
    return data, x, y


def handle_nulls(X_train, y_train):
    #checking existence of null values
    nullvalues1 = X_train.isna().sum()
#     print("total number of missing values in each column in xtrain \n", nullvalues1)
#     print("#################################\n")
    nullvalues2 = y_train.isna().sum()
#     print("total number of missing values in y_train\n", nullvalues2)
#     print("#################################\n")
#     print("datatype of each column in x_train\n", X_train.dtypes)
#     print("#################################\n")
#     print("datatype of  y_train\n", y_train.dtypes)
#     print("#################################\n")

    #handling null values in x_train
    filldata1 = X_train.fillna(X_train['Review_Total_Positive_Word_Counts'].mode()[0], inplace=True)
    filldata2 = X_train.fillna(X_train['Total_Number_of_Reviews_Reviewer_Has_Given'].mode()[0], inplace=True)
    filldata3 = X_train["Tags"].fillna(method ='ffill', inplace = True)
    filldata4 = X_train["days_since_review"].fillna( method ='ffill', inplace = True)
    filldata5 = X_train.fillna(X_train['lat'].mode()[0], inplace=True)
    filldata6 = X_train.fillna(X_train['lng'].mode()[0], inplace=True)
    newdata   = X_train.isna().sum()
#     print("number of nulls in x_train after handling missing values \n",newdata)
#     print("#################################\n")

    #handling null values in y_train
    filldata7 = y_train.fillna( method ='ffill', inplace = True)
    newdata2 = y_train.isna().sum()
#     print("number of nulls in y_train after handling missing values \n", newdata2)
#     print("#################################\n")
    return X_train, y_train


#outlier detection and handling 
def find_outliers_IQR(df):
    q1=df.quantile(0.25)
    q3=df.quantile(0.75)
    IQR=q3-q1
    outliers = df[((df<(q1-1.5*IQR)) | (df>(q3+1.5*IQR)))]
    return outliers

def impute_outliers_IQR(df):
    q1=df.quantile(0.25)
    q3=df.quantile(0.75)
    IQR=q3-q1
    upper = df[~(df>(q3+1.5*IQR))].max()
    lower = df[~(df<(q1-1.5*IQR))].min()
    df = np.where(df > upper,df.mean(),np.where(df < lower,df.mean(),df ) )
    return df


# In[194]:


def outlier(col):
    outliers = find_outliers_IQR(data[col])
#     print("number of outliers from " + col +" : "+str(len(outliers)))
#     print ("------------------------------------")
#     print("max outlier value: "+ str(outliers.max()))
#     print("min outlier value: "+ str(outliers.min()))
#     print ("------------------------------------")
    data[col] = impute_outliers_IQR(data[col])
    outliers = find_outliers_IQR(data[col])
#     print("max outlier value 2: "+ str(outliers.max()))
#     print("min outlier value 2: "+ str(outliers.min()))
#     print ("************")
# =============================================================================
# #handling duplicates
# X_train.drop_duplicates(inplace=True)
# y_train.drop_duplicates(inplace=True)
# =============================================================================


# In[195]:


#Feature encoding 
def Feature_Encoder(X):
    cols = ('Hotel_Address','Review_Date','Hotel_Name','Reviewer_Nationality',
        'Negative_Review','Positive_Review','Tags','days_since_review')
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))
    return X  
def scale(encoded_Data):
    #Feature scaling
    scaler = MinMaxScaler()
    scaler.fit(encoded_Data)
    new_data = scaler.transform(encoded_Data)
    new_data2 = pd.DataFrame(new_data,columns = ['Hotel_Address','Additional_Number_of_Scoring',
                 'Review_Date','Average_Score','Hotel_Name','Reviewer_Nationality','Negative_Review',
                 'Review_Total_Negative_Word_Counts','Total_Number_of_Reviews','Positive_Review',
                 'Review_Total_Positive_Word_Counts','Total_Number_of_Reviews_Reviewer_Has_Given',
                                                            'Tags','days_since_review','lat','lng'])
    return new_data2
#     print("feature scaling of x_train\n")
#     print(new_data2)


# In[196]:


def encode_y(y):
    y_test = []
    for i in y:
        if i == "Low_Reviewer_Score":
            y_test.append(0) 
        elif i == "Intermediate_Reviewer_Score":
            y_test.append(1) 
        else:
            y_test.append(2)
    return y_test

   


# In[197]:



data, x, y = read_data('hotel-classification-dataset.csv')
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
X_train, y_train = handle_nulls(X_train, y_train)

outlier("Review_Total_Negative_Word_Counts")
outlier("Review_Total_Positive_Word_Counts")
outlier("Total_Number_of_Reviews")
outlier("Total_Number_of_Reviews_Reviewer_Has_Given")
outlier("lng")
outlier("Additional_Number_of_Scoring")

X = X_train
Y = y_train
encoded_Data = Feature_Encoder(X)
Y = encode_y(Y)

# print("feature encoding of X_train")
# print(encoded_Data)
# print("**************")
# print("feature encoding of y_train\n")
# print(Y)
# print("**************")
new_data2 = scale(encoded_Data)#after scalling
X_train = new_data2
#=============================================================================================================


# In[198]:


#applying preprocessing on x_test/y-test
#handling missing values
# print("#########################################")
# print("checking null values of x_test\n",X_test.isna().sum()) 
# print("#########################################")
mode1 = X_train['lat'].mode()[0]
mode2 = X_train['lng'].mode()[0]
filldata8 = X_test['lat'].fillna(mode1, inplace=True)
filldata9 = X_test['lng'].fillna(mode2, inplace=True)
# print("X-test after handling null values", X_test.isna().sum())
# print("#########################################")
# #=====================================================================


# In[199]:



#feature encoding

x = X_test
y = y_test
encoded_Data2 = Feature_Encoder(x)
# print("y_test enconding")
y_test = encode_y(y_test)
# print(y_test)
# print("feature encoding of X_test")
# print(encoded_Data2)
# print("**************")
# =======================================================================================================
# detecting outliers
# print("checking presence of outliers in x_test\n")
outlier("Total_Number_of_Reviews")
outlier("Additional_Number_of_Scoring")
# print("#########################################")
# ==================================================================================
# Feature scaling
scaler = MinMaxScaler()
scaler.fit(encoded_Data2)
encoded_Data2 = scaler.transform(encoded_Data2)
encoded_Data2_dataframe = pd.DataFrame(encoded_Data2,
                                       columns=['Hotel_Address', 'Additional_Number_of_Scoring', 'Review_Date',
                                                'Average_Score', 'Hotel_Name', 'Reviewer_Nationality',
                                                'Negative_Review', 'Review_Total_Negative_Word_Counts',
                                                'Total_Number_of_Reviews', 'Positive_Review',
                                                'Review_Total_Positive_Word_Counts',
                                                'Total_Number_of_Reviews_Reviewer_Has_Given', 'Tags',
                                                'days_since_review', 'lat', 'lng'])
# print("x_test after scaling \n", encoded_Data2_dataframe)
# print("#########################################")
# ===============================================================================================================


# In[200]:


# feature selection in x_train/y-train
# for i in range(len(encoded_df)):
# In[201]:


encoded_Data = encoded_Data.drop(columns=['Hotel_Address', 'Additional_Number_of_Scoring','Review_Date','Average_Score','Hotel_Name','Reviewer_Nationality','Total_Number_of_Reviews','Positive_Review','Total_Number_of_Reviews_Reviewer_Has_Given','Tags','days_since_review','lat','lng'])

# In[202]:
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# In[203]:


# In[57]:
print(encoded_Data2_dataframe.dtypes)

def select_features(K, X, Y, X_test):
#     print (K)
#     print(X)
#     print(Y)
    features = SelectKBest(score_func = chi2, k = K).fit(X, Y)
    columns = features.get_support()
    selected_features = X.columns[columns]
    print("selected Features\n", selected_features)
    print("#########################################")
    selected = X[selected_features]
    X_test = X_test[selected_features]
    return selected, X_test

# X = select_features(6, new_data2, Y)
# X


# In[66]:


for i in range (len(X_train.columns)):
    print ("@ i = ", i + 1)
    x, X_test = select_features(i + 1, X_train, Y, encoded_Data2_dataframe)
    k_values = [1, 3, 5, 7, 9]
    distance_metrics = ['euclidean', 'manhattan', 'chebyshev']
    hyperparameter_grid = {'n_neighbors': k_values, 'metric': distance_metrics}
    # create KNN_classifer object
    classifier = KNeighborsClassifier()
    # Measure the training time
    start_time = time.time()

    # Create a grid search object with 5-fold cross-validation
    grid_search = GridSearchCV(classifier, hyperparameter_grid, cv=5, scoring='accuracy')

    # Fit the grid search object to the training data
    grid_search.fit(x,Y)
    end_time = time.time()
    
    best_knn = KNeighborsClassifier(n_neighbors=grid_search.best_params_['n_neighbors'],
                                metric=grid_search.best_params_['metric'])
    best_knn.fit(x, Y)
    y_pred_train = best_knn.predict(x)

    print("Best hyperparameters:", grid_search.best_params_)
    print("Accuracy:", grid_search.best_score_)
print(x)

# In[4]:


#save the model to disk
# filename = 'completed_modelclass.sav'
# pickle.dump(classifier, open(filename, 'wb'))
# train_time = end_time - start_time

# print("Best hyperparameters:", grid_search.best_params_)
# print("Accuracy:", grid_search.best_score_)
# # In[204]:


# In[ ]:


x, X_test = select_features(6, X_train, Y, encoded_Data2_dataframe)
k_values = [1, 3, 5, 7, 9]
distance_metrics = ['euclidean', 'manhattan', 'chebyshev']
hyperparameter_grid = {'n_neighbors': k_values, 'metric': distance_metrics}
# create KNN_classifer object
classifier = KNeighborsClassifier()
grid_search = GridSearchCV(classifier, hyperparameter_grid, cv=5, scoring='accuracy')

# Fit the grid search object to the training data
grid_search.fit(x,Y)
best_knn = KNeighborsClassifier(n_neighbors=grid_search.best_params_['n_neighbors'],
                                metric=grid_search.best_params_['metric'])
best_knn.fit(x, Y)


# In[14]:


#Predict the response for train dataset
y_pred_train = best_knn.predict(x)


# In[17]:


train_time = start_time - end_time


# In[15]:


#Predict the response for test dataset
# Measure the test time
start_time = time.time()
y_pred_test = best_knn.predict(encoded_Data2_dataframe)
end_time = time.time()
test_time = end_time - start_time


# In[18]:


from sklearn.metrics import mean_squared_error

accuracy = metrics.accuracy_score(y_test, y_pred_test)
print("Training time for k-NN: {:.4f} seconds",train_time)
print("Accuracy of train of knn :",metrics.accuracy_score(Y, y_pred_train))
print("Accuracy of test of knn :",accuracy)
print("Test time for k-NN: {:.4f} seconds",test_time)
# Compute the mean squared error (MSE)
mse = mean_squared_error(y_test, y_pred_test)
# Print the MSE
print("Mean Squared Error: {:.2f}".format(mse))


# In[19]:


plt.figure(figsize = (30,15))
plt.subplot(2,3,2)
plt.scatter(encoded_Data2_dataframe["Negative_Review"], encoded_Data2_dataframe["Review_Total_Negative_Word_Counts"], c=y_pred_test, marker= '*', s=80,edgecolors='purple')
plt.title("Predicted values with k=3", fontsize=40)


# In[20]:


# Create a bar graph for accuracy, training time, and test time
fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(["Accuracy","Training time", "Test time"], [accuracy, train_time, test_time])
ax.set_title("k-NN with k={} on Iris dataset".format(3))
ax.set_ylabel("Time (seconds) / Accuracy")
plt.show()


# In[ ]:


# model retrive step in test script:
# load the model from disk
#loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.score(X_test, Y_test)
#print(result)


# In[62]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pickle
import time
import matplotlib.pyplot as plt


# In[63]:


for i in range (len(X_train.columns)):
    print ("@ i = ", i + 1)
    x, X_test = select_features(i + 1, X_train, Y, encoded_Data2_dataframe)
    penalties = ['l1', 'l2']
    C_values = [0.1, 1, 10]
    hyperparameter_grid = {'penalty': penalties, 'C': C_values}

    logreg = LogisticRegression()

    grid_search = GridSearchCV(logreg, hyperparameter_grid, cv=5, scoring='accuracy')
    grid_search.fit(x, Y)
    clf = LogisticRegression(penalty = grid_search.best_params_['penalty'],
                         C = grid_search.best_params_['C'])
    # train the model and calculate training time
    start_time = time.time()
    clf.fit(x, Y)
    y_pred_train = clf.predict(x)
    accuracy_train = metrics.accuracy_score(Y, y_pred_train)
    print("Accuracy of train of Logistic_Regression:", accuracy_train)


    # predict the response for test dataset and calculate testing time
    start_time = time.time()
    y_pred_test = clf.predict(X_test)
    testing_time = time.time()- start_time

    # calculate accuracy of test dataset
    accuracy_test2 = metrics.accuracy_score(y_test, y_pred_test)
    print("Accuracy of test of Logistic_Regression:", accuracy_test2)
    print("Best hyperparameters:", grid_search.best_params_)
    print("Accuracy:", grid_search.best_score_)


# In[28]:


x, X_test = select_features( 6, X_train, Y, encoded_Data2_dataframe)
# create logistic regression object
clf = LogisticRegression(penalty= grid_search.best_params_['penalty'],
                         C= grid_search.best_params_['C'])

# train the model and calculate training time
start_time = time.time()
clf.fit(x, Y)
training_time = time.time() - start_time

# save the model to disk
# file = 'complete_modelclass.sav'
# pickle.dump(clf, open(file, 'wb'))

# predict the response for train dataset
y_pred_train = clf.predict(x)


# In[29]:


# calculate accuracy of train dataset
accuracy_train = metrics.accuracy_score(Y, y_pred_train)
print("Accuracy of train of Logistic_Regression:", accuracy_train)


# predict the response for test dataset and calculate testing time
start_time = time.time()
y_pred_test = clf.predict(encoded_Data2_dataframe)
testing_time = time.time()- start_time

# calculate accuracy of test dataset
accuracy_test2 = metrics.accuracy_score(y_test, y_pred_test)
print("Accuracy of test of Logistic_Regression:", accuracy_test2)

# print training and testing time
print("Training time of Logistic_Regression:", training_time)
print("Testing time of Logistic_Regression:", testing_time)


# In[30]:


# calculate mean square error of train dataset
mse_train = metrics.mean_squared_error(Y, y_pred_train)
print("Mean square error of train of Logistic_Regression:", mse_train)

# calculate mean square error of test dataset
mse_test = metrics.mean_squared_error(y_test, y_pred_test)
print("Mean square error of test of Logistic_Regression:", mse_test)


# In[31]:


# bar graph for Accuracy ,Training time and testing time
fig, ax = plt.subplots(figsize=(8, 6))

ax.bar(["Accuracy", "Training time", "Testing time"], [accuracy_test2, training_time, testing_time])

ax.set_title("bar graph of Logistic Regression ")

ax.set_ylabel("Time (s) / Accuracy")

plt.show()

# plot the predicted values against the actual values
plt.scatter(y_test, y_pred_test)

# Add a diagonal linefor comparison
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red')

# Add labels and title
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Actual vs. Predicted')

# show the plot
plt.show()

# model retrive step in test script:
# load the model from disk
#loaded_model= pickle.load(open(file, 'rb'))
#result = loaded_model.score(X_test, Y_test)
#print(result)


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

for i in range (len(X_train.columns)):
    print ("@ i = ", i + 1)
    x, X_test = select_features(i + 1, X_train, Y, encoded_Data2_dataframe)
    n_estimators = [100, 200, 300]
    max_depth = [5, 10, 15]
    min_samples_split = [2, 5, 10]
    hyperparameter_grid = {'n_estimators': n_estimators, 'max_depth': max_depth, 'min_samples_split': min_samples_split}
    
    start_time = time.time()
    # Create a Random Forest object
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(rf, hyperparameter_grid, cv=5, scoring='accuracy')
    grid_search.fit(x, Y)
    training_time = time.time()- start_time
    rf = RandomForestClassifier(n_estimators=grid_search.best_params_['n_estimators'],
                            max_depth=grid_search.best_params_['max_depth'],
                            min_samples_split=grid_search.best_params_['min_samples_split'])
    rf.fit(x, Y)
    y_pred_train = rf.predict(x)
    # calculate accuracy of train dataset of the SGDClassifier model
    accuracy1 = metrics.accuracy_score(Y,y_pred_train )
    print("Accuracy of train of Random forest model:", accuracy1)

    # Predict the classes of the testing dataset
    start_time = time.time()
    y_pred_test = rf.predict(X_test)
    testing_time = time.time()- start_time
    # calculate accuracy of test dataset of the SGDClassifier model
    accuracy2 = metrics.accuracy_score(y_test, y_pred_test)
    print("Accuracy of test of Random forest model:", accuracy2)
    print("Best hyperparameters:", grid_search.best_params_)
    print("Accuracy:", grid_search.best_score_)


# In[ ]:


print("Random forest model")

x, X_test = select_features(i + 1, X_train, Y, encoded_Data2_dataframe)
# Train the model
rf = RandomForestClassifier(n_estimators=grid_search.best_params_['n_estimators'],
                            max_depth=grid_search.best_params_['max_depth'],
                            min_samples_split=grid_search.best_params_['min_samples_split'])
rf.fit(x, Y)

# Predict the classes of the training dataset
y_pred_train = rf.predict(x)
# calculate accuracy of train dataset of the SGDClassifier model
accuracy1 = metrics.accuracy_score(Y,y_pred_train )
print("Accuracy of train of Random forest model:", accuracy1)

# Predict the classes of the testing dataset
start_time = time.time()
y_pred_test = rf.predict(X_test)
testing_time = time.time()- start_time
# calculate accuracy of test dataset of the SGDClassifier model
accuracy2 = metrics.accuracy_score(y_test, y_pred_test)
print("Accuracy of test of Random forest model:", accuracy2)

# print training and testing time
print("Training time of Random forest model:", training_time)
print("Testing time of Random forest model:", testing_time)

# calculate mean square error of train dataset
mse_train = metrics.mean_squared_error(Y, y_pred_train)
print("Mean square error of train of Random forest model:", mse_train)

# calculate mean square error of test dataset
mse_test = metrics.mean_squared_error(y_test, y_pred_test)
print("Mean square error of test of Random forest model:", mse_test)


# In[ ]:


#SGD


# In[59]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier 
#SGDClassifier (loss="hinge",max_iter=2000,random_state=42,alpha = 0.0001)

for i in range (len(X_train.columns)):
    print ("@ i = ", i + 1)
    x, X_test = select_features(i + 1, X_train, Y, encoded_Data2_dataframe)
    penalties = ['l1', 'l2']
    alphas = [0.0001, 0.001, 0.01]
    l1_ratios = [0.15, 0.5, 0.85]
    hyperparameter_grid = {'penalty': penalties, 'alpha': alphas, 'l1_ratio': l1_ratios}

    # Create a SGD classifier object
    sgd = SGDClassifier()

    # Create a grid search object with 5-fold cross-validation
    grid_search = GridSearchCV(sgd, hyperparameter_grid, cv=5, scoring='accuracy')

    # Fit the grid search object to the training data
    grid_search.fit(x, y_train)
    

    sgd = SGDClassifier(penalty=grid_search.best_params_['penalty'], 
                       alpha=grid_search.best_params_['alpha'],
                       l1_ratio=grid_search.best_params_['l1_ratio'])
    # Train the SGDClassifier model and calculate training time
    start_time = time.time()
    sgd.fit(x, Y) 
    training_time = time.time() - start_time
    y_pred_train = sgd.predict(x)
    accuracy1 = metrics.accuracy_score(Y,y_pred_train)
    print("Accuracy of train of SGDClassifier:", accuracy1)
    start_time = time.time()
    y_pred_test = sgd.predict(X_test)
    testing_time = time.time()- start_time
    # calculate accuracy of test dataset of the SGDClassifier model
    accuracy2 = metrics.accuracy_score(y_test, y_pred_test)
    print("Accuracy of test of SGDClassifier:", accuracy2)
    print("Best hyperparameters:", grid_search.best_params_)
    print("Accuracy:", grid_search.best_score_)
   


# In[ ]:


#assign the best model to save it
sgd = SGDClassifier(penalty = grid_search.best_params_['penalty'], 
                       alpha = grid_search.best_params_['alpha'],
                       l1_ratio=grid_search.best_params_['l1_ratio'])
x, X_test = select_features(5, X_train, Y, encoded_Data2_dataframe)
sgd.fit(x, Y)

# Save model to disk
# filename = 'SGDClassifier.sav'
# pickle.dump(sgd, open(filename, 'wb'))

# Predict the classes of the training dataset
y_pred_train = sgd.predict(x)
# calculate accuracy of train dataset of the SGDClassifier model
accuracy1 = metrics.accuracy_score(Y,y_pred_train )
print("Accuracy of train of SGDClassifier:", accuracy1)

# Predict the classes of the testing dataset
start_time = time.time()
y_pred_test = sgd.predict(X_test)
testing_time = time.time()- start_time
# calculate accuracy of test dataset of the SGDClassifier model
accuracy2 = metrics.accuracy_score(y_test, y_pred_test)
print("Accuracy of test of SGDClassifier:", accuracy2)

# print training and testing time
print("Training time of SGDClassifier model:", training_time)
print("Testing time of SGDClassifier model:", testing_time)

# calculate mean square error of train dataset
mse_train = metrics.mean_squared_error(Y, y_pred_train)
print("Mean square error of train of SGDClassifier:", mse_train)

# calculate mean square error of test dataset
mse_test = metrics.mean_squared_error(y_test, y_pred_test)
print("Mean square error of test of SGDClassifier:", mse_test)

# bar graph for Accuracy ,Training time and testing time
fig, ax = plt.subplots(figsize=(8, 6))

ax.bar(["Accuracy", "Training time", "Testing time"], [accuracy2, training_time, testing_time])

ax.set_title("bar graph of SGDClassifier model: ")

ax.set_ylabel("Time (s) / Accuracy")

plt.show()

# plot the predicted values against the actual values
plt.scatter(y_test, y_pred_test)

# Add a diagonal linefor comparison
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red')

# Add labels and title
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Actual vs. Predicted')

# show the plot
plt.show()


# In[ ]:





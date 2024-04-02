import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()

#select file data
file_path = r'C:\Users\Administrator\Videos\Bayo\BCSMASS 2010-2023.xlsx'
dataset = pd.read_excel(file_path)

#select the input and the ouput column
input_cols = ['BCSMASS 2011', 'BCSMASS 2012', 'BCSMASS 2013', 'BCSMASS 2014',
                 'BCSMASS 2015', 'BCSMASS 2016', 'BCSMASS 2017', 'BCSMASS 2018', 'BCSMASS 2019', 'BCSMASS 2020',
                     ]

output_cols = 'BCSMASS 2021'

#selecting the dependent and independent variables
X = dataset[input_cols].values
y = dataset[output_cols].values

#print (X)

#normalizing the dataset for X and y variable 
from sklearn.preprocessing import MinMaxScaler
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_x.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

#Split dataset into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=0)

#using the Random forest model
from sklearn.ensemble import RandomForestRegressor
#initialize the model and use it 
model = RandomForestRegressor(n_estimators=100, random_state=42)

#Fit the model
model.fit(X_train, y_train)

#model predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

#checking the model result for training set
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
Train_R_squared = r2_score(y_train, train_predictions)
Train_MSR = mean_squared_error(y_train, train_predictions)
Train_MAE = mean_absolute_error(y_train, train_predictions)

print ('Train_R-squared score', Train_R_squared)
print ('Train_Mean squared error', Train_MSR)
print ('Train_Mean absolute error', Train_MAE)

#checking the model result for test set
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
Test_R_squared = r2_score(y_test, test_predictions)
Test_MSR = mean_squared_error(y_test, test_predictions)
Test_MAE = mean_absolute_error(y_test, test_predictions)

print ('Test_R-squared score', Test_R_squared)
print ('Test_Mean squared error', Test_MSR)
print ('Test_Mean absolute error', Test_MAE)

#X = X.all()
#X = list(X)
#plot a graph

# Visualising the Random Forest Regression results (higher resolution)
# Plotting the results
fig = plt.figure(figsize=(10,8))
plt.scatter(y_test, test_predictions, alpha=0.5, label = 'Actual Values vs Predicted Values')

plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 
         linestyle='--', color='red', linewidth=2, label = 'Fit line y = 0.1054*x + 0.0243')

plt.title('Random Forest Regression: Actual BCSMASS vs. Predicted BCSMASS')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values ')
plt.show()
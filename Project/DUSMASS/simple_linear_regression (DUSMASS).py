# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import seaborn as sns
sns.set()
#select file data
file_path = r'C:\Users\Administrator\Videos\Bayo\DUSMASS 2010 - 2023.xlsx'
dataset = pd.read_excel(file_path)

#Handling missing values
columns_to_impute = ['DUSMASS 2010','DUSMASS 2011', 'DUSMASS 2012', 'DUSMASS 2013', 'DUSMASS 2014',
                 'DUSMASS 2015', 'DUSMASS 2016', 'DUSMASS 2017', 'DUSMASS 2018', 'DUSMASS 2019', 'DUSMASS 2020']

# Instantiate the SimpleImputer with the desired strategy (e.g., mean, median, most_frequent)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')

# Fit and transform the imputer on the specified columns
dataset[columns_to_impute] = imputer.fit_transform(dataset[columns_to_impute])


#select the input and the ouput column
input_cols = ['DUSMASS 2010','DUSMASS 2011', 'DUSMASS 2012', 'DUSMASS 2013', 'DUSMASS 2014',
                 'DUSMASS 2015', 'DUSMASS 2016', 'DUSMASS 2017', 'DUSMASS 2018', 'DUSMASS 2019', 'DUSMASS 2020'
                     ]

output_cols = 'DUSMASS 2021'

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

# Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
model = LinearRegression()
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

# Visualising the Random Forest Regression results (higher resolution)
# Plotting the results
fig = plt.figure(figsize=(10,8))
plt.scatter(y_test, test_predictions, alpha=0.5, label = 'Actual Values vs Predicted Values')

plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 
         linestyle='--', color='red', linewidth=2, label = 'Fit line y = 0.1054*x + 0.0243')

plt.title('Random Forest Regression: Actual DUSMASS vs. Predicted DUSMASS')


plt.xlabel('Actual Values', fontweight='bold', fontsize=12)
plt.ylabel('Predicted Values', fontweight='bold', fontsize=12)

plt.legend()
plt.show()

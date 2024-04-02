# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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


# Training the Polynomial Regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
model = PolynomialFeatures(degree = 4)
X_poly = model.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

Prediction = lin_reg_2.predict(X_poly)

print (Prediction)

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red', label = 'Actual Values vs Predicted Values')
plt.plot(X, lin_reg_2.predict(X_poly.fit_transform(X)), color = 'blue', label = 'Fit line y = 0.1054*x + 0.0243')
plt.title('Random Forest Regression: Actual BCSMASS vs. Predicted BCSMASS')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values ')
plt.legend()
plt.show()



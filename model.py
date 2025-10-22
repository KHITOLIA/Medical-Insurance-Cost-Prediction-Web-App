import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, adjusted_rand_score
import pickle
import warnings
warnings.filterwarnings("ignore")


insurance_data = pd.read_csv('insurance.csv')
insurance_data['sex'] = np.where(insurance_data['sex'] == 'male', 1, 0)
insurance_data['smoker'] = np.where(insurance_data['smoker'] == 'yes', 1, 0)
insurance_data.replace({'region' : {'southeast' : 0 , 'southwest' : 1, 'northeast' : 2, 'northwest' : 3}}, inplace=True)

# Splitting the features and target
X = insurance_data.drop(columns=['charges'], axis=1)
Y = insurance_data['charges']

# splitting into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Model Training Using Linear Regression while loading the Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Model Evaluation
Y_pred = regressor.predict(X_test)
r2 = r2_score(Y_test, Y_pred)
print(f"R2 Score: {r2}")
# Saving the model to disk
pickle.dump(regressor, open('model.pkl', 'wb'))
# # Loading the model from disk (for demonstration purposes)
# loaded_model = pickle.load(open('model.pkl', 'rb'))
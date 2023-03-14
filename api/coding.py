import pandas as pd 
import pickle
from sklearn.model_selection import train_test_split 
import shap

data = pd.read_pickle('data.pkl')
model = pickle.load(open('model.pkl', 'rb'))

X = data.drop(['Estimated_sales'], axis = 1)
y = data['Estimated_sales']

X_train ,X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 , random_state = 87)
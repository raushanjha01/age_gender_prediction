from matplotlib.pyplot import axis
import pandas as pd
import numpy as np
from sklearn import linear_model
import pickle
from category_encoders import BinaryEncoder

df = pd.read_csv(
    'event_ds.csv')

# predicting with longitude, latitude, Day, hour, week
# dropping the irrelevant columns in the multivariate dataset
df = df.dropna()
age_df = df.filter(['age'], axis=1)
df = df.drop(['Unnamed: 0', 'age','gender', 'group', 'app_id','is_installed','is_active','category', 'label_id'], axis='columns')

# encoding category data - Day
be = BinaryEncoder(cols=["Day"])
day_encode_model = be.fit(df)
new_df = day_encode_model.transform(df)

# regression model
reg = linear_model.LinearRegression()
reg.fit(new_df, age_df.age)

# Saving models to disk
pickle.dump(day_encode_model, open('ap_data_model.pkl', 'wb'))
pickle.dump(reg, open('age_model.pkl', 'wb'))


# test models
# Loading model to compare the results
dmodel = pickle.load(open('ap_data_model.pkl', 'rb'))
model = pickle.load(open('age_model.pkl', 'rb'))
data = [[117.0,34.0,'Monday', 3, 18]]
test_df = pd.DataFrame(data, columns=['longitude', 'latitude', 'Day', 'Hour', 'week'])
transformed_df = dmodel.transform(test_df)
print(round(model.predict(transformed_df)[0],2))


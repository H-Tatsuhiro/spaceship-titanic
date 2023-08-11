import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder

df_train = pd.read_csv("./train.csv")
df_test = pd.read_csv("./test.csv")

df_train = df_train.fillna({"CryoSleep": 2, "VIP": 2, "HomePlanet": "null", "Cabin": "null", "Destination": "null", "Age": 200, "RoomService":  15000, "FoodCourt": 30000, "ShoppingMall": 25000, "Spa": -1, "VRDeck": -1, "Name": "null"})

encoder = LabelEncoder()

for cname in df_train.columns:
    df_train[cname] = encoder.fit_transform(df_train[cname].values)

y = df_train['Transported']
X = df_train.drop('Transported', axis=1)

X_train = X.values
y_train = y.values

model = LinearSVC(random_state=0)
model.fit(X_train, y_train)

df_test_index = df_test.loc[:, ['PassengerId']]

df_test = df_test.fillna({"CryoSleep": 2, "VIP": 2, "HomePlanet": "null", "Cabin": "null", "Destination": "null", "Age": 200, "RoomService":  15000, "FoodCourt": 30000, "ShoppingMall": 25000, "Spa": -1, "VRDeck": -1, "Name": "null"})

for cname in df_test.columns:
    df_test[cname] = encoder.fit_transform(df_test[cname].values)


X_test = df_test.values
y_test = model.predict(X_test)
y_test_after = []

for i in y_test:
    if i == 0:
        y_test_after.append(True)
    else:
        y_test_after.append(False)

sub = pd.concat([df_test_index, pd.DataFrame(y_test_after, columns=['Transported'])], axis=1)
sub.to_csv('./sub.csv', index=False)

import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

df_train = pd.read_csv("./train.csv")
df_test = pd.read_csv("./test.csv")

amount_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
df_train.loc[df_train['CryoSleep']==True, amount_cols] = 0.0

firstName = []
lastName = []
for name in df_train['Name']:
    if type(name) != str:
        firstName.append("null"), lastName.append("null")
    else :
        firstName.append(name.split(' ')[0]), lastName.append(name.split(' ')[1])
df_train.drop('Name', axis=1, inplace=True)
df_train['FirstName'] = firstName
df_train['LastName'] = lastName

mean_imputer = SimpleImputer(strategy='mean')

num_cols = ["Age", "RoomService", "FoodCourt", "Spa", "ShoppingMall", "VRDeck"]
for col in num_cols:
    df_train[[col]] = mean_imputer.fit_transform(df_train[[col]])

sscaler = StandardScaler()
df_train_std = pd.DataFrame(sscaler.fit_transform(df_train.loc[:, num_cols]), columns=num_cols)

for col in num_cols:
    df_train[col] = df_train_std[col]

df_train = df_train.fillna({"CryoSleep": 2, "VIP": 2, "HomePlanet": "null", "Cabin": "null", "Destination": "null"})

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

df_test.loc[df_test['CryoSleep']==True, amount_cols] = 0.0

firstName = []
lastName = []
for name in df_test['Name']:
    if type(name) != str:
        firstName.append("null"), lastName.append("null")
    else :
        firstName.append(name.split(' ')[0]), lastName.append(name.split(' ')[1])
df_test.drop('Name', axis=1, inplace=True)
df_test['FirstName'] = firstName
df_test['LastName'] = lastName

for col in num_cols:
    df_test[[col]] = mean_imputer.fit_transform(df_test[[col]])

df_test_std = pd.DataFrame(sscaler.fit_transform(df_test.loc[:, num_cols]), columns=num_cols)

for col in num_cols:
    df_test[col] = df_test_std[col]

df_test = df_test.fillna({"CryoSleep": 2, "VIP": 2, "HomePlanet": "null", "Cabin": "null", "Destination": "null"})

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

import pandas as pd
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
# from pandas_profiling import ProfileReport
from lazypredict.Supervised import LazyRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor



def process_data(data):
    data = data.drop(['cbwd'], axis=1)
    for col in ['DEWP', 'TEMP', 'Iws', 'Is', 'Ir']:
        data[col] = data[col].apply(lambda x: 0.0 if x == 0 else x)
    return data.iloc[:5000]


data = pd.read_csv("PRSA_data.csv")


# profile = ProfileReport(data, title="Profiling Report")
# profile.to_file("Profiling_report.html")

# exit()

print(data.head())
print("#####################################################################################################################")
# sử lí giá trị bị thiếu
data = data.dropna()

# chuyển đổi cột "cbwd" thành giá trị
encoder = OneHotEncoder(sparse=False)
cbwd_encoded = encoder.fit_transform(data[['cbwd']])

cbwd_encoded_df = pd.DataFrame(cbwd_encoded, columns=encoder.get_feature_names_out(['cbwd']))

data = data.join(cbwd_encoded_df).drop('cbwd', axis=1)

# print(data.head())

# chuẩn hóa các cột số
numerical_cols = ['DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir']
scaler = StandardScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
# print(data.isnull().sum())
data = data.dropna()

# print(data.head())



# huấn luyện

# data = process_data(data)

X = data.drop(['pm2.5', 'No'], axis=1)
y = data['pm2.5']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X.tail())

# exit()

# reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=mean_squared_error)
# models, predictions = reg.fit(X_train, X_test, y_train, y_test)

# print(models)

# exit()

# dữ liệu để dự đoán dòng 26
sample_row = {
    'year': 2010, 
    'month': 1, 
    'day': 2, 
    'hour': 0,
    'DEWP': -16, 
    'TEMP': -4, 
    'PRES': 1020, 
    'Iws': 1.37, 
    'Is': 0, 
    'Ir': 0,
    'cbwd_NE': 0,
    'cbwd_NW': 0,
    'cbwd_SE': 1.00,
    'cbwd_cv': 0,
    
}

sample_df = pd.DataFrame([sample_row])
non_numerical_cols = sample_df.drop(columns=numerical_cols).columns
sample_df_scaled = scaler.transform(sample_df[numerical_cols])
sample_df_scaled = pd.DataFrame(sample_df_scaled, columns=numerical_cols)
sample_df = pd.concat([sample_df_scaled, sample_df[non_numerical_cols].reset_index(drop=True)], axis=1)
sample_df = sample_df[X_train.columns]

# sử dụng LinearRegression
lin_reg = LinearRegression()

lin_reg.fit(X_train, y_train)

y_pred = lin_reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("this is LinearRegression mse :",mse)
print("r2 on LinearRegression:",r2)
print("the predict pm2.5 is :",lin_reg.predict(sample_df))


    #sử dụng RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=200, random_state=42)

rf_reg.fit(X_train, y_train)

y_pred_rf = rf_reg.predict(X_test)

mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("this is mse of RandomForestRegressor :",mse_rf)
print("r2  RandomForestRegressor :",r2_rf)
print("the predict pm2.5 is :",rf_reg.predict(sample_df))

# sử dụng ExtraTreesRegressor
et_reg = ExtraTreesRegressor(n_estimators=200, random_state=42)

et_reg.fit(X_train, y_train)

y_pred_et = et_reg.predict(X_test)

mse_et = mean_squared_error(y_test, y_pred_et)
r2_et = r2_score(y_test, y_pred_et)

print("mse of ExtraTreesRegressor:", mse_et)
print("r2 of ExtraTreesRegressor:", r2_et)
print("the predict pm2.5 is :",et_reg.predict(sample_df))

# sử dụng GridSearchCV

# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_features': ['auto', 'sqrt', 'log2'],
#     'max_depth': [30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }

# grid_search = GridSearchCV(estimator=rf_reg, param_grid=param_grid, cv=3, n_jobs=-1, verbose=0, scoring='neg_mean_squared_error')

# grid_search.fit(X_train, y_train)

# best_params = grid_search.best_params_
# best_mse = -grid_search.best_score_

# print("sau khi GridSearchCV")
# print("best_params :",best_params)
# print("best_mse :",best_mse)

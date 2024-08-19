import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV


df = pd.read_csv("/content/drive/MyDrive/data science/AviaChiptaNarxlari/train_data.csv")
features = ['airline', 'flight', 'source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class', 'duration', 'days_left']
target = 'price'

X = pd.get_dummies(df[features])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
plt.figure(figsize=(10,6))
sns.histplot(df['price'], bins=50, kde=True)
plt.title('Narxlarning Taqsimoti')
plt.show()
numeric_df = df.select_dtypes(include=[np.number])

plt.figure(figsize=(12,8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Korrelatsiya Jadvali')
plt.show()

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
print(f'Linear Regression MSE: {mse_lr}')

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-scores)
print(f'Random Forest RMSE (cross-validated): {rmse_scores.mean()}')

rf_model = RandomForestRegressor(random_state=42)

param_distributions = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],        
    'min_samples_split': [2, 5]          
}

random_search = RandomizedSearchCV(rf_model, param_distributions, n_iter=5, cv=5, scoring='neg_mean_squared_error', random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)
best_rf_model = random_search.best_estimator_

y_pred_rf = best_rf_model.predict(X_test)
final_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print(f'Final Random Forest RMSE: {final_rmse}')

test_df = pd.read_csv("/content/drive/MyDrive/data science/AviaChiptaNarxlari/test_data.csv")
test_X = pd.get_dummies(test_df[features])
test_X = test_X.reindex(columns=X.columns, fill_value=0)
test_predictions = best_rf_model.predict(test_X)

output = pd.DataFrame({'id': test_df['id'], 'price': test_predictions})
output.to_csv('/content/drive/MyDrive/data science/AviaChiptaNarxlari/submission.csv', index=False)

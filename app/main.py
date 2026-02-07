import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# 1️⃣ Загрузка данных
sales_data = pd.read_csv("../data/raw/Online Retail.csv")
print("Sales data loaded:")
print(sales_data.head())

# -----------------------------
# 2️⃣ Преобразование InvoiceDate
# -----------------------------
sales_data['InvoiceDate'] = pd.to_datetime(sales_data['InvoiceDate'], errors='coerce')
sales_data = sales_data.dropna(subset=['InvoiceDate'])
print("InvoiceDate converted to datetime:")
print(sales_data[['InvoiceDate']].head())

# -----------------------------
# 3️⃣ Создание признаков даты
# -----------------------------
sales_data['Year'] = sales_data['InvoiceDate'].dt.year
sales_data['Month'] = sales_data['InvoiceDate'].dt.month
sales_data['Day'] = sales_data['InvoiceDate'].dt.day
sales_data['Week'] = sales_data['InvoiceDate'].dt.isocalendar().week
sales_data['DayOfWeek'] = sales_data['InvoiceDate'].dt.dayofweek + 1  # 1=Monday

print("Date features created:")
print(sales_data[['Year','Month','Day','Week','DayOfWeek']].head())

# -----------------------------
# 4️⃣ Агрегация по дням
# -----------------------------
daily_sales_data = sales_data.groupby(
    ['Country','StockCode','InvoiceDate','Year','Month','Day','Week','DayOfWeek']
).agg({'Quantity':'sum', 'UnitPrice':'mean'}).reset_index()

print("Aggregated daily sales data:")
print(daily_sales_data.head())

# -----------------------------
# 5️⃣ Разделение на train/test
# -----------------------------
split_date_train_test = pd.to_datetime("2011-09-25")
train_data = daily_sales_data[daily_sales_data['InvoiceDate'] <= split_date_train_test].copy()
test_data = daily_sales_data[daily_sales_data['InvoiceDate'] > split_date_train_test].copy()
print(f"Train data count: {len(train_data)}, Test data count: {len(test_data)}")

# -----------------------------
# 6️⃣ Кодирование категориальных признаков
# -----------------------------
le_country = LabelEncoder()
le_stock = LabelEncoder()

train_data.loc[:, 'CountryIndex'] = le_country.fit_transform(train_data['Country'])
train_data.loc[:, 'StockCodeIndex'] = le_stock.fit_transform(train_data['StockCode'])

# Функция для unseen labels в тесте
def encode_with_unknown(le, values):
    encoded = []
    for v in values:
        if v in le.classes_:
            encoded.append(le.transform([v])[0])
        else:
            encoded.append(-1)  # новые страны/товары
    return encoded

test_data = test_data[test_data['Country'].isin(train_data['Country'])]
test_data = test_data[test_data['StockCode'].isin(train_data['StockCode'])]

test_data.loc[:, 'CountryIndex'] = le_country.transform(test_data['Country'])
test_data.loc[:, 'StockCodeIndex'] = le_stock.transform(test_data['StockCode'])

# -----------------------------
# 7️⃣ Формируем признаки и цель
# -----------------------------
feature_cols = ['CountryIndex','StockCodeIndex','Month','Year','DayOfWeek','Day','Week']
X_train = train_data[feature_cols]
y_train = train_data['Quantity']
X_test = test_data[feature_cols]
y_test = test_data['Quantity']

# -----------------------------
# 8️⃣ Обучение модели Random Forest
# -----------------------------
print("Training Random Forest model...")
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
print("Model training completed.")

# -----------------------------
# 9️⃣ Предсказания и MAE
# -----------------------------
test_data['prediction'] = rf.predict(X_test)
mae = mean_absolute_error(y_test, test_data['prediction'])
print(f"Mean Absolute Error (MAE) on test set: {mae}")

# -----------------------------
# 🔟 Агрегация по неделям и прогноз на 39-ю неделю
# -----------------------------
weekly_test_predictions = test_data.groupby(['Year','Week'])['prediction'].sum().reset_index()
quantity_sold_w39 = int(weekly_test_predictions.loc[
    (weekly_test_predictions['Year']==2011) & (weekly_test_predictions['Week']==39),
    'prediction'
].values[0])
print(f"Predicted quantity sold in week 39 of 2011: {quantity_sold_w39}")
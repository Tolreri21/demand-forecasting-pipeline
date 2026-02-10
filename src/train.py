import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import mlflow
from sympy.printing.pretty.pretty_symbology import gb

# Загрузка данных
X_test = pd.read_parquet(
    "/Users/anatolijperederij/PycharmProjects/demand-forecasting-pipeline/data/processed/X_test.parquet")
X_train = pd.read_parquet(
    "/Users/anatolijperederij/PycharmProjects/demand-forecasting-pipeline/data/processed/X_train.parquet")
y_train = pd.read_parquet(
    "/Users/anatolijperederij/PycharmProjects/demand-forecasting-pipeline/data/processed/y_train.parquet")
y_test = pd.read_parquet(
    "/Users/anatolijperederij/PycharmProjects/demand-forecasting-pipeline/data/processed/y_test.parquet")

y_train = y_train.squeeze()
y_test = y_test.squeeze()

print(X_train.head())

# Настройка эксперимента
mlflow.set_experiment("models with 100 estimators ")

# Определяем модели
models = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42 , max_depth=10, min_samples_split=5, max_features="sqrt"),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=10, min_samples_split=5, max_features="sqrt")
}


def train_evaluate_model(model_name, model, X_train, y_train, X_test, y_test):
    """
this function trains our models and is logging all info about them
    """
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)


        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Логируем дополнительные метрики
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Логируем название модели как параметр
        mlflow.log_param("model_type", model_name)

        print(f"\n{model_name} Results:")
        print(f"  MSE:  {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R²:   {r2:.4f}")

        joblib.dump(model, f"../app/models/{model_name}.pkl")

        return model


# Обучение всех моделей
for model_name, model in models.items():
    train_evaluate_model(model_name, model, X_train, y_train, X_test, y_test)


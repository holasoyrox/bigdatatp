# TP Final - Herramientas para Grandes Volúmenes de Datos

## Predicción de tarifa (`fare_amount`) - NYC Yellow Taxi

Notebook: [`TP_Final_BigData_NYCTaxi.ipynb`](TP_Final_BigData_NYCTaxi.ipynb)

### Dataset

Muestra aleatoria (aprox. 400.000 filas) del dataset público **NYC Yellow Taxi Trip Records - de KAGGLE**
(NYC Taxi & Limousine Commission) - no la de databricks, tomada de 4 meses distintos (2015-01, 2016-01, 2016-02, 2016-03)
del dataset completo (aprox. 47 millones de filas, aprox. 7.2 GB). La limpieza y el feature
engineering se hacen con PySpark antes de reducir a un subconjunto manejable para el
entrenamiento con scikit-learn.

**Objetivo** estimar la tarifa de un viaje de taxi (`fare_amount`) a partir de datos
conocidos antes/durante el viaje (distancia, ubicación, hora, pasajeros).

### Índice del notebook

1. Carga y exploración con PySpark
2. Limpieza y feature engineering con PySpark
3. Muestra de trabajo para scikit-learn
4. Experimentos con MLflow Tracking (4 corridas con `RandomForestRegressor`)
5. Optimización de hiperparámetros con Optuna
6. Métricas comparativas de todas las corridas
7. Registro del modelo en MLflow Model Registry (Unity Catalog)
8. Validación end-to-end del modelo registrado (inferencia local, sin Model Serving)
9. Observabilidad y drift con Evidently (2015 vs 2016)
10. Interpretabilidad con SHAP
11. Instrucciones para reproducir
12. Conclusiones

### Resumen de resultados

- **Mejor modelo:** `RandomForestRegressor(n_estimators=150, max_depth=12)`, elegido entre 7
  corridas (4 de grid manual + 3 de Optuna), con RMSE=0.91 USD, MAE=0.29 USD y R²=0.991.
- **Features más importantes:** `trip_distance` (aprox. 81%) y `trip_duration_min` (aprox. 16%)
  concentran casi toda la importancia del modelo.
- **Drift 2015 vs 2016:** drift estadísticamente significativo en 7 de 11 columnas, con una
  degradación moderada del error (RMSE de 0.97 a 1.42 USD) sin colapso del modelo (R² > 0.98).
- **Limitaciones:** muestra parcial del dataset completo; Databricks Free Edition no soporta
  Model Serving pago, por lo que la validación end-to-end se hace con inferencia local.

El notebook está preparado para Databricks (usa `spark`, `dbutils`, `displayHTML`). La tabla
(`workspace.default.nyctaxi_sample`), el experimento de MLflow y el modelo de destino ya están
configurados en la celda de configuración. 
Todas las semillas aleatorias están fijadas en
`SEED = 584154` para que la ejecución sea reproducible.

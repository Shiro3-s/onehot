import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, KBinsDiscretizer

# 1. Carga del dataset y limpieza de columnas de leakage
df = pd.read_csv("maestro_joven_uk_muestra.csv")
X = df.drop(columns=["precio_mediana", "precio_medio", "anos_salario"])
y = df["precio_mediana"]

# 2. Separación de datos (Prevenir Data Leakage)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Agrupación de variables
cat_features = ["region"]
discrete_features = ["lluvia_anom"]
num_features = [col for col in X.columns if col not in cat_features + discrete_features]

# 4. Pipelines individuales
num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

discrete_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('discretizer', KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile'))
])

# 5. Pipeline unificado con ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features),
        ('disc', discrete_pipeline, discrete_features)
    ],
    remainder='passthrough'
)

# 6. Ejecución y extracción de features
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Mapeo a DataFrame final (Scikit-Learn >= 1.2)
feature_names = preprocessor.get_feature_names_out()
df_final_train = pd.DataFrame(X_train_processed, columns=feature_names, index=X_train.index)
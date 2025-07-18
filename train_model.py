import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import make_scorer, mean_absolute_error

def custom_mae(y_true, y_pred):
    """Custom scorer that ignores NaN predictions"""
    mask = ~np.isnan(y_pred)
    return mean_absolute_error(y_true[mask], y_pred[mask])

def train():
    df = pd.read_csv("enhanced_weather.csv")
    
    # Feature engineering
    X = df[[
        "temperature_2m_max", "temperature_2m_min",
        "relative_humidity_2m", "cloud_cover_proxy",
        "precipitation_lag1", "precipitation_lag2",
        "is_monsoon", "season", "temp_rolling_7"
    ]]
    y = df["precipitation_sum"]
    
    # Preprocessing pipeline
    numeric_features = ["temperature_2m_max", "temp_rolling_7"]
    categorical_features = ["season"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])
    
    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Model pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])
    
    # Hyperparameter grid
    param_grid = {
        'regressor__n_estimators': [100, 200],
        'regressor__max_depth': [10, 20, None],
        'regressor__min_samples_split': [2, 5]
    }
    
    # Grid search with time-series validation
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=tscv,
        scoring=make_scorer(custom_mae),
        n_jobs=-1,
        verbose=2
    )
    
    grid_search.fit(X, y)
    
    # Save best model
    joblib.dump(grid_search.best_estimator_, "rainfall_model_v2.pkl")
    
    # Evaluation
    print(f"Best MAE: {-grid_search.best_score_:.2f} mm")
    print("Best params:", grid_search.best_params_)
    
    # Feature importance
    if hasattr(grid_search.best_estimator_.named_steps['regressor'], 'feature_importances_'):
        importances = grid_search.best_estimator_.named_steps['regressor'].feature_importances_
        features = numeric_features + \
                  list(grid_search.best_estimator_.named_steps['preprocessor']
                      .named_transformers_['cat']
                      .get_feature_names_out(categorical_features))
        
        print("\nFeature Importances:")
        for feat, imp in sorted(zip(features, importances), key=lambda x: x[1], reverse=True):
            print(f"{feat}: {imp:.3f}")

if __name__ == "__main__":
    train()
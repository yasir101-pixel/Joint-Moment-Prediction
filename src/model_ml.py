import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from evaluate import evaluate_model, print_results, save_results
import os

def train_ml_models(X_train, y_train, X_test, y_test,
                    moment_names=None, output_dir=None, subject_name=''):
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    X_train_flat = X_train.reshape(n_train, -1)
    X_test_flat = X_test.reshape(n_test, -1)

    models = {
        'Ridge':        Ridge(alpha=1.0),
        'KNN':          KNeighborsRegressor(n_neighbors=5),
        'RandomForest': MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        'SVR':          MultiOutputRegressor(SVR(kernel='rbf', C=1.0, epsilon=0.1)),
    }

    all_results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_flat, y_train)
        y_pred = model.predict(X_test_flat)
        results = evaluate_model(y_test, y_pred, moment_names=moment_names)
        print_results(results, model_name=f"{name} - {subject_name}")
        all_results[name] = results
        if output_dir:
            csv_path = os.path.join(output_dir, f"results_ml_{name}_{subject_name}.csv")
            save_results(results, model_name=name, output_path=csv_path)
    return all_results
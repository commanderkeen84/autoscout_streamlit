# -*- coding: utf-8 -*-

import streamlit as st 
import pandas as pd 
import json

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split, cross_val_score 
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures,  StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
    
from sklearn.compose import ColumnTransformer
    
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
    
from sklearn.metrics import  r2_score, mean_absolute_error, mean_squared_error, make_scorer
import scipy.stats

from xgboost import XGBClassifier 
import xgboost 
import optuna


def app(): 
    st.header("Hyperparameter-tuning")
    st.subheader("Random Forest Parameters using Grid-Search")

    #read DataFrame 
    df_5= pd.read_csv("ML_data.csv")

    # preparting the dataset 
    X = df_5.drop(columns='price')
    y = df_5['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=9874654)

    # Define feature names
    categorical_features = ['make', 'model', 'fuel', 'gear', 'offerType']
    numerical_features = ['mileage', 'hp']  
    
    # Define transformers
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))]) 
    numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    
    # Define preprocessor 
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features), 
            ('num', numerical_transformer, numerical_features)
        ]
    )

    st.code('''

# Define the RandomForestRegressor model
rf_regressor = RandomForestRegressor(random_state=8956165)

# Define the pipeline with preprocessing and the RandomForestRegressor
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', rf_regressor)
])

# Define the hyperparameters grid for grid search
param_grid = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [None, 10, 20],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4],
    'regressor__max_features': [0.3, 'sqrt']
}

# Perform grid search using cross-validation
start_time = time.time()

grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

end_time = time.time()
elapsed_time = (end_time - start_time)/60
print(f"elapsed time {elapsed_time:.5f} minutes")

# Print the best hyperparameters found
print("Best parameters found by grid search:")
print(grid_search.best_params_)

# Get the best parameters found
best_params = grid_search.best_params_

# Define the Random Forest model with the best parameters
best_model = pipeline.set_params(**best_params)

# Fit the model to the data
best_model.fit(X_train, y_train)

# Make predictions on test data
y_pred = pipeline.predict(X_test)

# Evaluate performance
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)


    ''')

    elapsed_time = 13.24318
    st.write(f"elapsed time: {elapsed_time} minutes ")
    best_params={'regressor__max_depth': None, 'regressor__max_features': 'sqrt', 'regressor__min_samples_leaf': 1, 'regressor__min_samples_split': 5, 'regressor__n_estimators': 300}
    st.write(f"Best parameters found by grid search: {best_params}" )

    rf_regressor = RandomForestRegressor(random_state=8956165)

    pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', rf_regressor)
    ]) 

    st.write(f"Random forest regressor with gridsearch best parameters accuracy:")

    # Define the Random Forest model with the best parameters
    best_model = pipeline.set_params(**best_params)

    # Fit the model to the data
    best_model.fit(X_train, y_train)

    # Make predictions on test data
    y_pred = pipeline.predict(X_test)

    # Evaluate performance
    mse_rf = mean_squared_error(y_test, y_pred)
    mae_rf = mean_absolute_error(y_test, y_pred)

    # Display metrics in Streamlit
    st.write(f"- Test Mean Squared Error (MSE): {round(mse_rf, 1)}")
    st.write(f"- Test Mean Absolute Error (MAE): {round(mae_rf, 1)}")    
    
    with open('mae_rf_default.json', 'r') as f:
        acc_scores = json.load(f)
    mae_rf_default = acc_scores['mae_rf_default']
    st.write(f"  \nGrid search parameters slightly improve (test-) accuracy compared to default parameters  \nwith Test-MAE of {round(mae_rf_default),1} \n")

    st.write("Gridsearch with few parameters already takes 13 minutes, testing of more parameters with e.g. RandomizedSearchCV would take too much time")  
             
    # st.subheader("Gradient Boosted Random Forest Parameters using Grid search") -> daurt ca 40 minuten für die grid search 

    st.subheader("Optimizing hyperparameters using OPTUNA")

    st.code(''' 
            
def objective(trial):
    # Define hyperparameters to be optimized
    n_estimators = trial.suggest_int('n_estimators', 100, 600)
    max_depth = trial.suggest_int('max_depth', 3, 15)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    max_features = trial.suggest_categorical('max_features', [0.3, 'sqrt', 'log2'])
    random_state=584646231

    # Create Random Forest model with hyperparameters
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features
    )

    # Create pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

    # Evaluate model using cross-validation
    score = -cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error').mean()
    return score  # Return negative mean squared error as Optuna minimizes the objective

# Create study object and optimize hyperparameters
start_time = time.time()

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# Get the best hyperparameters
best_params = study.best_params
end_time = time.time()
elapsed_time = (end_time - start_time)/60

print(elapsed_time)
print("Best hyperparameters found:", best_params)
            
            ''') 
    
    best_params_opt={'n_estimators': 263, 'max_depth': 15, 'min_samples_split': 4, 'min_samples_leaf': 1, 'max_features': 0.3}
    elapsed_time=69.8

    st.write(f"elapsed time: {elapsed_time} minutes ")
    st.write(f"Best parameters found by grid search: {best_params_opt}" )

    # Create Random Forest model with the best hyperparameters and preprocessing
    best_model = RandomForestRegressor(**best_params_opt)

    # Create pipeline with best preprocessing and model
    rf_optuna = Pipeline(steps=[('preprocessor', preprocessor), ('model', best_model)])

    # Fit the pipeline to the data
    rf_optuna.fit(X_train, y_train)

    # Make predictions on test data
    y_pred = rf_optuna.predict(X_test)

    # Evaluate performance
    mse_optuna = mean_squared_error(y_test, y_pred)
    mae_optuna = mean_absolute_error(y_test, y_pred)

    # Display metrics
    st.write("Test-Accuracy:")
    st.write(f"- Mean Squared Error (MSE): {round(mse_optuna, 1)}")
    st.write(f"- Mean Absolute Error (MAE): {round(mae_optuna, 1)}")

    st.write("Test accuracy was better with the parameters chosen by grid-search")

# Hi dit is de functie module
import streamlit as st
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

#################################
# Page 1: OLS Regression
#################################

# Function to run OLS regression
def run_ols(x, y):
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    return model

#################################
# Page 3: Variable Selection Tool
#################################

# Code for data preparation
def data_preparation(dataset, relevant_columns):
    df_main = dataset.copy()
    df_main.replace(["", "-", "0"], np.nan, inplace=True)   # Convert potential empty strings or placeholders into NaN
    #df_main = df_main.dropna(subset=relevant_columns)       # Remove missing values
    df_main.reset_index(drop=True, inplace=True)

    # Apply Standardization (Z-score normalization)
    scaler = StandardScaler()
    df_main[relevant_columns] = scaler.fit_transform(df_main[relevant_columns])

    return df_main


# Function to plot correlation matrix
def correlation_matrix(var, df):
    # Compute the correlation matrix
    variables_corr = df[var]
    corr_matrix = variables_corr.corr()

    # Plot the correlation matrix with numbers inside each block
    fig, ax = plt.subplots(figsize=(26, 20))
    cax = ax.matshow(corr_matrix, cmap='coolwarm')
    
    # Adding numbers inside each cell
    for (i, j), val in np.ndenumerate(corr_matrix):
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black')
        
    # Adding color bar and labels
    plt.colorbar(cax)
    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_yticks(range(len(corr_matrix.index)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45)
    ax.set_yticklabels(corr_matrix.index)
    plt.title("Correlation Matrix with Numbers", pad=20)
    
    plt.show()


# Function to select & fit with Support Vector Method (linear)
def SVM_linear_select_fit(X, y, n_features = 2, split_data=False, test_size=0.3, random_state=42):
    
    # Optionally split dataset
    split_result = train_test_split(X, y, test_size=test_size, random_state=random_state) if split_data else (X, None, y, None)
    X_train, X_test, y_train, y_test = split_result
    if split_data:
        print(f"Training Set Size: {X_train.shape[0]} rows")
        print(f"Test Set Size: {X_test.shape[0]} rows")

    # Hyperparameter tuning using GridSearchCV
    param_grid = {
        'C': [0.1, 1, 10],
        'epsilon': [0.01, 0.1, 0.2]
    }
    svr = GridSearchCV(SVR(kernel='linear'), param_grid, cv=3, scoring='r2', n_jobs=-1) # n_jobs=-1 makes sure that work is divided in 8 CPU cores (computer chip workload)
    svr.fit(X_train, y_train)
    best_svr = svr.best_estimator_
    print("Best Hyperparameters:", best_svr)

    # Apply Recursive Feature Elimination (RFE)
    rfe = RFE(estimator=best_svr, n_features_to_select=n_features)  # Select number of features
    rfe_model = rfe.fit(X_train, y_train)

    # Transform the dataset to only include selected features
    X_train_rfe = rfe.transform(X_train)
    X_test_rfe = rfe.transform(X_test) if split_data else None
    X_rfe = rfe.transform(X) if split_data else None

    # Selected features
    selected_features = rfe.support_
    print("Selected Features (after refit):", selected_features)

    # Map indices to actual feature names if available
    if hasattr(X_train, 'columns'):
        select_feat_SVM = X_train.columns[rfe.support_]
        print("Selected Features Names (after refit):", select_feat_SVM)

    # Fit the model to the transformed data
    svr_model = best_svr.fit(X_train_rfe, y_train)

    # Check convergence
    if hasattr(svr_model, 'n_iter_'):
        print(f"Support Vector Method converged in {svr_model.n_iter_} iterations.")
    else:
        print("Convergence information not available.")

    # Return values based on split mode
    if split_data:
        return X_train, X_test, y_train, y_test, X_train_rfe, X_test_rfe, X_rfe, select_feat_SVM, svr_model
    else:
        return X_train_rfe, select_feat_SVM, svr_model
    

# Function to extract coefficients
def SVM_coefficients(svr_model):
    # extract coefficients
    SVM_coeff = svr_model.coef_.flatten() # Flatten bc it is multi-dimensial array, 'enumerate' won't work if not flattened
    print("SVM coefficients (after refit):", SVM_coeff)

    return SVM_coeff


# Function to predict SVM
def predict_SVM(X_test_rfe, y_test, svr_model):

    # Make predictions on the test data
    y_pred = svr_model.predict(X_test_rfe)

    # Evaluate model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R^2 Score: {r2:.2f}")

    return y_pred


# Function to fit with Elastic Net (Lasso+Ridge)
def elastic_net_fit_all(X, y, split_data=False, test_size=0.3, random_state=42):

    # Optionally split dataset
    if split_data:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        print(f"Training Set Size: {X_train.shape[0]} rows")
        print(f"Test Set Size: {X_test.shape[0]} rows")
    else:
        X_train, y_train = X, y
        X_test, y_test = None, None  # No test set in this case

    # Define the Elastic Net model
    elastic_net = ElasticNet(max_iter=1000000)

    # Perform grid search to tune hyperparameters
    param_grid = {
        'alpha': [0.1, 1, 10],  # Regularization strength
        'l1_ratio': [0.1, 0.5, 0.9]  # Mix of Lasso and Ridge (alpha in formula)
    }
    grid_search = GridSearchCV(elastic_net, param_grid, cv=5, scoring='neg_mean_squared_error')

    # Fit the model to the data
    EN_model_all = grid_search.fit(X_train, y_train)

    # Best estimators based on best hyperparameters
    EN_hyperparams = EN_model_all.best_estimator_
    print("Best Hyperparameters:", EN_model_all.best_params_)

    # Coefficients of the selected model
    EN_coeff_all = EN_hyperparams.coef_
    EN_coeff_nonzero = EN_coeff_all[EN_coeff_all != 0]

    # Print coefficients with variable names if x is a DataFrame
    if hasattr(X_train, 'columns'):
        feature_names = X_train.columns
    else:
        feature_names = [f'Feature_{i}' for i in range(len(EN_coeff_all))]  # Default to generic names

    # Print feature names and their corresponding coefficients
    print("\nAll Feature Coefficients (before refit):")
    for feature, coef in zip(feature_names, EN_coeff_all):
        print(f"{feature}: {coef:.4f}")

    # Check convergence
    if hasattr(EN_model_all, 'n_iter_'):
        print(f"Elastic Net converged in {EN_model_all.n_iter_} iterations.")
    else:
        print("Convergence information not available.")
    
    # Return values based on split mode
    if split_data:
        return X_train, X_test, y_train, y_test, EN_hyperparams, EN_coeff_all, EN_coeff_nonzero, EN_model_all
    else:
        return EN_hyperparams, EN_coeff_all, EN_coeff_nonzero, EN_model_all


# Function to select features from Elastic Net
def elastic_fit_select(x, y, EN_hyperparams, X_train=None, X_test=None, y_train=None, n_features=2):

    # Apply Recursive Feature Elimination (RFE)
    rfe = RFE(estimator=EN_hyperparams, n_features_to_select=n_features)

    # Only transform test data if data was splitted before
    if X_train is not None and X_test is not None and y_train is not None:
        rfe.fit(X_train, y_train)
        X_train_rfe = rfe.transform(X_train)
        X_test_rfe = rfe.transform(X_test)
        X_rfe = rfe.transform(x)
    else:
        rfe.fit(x, y)
        y_train = y
        X_train_rfe = rfe.transform(x)
        X_test_rfe = None
        X_rfe = None

    # Select feature indices from RFE
    selected_indices = np.where(rfe.support_)[0]

    # Map indices to actual feature names if available
    if hasattr(x, 'columns'):
        selected_feat_EN = x.columns[selected_indices].tolist()
    else:
        selected_feat_EN = selected_indices.tolist()  # Return indices if no feature names

    # Retrain Elastic Net on selected features
    EN_retrained = ElasticNet(alpha=EN_hyperparams.alpha, l1_ratio=EN_hyperparams.l1_ratio, max_iter=10000)
    EN_model_refit = EN_retrained.fit(X_train_rfe, y_train)
    EN_best_coeff = EN_model_refit.coef_

    print(f"\nSelected Features (after refit): {selected_feat_EN}")
    print(f"Selected Feature Coefficients (after refit): {EN_best_coeff}")

    # Check convergence
    if hasattr(EN_model_refit, 'n_iter_'):
        print(f"Elastic Net converged in {EN_model_refit.n_iter_} iterations.")
    else:
        print("Convergence information not available.")

    # Return values based on split mode
    if X_train is not None and X_test is not None and y_train is not None:
        return selected_feat_EN, X_train_rfe, X_test_rfe, X_rfe, EN_model_refit, EN_best_coeff
    else:
        return selected_feat_EN, X_train_rfe, EN_model_refit, EN_best_coeff


# Function to predict EN
def elastic_predict(X_test_rfe, y_test, EN_model_refit):
    # Make predictions on the test data
    y_pred = EN_model_refit.predict(X_test_rfe)

    # Evaluate model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R^2 Score: {r2:.2f}")

    return y_pred


# Function voor nonlinear SVM
def SVM_nonlinear_select_fit(X, y, n_features=2, split_data=False, test_size=0.3, random_state=42):
    
    # Optionally split dataset
    if split_data:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        print(f"Training Set Size: {X_train.shape[0]} rows")
        print(f"Test Set Size: {X_test.shape[0]} rows")    
    else:
        X_train, y_train = X, y
        X_test, y_test = None, None  # No test set in this case

    # Hyperparameter tuning using GridSearchCV
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 0.01, 0.1],
        'epsilon': [0.01, 0.1, 0.2]
    }
    svr = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=5, scoring='r2', n_jobs=-1)

    # Fit the model on the training data
    svr.fit(X_train, y_train)
    best_svr = svr.best_estimator_
    print("Best Hyperparameters:", svr.best_params_)
    
    # Compute feature importance using permutation importance
    perm_importance = permutation_importance(best_svr, X_train, y_train, scoring='r2', n_repeats=10, random_state=random_state)
    
    # Get the top n_features based on importance scores
    sorted_idx = perm_importance.importances_mean.argsort()[::-1][:n_features]
    selected_features = sorted_idx

    # Transform the dataset to only include selected features
    X_train_rfe = X_train.iloc[:, selected_features]

    # Only transform test data if split_data is True
    if split_data:
        X_test_rfe = X_test.iloc[:, selected_features]
        X_rfe = X.iloc[:, selected_features]
    else:
        X_test_rfe = None
        X_rfe = None

    # Selected features
    print("Selected Features Indices (after permutation importance):", selected_features)

    # Map indices to actual feature names if available
    if hasattr(X_train, 'columns'):
        select_feat_SVM = X_train.columns[selected_features]
        print("Selected Features Names (after permutation importance):", select_feat_SVM)

    # Fit the model to the training data
    svr_nonlinear = SVR(kernel='rbf', C=best_svr.C, epsilon=best_svr.epsilon, gamma=best_svr.gamma)
    svr_nonlinear = svr_nonlinear.fit(X_train_rfe, y_train)

    # Return values based on split mode
    if split_data:
        return X_train, X_test, y_train, y_test, X_train_rfe, X_test_rfe, X_rfe, select_feat_SVM, svr_nonlinear, perm_importance
    else:
        return X_train_rfe, select_feat_SVM, svr_nonlinear, perm_importance
    

# Function to predict
def predict_SVM_nonlinear(X_test_rfe, y_test, svr_nonlinear):
    # Make predictions on the test data
    y_pred = svr_nonlinear.predict(X_test_rfe)

    # Evaluate model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R^2 Score: {r2:.2f}")

    return y_pred


# Function to visualize feature importances nonlinear SVM (optional)
def visualize_feature_nonlinear_SVM(select_feat_SVM, perm_importance):
    # Extract importance values
    selected_importance_values = perm_importance.importances_mean[perm_importance.importances_mean.argsort()[::-1][:len(select_feat_SVM)]]
    
    # Plot feature importances
    plt.bar(select_feat_SVM, selected_importance_values)
    plt.xlabel('Feature Name')
    plt.ylabel('Permutation Importance Score')
    plt.title('Feature Importances (Non-Linear SVM - RBF Kernel)')
    plt.xticks(rotation=45)
    plt.show()


# Function to visualize feature importances linear SVM (optional)
def visualize_feature_importances_SVM(select_feat_SVM, SVM_coeff):
    plt.bar(select_feat_SVM, SVM_coeff)
    plt.xlabel('Feature Name')
    plt.ylabel('Coefficient Value')
    plt.title('Feature Importances (SVM)')
    plt.show()


# Function to visualize feature importances EN (optional)
def visualize_feature_importances_EN(select_feat_EN, EN_best_coeff):
    plt.bar(select_feat_EN, EN_best_coeff)
    plt.xlabel('Feature Name')
    plt.ylabel('Coefficient Value')
    plt.title('Feature Importances (Elastic Net)')
    plt.tight_layout()
    plt.show()


# Function to plot actual vs prediction results (optional)
def plot_pred_actual_results(y_test, y_pred):
    plt.scatter(y_test, y_pred, color='blue', label='Predicted')
    plt.plot([y_test.min(), y_test.max()], [y_pred.min(), y_pred.max()], 'r--', lw=2)  # Diagonale lijn
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.title('Regression Results')
    plt.show()


# Function for cross_validation
def cross_validation(select_method, x, y):
    # Perform 5-fold cross-validation
    scores = cross_val_score(select_method, x, y, cv=5, scoring='r2')
    print("Cross-validated R^2 scores:", scores)
    print("Mean R^2 score:", scores.mean())
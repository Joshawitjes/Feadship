########################################################################
# Page 3: Variable Selection Tool
########################################################################

import streamlit as st
import pandas as pd

#sys.path.append(str(Path(__file__).resolve().parent.parent))
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from utils import Functions as fn
# from pathlib import Path
# import sys 

#################################
# Page 3: Functions
#################################
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, cross_val_score

# Code for data preparation
def data_preparation(df_main):
    #df_main = dataset.copy()
    df_main.replace(["", "-", "0"], np.nan, inplace=True)   # Convert potential empty strings or placeholders into NaN
    df_main = df_main.dropna()       # Remove missing values
    df_main.reset_index(drop=True, inplace=True)

    # Apply Standardization (Z-score normalization)
    scaler = StandardScaler()
    df_main_scaled = pd.DataFrame(
        scaler.fit_transform(df_main),
        columns=df_main.columns,
        index=df_main.index
    )
    return df_main_scaled

import plotly.express as px

# Function to create a correlation matrix plot
def correlation_matrix(var, df):
    corr_matrix = df[var].corr()
    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale='RdBu_r',
        aspect="auto",
        labels=dict(color="Correlation"),
        zmin=-1, zmax=1
    )
    fig.update_layout(
        width=max(600, 40 * len(corr_matrix.columns)),
        height=max(600, 40 * len(corr_matrix.index)),
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis_tickangle=45
    )
    st.plotly_chart(fig, use_container_width=True)

# Function to select & fit with Support Vector Method (linear)
def SVM_linear_select_fit(X, y, n_features=2, split_data=False, test_size=0.3, random_state=42):
    # Optionally split dataset
    split_result = train_test_split(X, y, test_size=test_size, random_state=random_state) if split_data else (X, None, y, None)
    X_train, X_test, y_train, y_test = split_result

    # Hyperparameter tuning using GridSearchCV
    param_grid = {
        'C': [0.1, 1, 10],
        'epsilon': [0.01, 0.1, 0.2]
    }
    svr = GridSearchCV(SVR(kernel='linear'), param_grid, cv=3, scoring='r2', n_jobs=-1)
    svr.fit(X_train, y_train)
    best_svr = svr.best_estimator_

    # Apply Recursive Feature Elimination (RFE)
    rfe = RFE(estimator=best_svr, n_features_to_select=n_features)
    rfe_model = rfe.fit(X_train, y_train)

    # Transform the dataset to only include selected features
    X_train_rfe = rfe.transform(X_train)
    X_test_rfe = rfe.transform(X_test) if split_data else None
    X_rfe = rfe.transform(X) if split_data else None

    # Selected features
    selected_features = rfe.support_

    # Map indices to actual feature names if available
    if hasattr(X_train, 'columns'):
        select_feat_SVM = X_train.columns[rfe.support_]
    else:
        select_feat_SVM = [f"Feature_{i}" for i, sel in enumerate(rfe.support_) if sel]

    # Fit the model to the transformed data
    svr_model = best_svr.fit(X_train_rfe, y_train)

    # Return values based on split mode
    if split_data:
        return X_train, X_test, y_train, y_test, X_train_rfe, X_test_rfe, X_rfe, select_feat_SVM, svr_model
    else:
        return X_train_rfe, select_feat_SVM, svr_model

# Function to extract coefficients
def SVM_coefficients(svr_model):
    SVM_coeff = svr_model.coef_.flatten()
    return SVM_coeff

# Function to predict SVM
def predict_SVM(X_test_rfe, y_test, svr_model):
    y_pred = svr_model.predict(X_test_rfe)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"R² Score: {r2:.2f}")
    return y_pred

from sklearn.inspection import permutation_importance

# Function voor nonlinear SVM
def SVM_nonlinear_select_fit(X, y, n_features=2, split_data=False, test_size=0.3, random_state=42):
    
    # Optionally split dataset
    if split_data:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        st.write(f"Training Set Size: {X_train.shape[0]} rows")
        st.write(f"Test Set Size: {X_test.shape[0]} rows")
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
    st.write("Best Hyperparameters:", svr.best_params_)
    
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

    # Map indices to actual feature names if available
    if hasattr(X_train, 'columns'):
        select_feat_SVM = X_train.columns[selected_features]
        st.write("Selected Features Names (after permutation importance):", select_feat_SVM)

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

    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"R^2 Score: {r2:.2f}")
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

    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"R^2 Score: {r2:.2f}")

    return y_pred

# Function to visualize feature importances linear SVM (Streamlit compatible)
def visualize_feature_importances_SVM(select_feat_SVM, SVM_coeff):
    fig, ax = plt.subplots()
    ax.bar(select_feat_SVM, SVM_coeff)
    ax.set_xlabel('Feature Name')
    ax.set_ylabel('Coefficient Value')
    ax.set_title('Feature Importances (SVM)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

# Function to visualize feature importances EN (Streamlit compatible)
def visualize_feature_importances_EN(select_feat_EN, EN_best_coeff):
    fig, ax = plt.subplots()
    ax.bar(select_feat_EN, EN_best_coeff)
    ax.set_xlabel('Feature Name')
    ax.set_ylabel('Coefficient Value')
    ax.set_title('Feature Importances (Elastic Net)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

# Function to plot actual vs prediction results (Streamlit compatible)
def plot_pred_actual_results(y_test, y_pred):
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, color='blue', label='Predicted')
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.legend()
    ax.set_title('Regression Results')
    plt.tight_layout()
    st.pyplot(fig)

# Function for cross_validation (Streamlit compatible)
def cross_validation(select_method, x, y):
    scores = cross_val_score(select_method, x, y, cv=5, scoring='r2')
    st.write("Cross-validated R² scores:", scores)
    st.write("Mean R² score:", scores.mean())
    
#################################
#################################
#################################

st.header("📈Variable Selection Tool")
st.write("This page allows you to select the most relevant variables for your model.")

# File upload
uploaded_file = st.file_uploader("Upload your dataset (Excel or CSV file)", type=["csv", "xlsx"])
if uploaded_file:
    if uploaded_file.name.endswith('.xlsx'):
        table = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith('.csv'):
        table = pd.read_csv(uploaded_file)

    with st.expander("Preview of the Dataset"):
            st.dataframe(table)
            len_table = len(table)
            st.write(f"Number of rows: {len_table}")

    corr_columns = st.multiselect("**Select all variables for the Correlation matrix**", options=table.columns)

    # Ensure all selected independent variables are numeric
    non_numeric_columns = [col for col in corr_columns if not pd.api.types.is_numeric_dtype(table[col])]
    if non_numeric_columns:
        st.error(f"The following selected variables contain non-numerical values: **{', '.join(non_numeric_columns)}**. Please select only numeric variables.")
        st.stop()

#######################################
    # Display Correlation matrix
    st.subheader("Correlation Matrix of Selected Variables")
    if corr_columns:
        #selected_columns = list(corr_columns)
        df_main = table[corr_columns].copy()
        correlation_matrix(corr_columns, df_main)
        df_main = data_preparation(df_main)

#######################################
        # Allow user to finalize the independent variables (X) and dependent variable (Y) after viewing the correlation matrix
        st.subheader("Choose Final Variables for Selection Analysis")

        # Select dependent variable (Y)
        y_column = st.selectbox("Select the dependent variable (Y) for your analysis:", options=df_main.columns)
        
        # Ensure the selected dependent variable is numeric
        if not pd.api.types.is_numeric_dtype(df_main[y_column]):
            st.error(f"The selected dependent variable **{y_column}** contains non-numerical values. Please select a numeric variable.")
            st.stop()

        # Select independent variables (X), excluding the chosen Y
        x_columns = st.multiselect(
            "Select the final independent variables (X) to use for your analysis (minimum 2):",
            options=df_main.columns.drop(y_column)
        )

        # Warn if fewer than 2 independent variables are selected
        if len(x_columns) < 2:
            st.warning("Please select at least 2 independent variables (X) for analysis.")
            st.stop()
        
        # Ensure all selected independent variables are numeric
        non_numeric_columns = [col for col in x_columns if not pd.api.types.is_numeric_dtype(table[col])]
        if non_numeric_columns:
            st.error(f"The following selected independent variables contain non-numerical values: **{', '.join(non_numeric_columns)}**. Please select only numeric variables.")
            st.stop()

        if not y_column or not x_columns:
            st.warning("Please select a dependent variable (Y) and at least one independent variable (X) for analysis.")

#######################################
        # Drop missing values of x and y
        if y_column and x_columns:
            # Select only the relevant columns and drop rows with missing values
            selected_cols = [y_column] + x_columns
            df_main_cleaned = df_main[selected_cols].copy()
            df_main_cleaned = data_preparation(df_main_cleaned)

            # Prepare data
            y = df_main_cleaned[y_column]
            x = df_main_cleaned[x_columns]
            x = sm.add_constant(x)
            
            no_features = st.number_input(
                "Number of features to select:",
                min_value=2,
                max_value=len(x_columns),
                value=min(2, len(x_columns)),
                step=1
            )

        with st.expander("Preview Cleaned Dataset (without missing values)"):
            st.dataframe(df_main_cleaned)
            len_df_main_cleaned = len(df_main_cleaned)
            st.write(f"Number of rows: {len_df_main_cleaned}")

#######################################
# SVM linear: splitting/no splitting of dataset
#######################################
        st.header("Support Vector Method (SVM) - Linear Kernel")
        # Let user choose whether to split the dataset
        split_data = st.checkbox("Split dataset into train/test?", value=True)

        # Select features and fit SVM based on user choice
        if split_data:
            X_train, X_test, y_train, y_test, X_train_rfe, X_test_rfe, X_rfe, selected_lin_SVM, svr_model = SVM_linear_select_fit(
            x, y, n_features=no_features, split_data=True
            )

            # Extract SVM coefficients
            SVM_coeff = SVM_coefficients(svr_model)
            # Predict SVM results
            y_pred_SVM = predict_SVM(X_test_rfe, y_test, svr_model)
            # Visualize feature importances
            visualize_feature_importances_SVM(selected_lin_SVM, SVM_coeff)
            # Plot actual vs predicted results
            plot_pred_actual_results(y_test, y_pred_SVM)
            # Cross-validation results
            cross_validation(svr_model, X_rfe, y)

        else:
            X_train_rfe, selected_lin_SVM, svr_model = SVM_linear_select_fit(
            x, y, n_features=no_features, split_data=False
            )

            # Extract SVM coefficients
            SVM_coeff = SVM_coefficients(svr_model)
            # Predict SVM results
            y_pred_SVM = predict_SVM(X_train_rfe, y, svr_model)
            # Visualize feature importances
            visualize_feature_importances_SVM(selected_lin_SVM, SVM_coeff)
            # Plot actual vs predicted results
            plot_pred_actual_results(y, y_pred_SVM)
            # Cross-validation results
            cross_validation(svr_model, X_train_rfe, y)


#######################################
# SVM non-linear: splitting/no splitting of dataset
#######################################
        st.header("Support Vector Method (SVM) - Non-Linear Kernel")

        # Select features and fit SVM based on user choice
        if split_data:
            X_train, X_test, y_train, y_test, X_train_rfe, X_test_rfe, X_rfe, selected_nl_SVM, svr_nonlinear, perm_importance = SVM_nonlinear_select_fit(
            x, y, n_features=no_features, split_data=True
            )

            # Predict SVM results
            y_pred_SVM = predict_SVM_nonlinear(X_test_rfe, y_test, svr_nonlinear)
            # Visualize feature importances
            visualize_feature_nonlinear_SVM(selected_nl_SVM, perm_importance)
            # Plot actual vs predicted results
            plot_pred_actual_results(y_test, y_pred_SVM)
            # Cross-validation results
            cross_validation(svr_nonlinear, X_rfe, y)

        else:
            X_train_rfe, selected_nl_SVM, svr_nonlinear, perm_importance = SVM_nonlinear_select_fit(
            x, y, n_features=no_features, split_data=False
            )

            # Predict SVM results
            y_pred_SVM = predict_SVM_nonlinear(X_train_rfe, y, svr_nonlinear)
            # Visualize feature importances
            visualize_feature_nonlinear_SVM(selected_nl_SVM, perm_importance)
            # Plot actual vs predicted results
            plot_pred_actual_results(y, y_pred_SVM)
            # Cross-validation results
            cross_validation(svr_nonlinear, X_train_rfe, y)

#######################################
# Elastic Net: splitting/no splitting of dataset
#######################################
        st.header("Elastic Net Regression (Linear)")

        # Select features and fit Elastic Net based on user choice
        if split_data:
            # Fit Elastic Net on all features and get best estimator
            X_train, X_test, y_train, y_test, EN_hyperparams, EN_coeff_all, EN_coeff_nonzero, EN_model_all = elastic_net_fit_all(
            x, y, split_data=True
            )
            # Feature selection using RFE with best estimator
            selected_EN, X_train_rfe, X_test_rfe, X_rfe, EN_model_refit, EN_best_coeff = elastic_fit_select(
            x, y, EN_hyperparams, X_train, X_test, y_train, n_features=no_features
            )
            st.write(f"Training Set Size: {X_train.shape[0]} rows")
            st.write(f"Test Set Size: {X_test.shape[0]} rows")

            # Predict Elastic Net results
            y_pred_EN = elastic_predict(X_test_rfe, y_test, EN_model_refit)
            # Visualize feature importances
            visualize_feature_importances_EN(selected_EN, EN_best_coeff)
            # Plot actual vs predicted results
            plot_pred_actual_results(y_test, y_pred_EN)
            # Cross-validation results
            cross_validation(EN_model_refit, X_rfe, y)

        else:
            # Fit Elastic Net on all features and get best estimator
            EN_hyperparams, EN_coeff_all, EN_coeff_nonzero, EN_model_all = elastic_net_fit_all(
            x, y, split_data=False
            )
            # Feature selection using RFE with best estimator
            selected_EN, X_train_rfe, EN_model_refit, EN_best_coeff = elastic_fit_select(
            x, y, EN_hyperparams, n_features=no_features
            )

            # Predict Elastic Net results
            y_pred_EN = elastic_predict(X_train_rfe, y, EN_model_refit)
            # Visualize feature importances
            visualize_feature_importances_EN(selected_EN, EN_best_coeff)
            # Plot actual vs predicted results
            plot_pred_actual_results(y, y_pred_EN)
            # Cross-validation results
            cross_validation(EN_model_refit, X_train_rfe, y)

#######################################
            # Make a prediction for new data
            st.subheader("Make a Prediction with SVM Linear, Non-Linear, and Elastic Net")

            # Collect all unique columns needed for prediction
            all_predict_cols = set(selected_lin_SVM) | set(selected_nl_SVM) | set(selected_EN)
            st.markdown("**Prediction Input for All Models**")
            input_values = {}
            for col in all_predict_cols:
                input_values[col] = st.number_input(f"Enter value for **{col}**:", value=0.0, key=f"predict_{col}")

            # Prepare input dicts for each model
            input_values_linear = {col: input_values[col] for col in selected_lin_SVM}
            input_values_nonlinear = {col: input_values[col] for col in selected_nl_SVM}
            input_values_en = {col: input_values[col] for col in selected_EN}

            if st.button("Predict"):
                # SVM Linear prediction
                x_new_linear = pd.DataFrame([input_values_linear])
                x_new_linear = x_new_linear[selected_lin_SVM]
                #x_new_linear = data_preparation(x_new_linear)
                y_prediction_linear = svr_model.predict(x_new_linear)
                st.success(f"SVM Linear Prediction: **{y_column}: {y_prediction_linear[0]:.2f}**")

                # SVM Non-Linear prediction
                x_new_nonlinear = pd.DataFrame([input_values_nonlinear])
                x_new_nonlinear = x_new_nonlinear[selected_nl_SVM]
                #x_new_nonlinear = data_preparation(x_new_nonlinear)
                y_prediction_nonlinear = svr_nonlinear.predict(x_new_nonlinear)
                st.success(f"SVM Non-Linear Prediction: **{y_column}: {y_prediction_nonlinear[0]:.2f}**")

                # Elastic Net prediction
                x_new_en = pd.DataFrame([input_values_en])
                x_new_en = x_new_en[selected_EN]
                #x_new_en = data_preparation(x_new_en)
                y_prediction_en = EN_model_refit.predict(x_new_en)
                st.success(f"Elastic Net Prediction: **{y_column}: {y_prediction_en[0]:.2f}**")
                st.balloons()

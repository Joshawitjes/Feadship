########################################################################
# Page 2: Random Forest Regression
########################################################################
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import plot_tree

st.header("📈 Random Forest Regression Page")
st.write("Upload your dataset and predict outcomes using Random Forest regression.")

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

    # Variable selection
    y_column = st.selectbox("**Select dependent variable (y)**", table.columns)

    # Ensure the selected dependent variable is numeric
    if not pd.api.types.is_numeric_dtype(table[y_column]):
        st.error(f"The selected dependent variable **{y_column}** contains non-numerical values. Please select a numeric variable.")
        st.stop()

    x_columns = st.multiselect("**Select independent variables (X)**", options=table.columns.drop(y_column))

    # Ensure all selected independent variables are numeric
    non_numeric_columns = [col for col in x_columns if not pd.api.types.is_numeric_dtype(table[col])]
    if non_numeric_columns:
        st.error(f"The following selected independent variables contain non-numerical values: **{', '.join(non_numeric_columns)}**. Please select only numeric variables.")
        st.stop()

#######################################
    # Drop missing values of x and y
    if y_column and x_columns:
        table = table.dropna(subset=[y_column] + x_columns)
        with st.expander("Preview Cleaned Dataset (without missing values)"):
            st.dataframe(table)
            len_table = len(table)
            st.write(f"Number of rows: {len_table}")

        # Prepare data
        y = table[y_column]
        x = table[x_columns]

        # Split dataset
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

#######################################
        # Initialize Random Forest model
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

        # Fit the model
        rf_model = rf.fit(x_train, y_train)
        y_train_pred = rf_model.predict(x_train)
        y_test_pred = rf_model.predict(x_test)

        # Feature importances
        st.subheader("Feature Importances")
        importances = rf_model.feature_importances_
        feature_df = pd.DataFrame({
            'Feature': x_train.columns,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        # Create figure in aesthetic style
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_df, palette='viridis', ax=ax)
        ax.set_title("Ranked Feature Importances", fontsize=16, weight='bold', pad=15)
        ax.set_xlabel("Importance Score", fontsize=12)
        ax.set_ylabel("Feature", fontsize=12)
        ax.xaxis.grid(True)  # Add horizontal gridlines for readability
        ax.set_axisbelow(True)
        st.pyplot(fig)
        st.caption("⚠️ Note: The x-axis represents the Importance Score, which is NOT equivalent to the coefficients from a regression analysis.")

#######################################
        # Plot Actual vs Predicted values
        st.subheader("Actual vs Predicted (Test Set)")

        # Create figure in aesthetic style
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            y_test, y_test_pred,
            alpha=0.6,
            edgecolor='k',
            linewidth=0.5,
            s=70,
            c=y_test_pred,  # Optional: color by prediction
            cmap='viridis'
        )
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
        ax.set_title('Actual vs Predicted Values (Test Set)', fontsize=16, weight='bold', pad=15)
        ax.set_xlabel('Actual Values', fontsize=12)
        ax.set_ylabel('Predicted Values', fontsize=12)
        ax.legend(loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.6)
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Predicted Value Intensity')
        st.pyplot(fig)

        # Add predictions and residuals to the dataset dynamically based on selected y_column
        st.subheader("Predictions and Residuals")
        table[f"Predicted_{y_column}_RF"] = rf_model.predict(x)
        table[f"Residual_{y_column}_RF"] = table[y_column] - table[f"Predicted_{y_column}_RF"]
        table[f"Residual_{y_column}_RF_%"] = (table[f"Residual_{y_column}_RF"] / table[y_column]) * 100
        st.dataframe(table[[f"{y_column}", f"Predicted_{y_column}_RF", f"Residual_{y_column}_RF", f"Residual_{y_column}_RF_%"]])

#######################################
        # Performance metrics
        st.subheader("Model Performance")
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)
        rmse_test = np.sqrt(mse_test)
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"R² (test set): {r2_test:.2f}")
            st.write(f"R² (training set): {r2_train:.2f}")
            st.write("")
            st.write("")
            st.write(f"Mean Squared Error (MSE test set): {mse_test:.2f}")
            st.write("")
            st.write("")
            st.write(f"Root Mean Squared Error (RMSE test set): {rmse_test:.2f}")
        with col2:
            with st.expander("R² TEST vs TRAINING setℹ️"):
                st.caption("Indicates the proportion/percentage of variance explained by the chosen explanatory variables. Greater values indicate a better fit.")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            with st.expander("MSE ℹ️"):
                st.caption("Measures the average squared difference between actual and predicted values.")
                st.caption("**Be cautious**: this is an absolute value which should NOT be interpreted as a standalone metric. Compare this value to the MSE of another model to determine the model with the better fit. The lowest MSE provides the best fit.")
            with st.expander("RMSE ℹ️"):
                st.caption("Represents the standard deviation of the prediction errors. The lower, the more accurate the model.")


#######################################
        # Cross-validation
        st.subheader("Cross-Validation Results")
        cv_scores = cross_val_score(rf, x_train, y_train, cv=5, scoring='r2')
        avg_cv_score = np.mean(cv_scores)
        with st.expander("R² Scores (5-Fold Cross-Validation) ℹ️"):
            # Display R² scores
            st.table(pd.DataFrame({
                "Fold": [f"Fold {i+1}" for i in range(len(cv_scores))],
                "R² Score": [f"{score:.4f}" for score in cv_scores]
            }))
        st.write(f"**Average R² Score:** {avg_cv_score:.4f}")

        # Adjusted R-squared calculation
        n = x_train.shape[0]  # Number of observations in the training dataset
        p = x_train.shape[1]  # Number of predictors in the training dataset
        adjusted_r2_scores = [1 - ((1 - r2) * (n - 1) / (n - p - 1)) for r2 in cv_scores]
        avg_adjusted_r2 = np.mean(adjusted_r2_scores)
        with st.expander("Adjusted R² Scores (5-Fold Cross-Validation) ℹ️"):
            # Display Adjusted R² scores
            st.table(pd.DataFrame({
                "Fold": [f"Fold {i+1}" for i in range(len(adjusted_r2_scores))],
                "Adjusted R² Score": [f"{score:.4f}" for score in adjusted_r2_scores]
            }))
        st.write(f"**Average Adjusted R² Score:** {avg_adjusted_r2:.4f}")


#######################################
        # Make a prediction for new data
        st.header("Make a Prediction")
        input_values = {}
        for col in x_columns:
            input_values[col] = st.number_input(f"Enter value for {col}:", value=0.0)

        if st.button("Predict"):
            x_new = pd.DataFrame([input_values])
            y_prediction = rf.predict(x_new)  # Random Forest does not require a constant
            st.success(f"Prediction Completed!: **{y_column}: {y_prediction[0]:.2f}**")
            st.balloons()

#######################################
        # Visualize a single decision tree
        st.subheader("Example Decision Tree Visualization")
        st.write("An example of 1 of the 100 underlying decision trees that the Random Forest model creates on the background. Keep in mind that this is just meant as intuition for the ones that are interested in what the model actually does.")
        st.write("No relevant information can be deducted from this figure!")
        fig, ax = plt.subplots(figsize=(15, 10))
        plot_tree(rf.estimators_[0], feature_names=x.columns, filled=True, rounded=True, ax=ax)
        st.pyplot(fig)
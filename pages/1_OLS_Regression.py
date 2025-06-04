import streamlit as st
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Add a page configuration for multi-page navigation
st.set_page_config(page_title="OLS Regression App")

########################################################################
# Page 1: OLS Regression
########################################################################

st.markdown("## 📊 OLS Regression for Prediction")
st.markdown("Upload your dataset and predict outcomes using multivariable linear regression.")

# Upload dataset
uploaded_file = st.file_uploader("Upload your dataset (Excel or CSV file):", type=["xlsx", "csv"])
if uploaded_file:
    if uploaded_file.name.endswith('.xlsx'):
        table = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith('.csv'):
        table = pd.read_csv(uploaded_file)

    with st.expander("Preview of the Dataset"):
            st.dataframe(table)
            len_table = len(table)
            st.write(f"Number of rows: {len_table}")

    # Select target variable
    y_column = st.selectbox("**Select your dependent variable (y)**", table.columns)

    # Ensure the selected dependent variable is numeric
    if not pd.api.types.is_numeric_dtype(table[y_column]):
        st.error(f"The selected dependent variable **{y_column}** contains non-numerical values. Please select a numeric variable.")
        st.stop()

    # Select explanatory variables
    x_columns = st.multiselect("**Select your independent variables (X)**", options=table.columns.drop(y_column))

    # Ensure all selected independent variables are numeric
    non_numeric_columns = [col for col in x_columns if not pd.api.types.is_numeric_dtype(table[col])]
    if non_numeric_columns:
        st.error(f"The following selected independent variables contain non-numerical values: **{', '.join(non_numeric_columns)}**. Please select only numeric variables.")
        st.stop()

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
        x = sm.add_constant(x)

        # Run Regression
        model = sm.OLS(y, x)
        results = model.fit(cov_type="HC0")

        # Display regression summary
        st.subheader("Regression Summary")
        st.code(results.summary().as_text())

        summary_df = pd.DataFrame({
            "coef": results.params,
            "std err": results.bse,
            "P>|t|": results.pvalues
        })
        summary_df["Significant"] = summary_df["P>|t|"].apply(lambda p: "Yes" if p < 0.05 else "No")
        st.dataframe(summary_df.style.format(precision=3))

        # Residuals and MSE
        st.subheader("Model Performance")
        predicted_values = results.predict(x)
        mse = mean_squared_error(y, predicted_values)
        rmse = np.sqrt(mse)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"Adjusted R²: {results.rsquared_adj:.2f} or {results.rsquared_adj * 100:.2f}%")
            st.write("")
            st.write("")
            st.write(f"Mean Squared Error (MSE): {mse:.2f}")
            st.write("")
            st.write("")
            st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

        with col2:
            with st.expander("Adj. R²ℹ️"):
                st.caption("Indicates the proportion/percentage of variance explained by the chosen explanatory variables, adjusted for the number of predictors. Greater values indicate a better fit.")
            with st.expander("MSE ℹ️"):
                st.caption("Measures the average squared difference between actual and predicted values.")
                st.caption("**Be cautious**: this is an absolute value which should NOT be interpreted as a standalone metric. Compare this value to the MSE of another model to determine the model with the better fit. The lowest MSE provides the best fit.")
            with st.expander("RMSE ℹ️"):
                st.caption("Represents the standard deviation of the prediction errors. The lower, the more accurate the model.")

        # Plot actual vs predicted values
        st.subheader("Actual vs Predicted Values")

        # Create figure in aesthetic style
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            y, predicted_values,
            alpha=0.6,
            edgecolor='black',
            linewidth=0.5,
            s=70,
            c=predicted_values,
            cmap='viridis'
        )
        ax.plot(
            [y.min(), y.max()],
            [y.min(), y.max()],
            'r--',
            lw=2,
            label='Perfect Prediction'
        )
        ax.set_title("Actual vs Predicted Values", fontsize=16, weight='bold', pad=15)
        ax.set_xlabel("Actual Values", fontsize=12)
        ax.set_ylabel("Predicted Values", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc="upper left")
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Predicted Value Intensity')
        st.pyplot(fig)

        # Add predictions and residuals to the dataset dynamically based on selected y_column
        st.subheader("Predictions and Residuals")
        table[f"Predicted_{y_column}_OLS"] = predicted_values
        table[f"Residual_{y_column}_OLS"] = table[y_column] - table[f"Predicted_{y_column}_OLS"]
        table[f"Residual_{y_column}_OLS_%"] = (table[f"Residual_{y_column}_OLS"] / table[y_column]) * 100
        st.dataframe(table[[f"{y_column}", f"Predicted_{y_column}_OLS", f"Residual_{y_column}_OLS", f"Residual_{y_column}_OLS_%"]])

        # Input fields for prediction
        st.header("Make a Prediction")
        input_values = {}
        for col in x_columns:
            input_values[col] = st.number_input(f"Enter value for {col}:", value=0.0)

        # Prediction logic
        if st.button("Predict"):
            prediction = results.params['const']
            for col in x_columns:
                prediction += results.params[col] * input_values[col]
            st.success(f"Prediction completed! **{y_column}: {prediction:.2f}**")
            st.balloons()

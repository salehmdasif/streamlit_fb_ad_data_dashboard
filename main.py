import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm  # for hypothesis testing
import matplotlib.pyplot as plt
from scripts.script import winningAdSelector, highlight_column

# Set the app to wide mode
st.set_page_config(layout="wide")
st.title("Facebook Ad Data Analysis Dashboard")

# Sidebar content
with st.sidebar:
    st.header("About App")
    st.write("This app analyzes uploaded data using linear regression and visualizes relevant insights.")
    st.markdown("© [Mohammad Asif](https://www.linkedin.com/in/salehmdasif)")
    st.subheader("\nHow to Use the App")
    st.write("""
        Use dataset only after cleaning.
        \nUpload your dataset using the file uploader in the sidebar.
        \nSelect the target variable and desired features for analysis.
        \nExplore correlations between variables and generate visualizations.
        \nReview the linear regression analysis results and hypothesis testing outputs to gain insights into feature impacts.
    """)

# Placeholder for DataFrame
df = None

# Create columns for file upload and link input
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Choose a CSV/XLS file", type=["csv", "xls", "xlsx"])

    # Process the uploaded file if it is uploaded
    if uploaded_file is not None:
        st.write("File uploaded successfully...")

        try:
            # Read the file based on its type
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(uploaded_file)

            # Show the name of the uploaded file in the sidebar
            with st.sidebar:
                st.write(f"Uploaded File: **{uploaded_file.name}**")  # Display uploaded file name

        except Exception as e:
            st.error(f"An error occurred while reading the file: {e}")

    with col2:
        data_link = st.text_input("Or please provide link to your data file ", key="data_link")
        # Displaying as plain text using HTML
        link_text = "https://ravelweb.com/data/cleaned_ad_data.csv"
        st.markdown(
            f"Or try with this link (copy and paste) 👉 <span style='pointer-events: none; color: black;'>{link_text}</span>",
            unsafe_allow_html=True)

        # Check if there was an update in the input and the button is clicked or data_link changed
        if st.button("Get Data from Link") or data_link:
            if data_link:  # Check if a data link is provided
                st.write("Link provided for data upload...")

                try:
                    # Read the data from the provided link
                    if data_link.endswith('.csv'):
                        df = pd.read_csv(data_link)
                    elif data_link.endswith(('.xls', '.xlsx')):
                        df = pd.read_excel(data_link)
                    else:
                        st.error("Please provide a valid URL ending with .csv, .xls, or .xlsx.")

                    # Successfully loaded from link
                    st.write(f"Data loaded successfully")

                    with st.sidebar:
                        st.write(f"Uploaded file with provided link: **{data_link}**")  # Display uploaded file name

                except Exception as e:
                    st.error(f"An error occurred while reading the data from the link: {e}")
            else:
                st.error("Please provide a link to the data file.")


# Define columns for the select box
if df is not None:  # Check if df is available

    # Set 'serial' as index if it exists in the DataFrame
    if 'serial' in df.columns:
        df.set_index('serial', inplace=True)

    # General columns
    columns = df.columns.tolist()
    # Numeric columns
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
else:
    columns = []
    numeric_columns = []

# Check if df is available
if df is not None:
    # Data preview
    st.subheader("Data preview")
    st.write(f"Data shape: {df.shape[0]} Rows & {df.shape[1]} Columns")
    st.write(df.head())

    # Correlation Analysis
    st.subheader("Correlation analysis")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        # Dropdown for selecting x-axis from numeric columns
        x_axis = st.selectbox("Select X Axis (Numeric)", numeric_columns, key="x_axis_select")
    with col2:
        # Dropdown for selecting y-axis, excluding the selected x_axis
        y_axis = st.selectbox("Select Y Axis (Numeric)", [col for col in numeric_columns if col != x_axis],
                              key="y_axis_select")

    col1, col2 = st.columns(2)
    with col1:
        # Button to generate correlation calculation with a unique key
        if st.button("Show Correlation", key="show_correlation_button"):
            # Calculate correlation
            correlation_value = df[x_axis].corr(df[y_axis])

            # Display the correlation value
            st.write(f"Correlation between `{x_axis}` and `{y_axis}`: **{correlation_value:.2f}**")

            # Determine if the correlation is positive, negative or zero
            if correlation_value > 0:
                st.success("There is a **positive correlation**.")
            elif correlation_value < 0:
                st.warning("There is a **negative correlation**.")
            else:
                st.info("There is **no correlation**.")

    # Data summary
    st.subheader("Data summary")
    st.write(df.describe())

    # Top 10 winning result
    selector = winningAdSelector(df)
    best_adds = selector.get_winning_ads(top_n=10)
    # Initialize the selector and get winning ads only after df is not None
    st.subheader("Top 10 winning ads (according to some predefined matrix)")
    st.write(best_adds)

    # Filtering columns with it's unique value
    st.subheader("Filtering columns with it's unique value")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        selected_column = st.selectbox("Select column to filter by", columns)
    with col2:
        if selected_column:  # Ensure selected_column is available
            unique_value = df[selected_column].unique()
            selected_value = st.selectbox("Select unique value", unique_value)

    # Filter the DataFrame based on the selected column and value
    filtered_df = df[df[selected_column] == selected_value]
    st.write(filtered_df)

    # Generate Line Plot
    if 'x_column' not in st.session_state:
        st.session_state.x_column = columns[0] if columns else None
    if 'y_column' not in st.session_state:
        st.session_state.y_column = columns[1] if len(columns) > 1 else None

    st.subheader("Select x-axis and y-axis to generate line plot")
    col1, col2, col3, col4 = st.columns(4)

    with col1:  # Select X-axis column
        st.session_state.x_column = st.selectbox("Select x-axis column", columns,
                                                 index=columns.index(st.session_state.x_column))

    with col2:  # Select Y-axis column excluding the selected X-axis column
        options_for_y = [col for col in columns if col != st.session_state.x_column]
        st.session_state.y_column = st.selectbox("Select y-axis column", options_for_y, index=options_for_y.index(
            st.session_state.y_column) if st.session_state.y_column in options_for_y else 0)

    if st.button("Generate Plot"):  # Button to generate the plot
        if st.session_state.x_column == st.session_state.y_column:
            st.error("Error: X-axis and Y-axis cannot be the same.")
        else:
            # Assuming filtered_df is your DataFrame
            st.line_chart(filtered_df.set_index(st.session_state.x_column)[st.session_state.y_column])

    if st.button("Reset Selections"):  # Reset button to clear the selections
        # Clear selections by resetting session state variables to default
        st.session_state.x_column = columns[0] if columns else None
        st.session_state.y_column = columns[1] if len(columns) > 1 else None

    # Top 10 Ads with Filtered Column
    st.subheader("Top 10 ads with filtered column")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        filtered_column = st.selectbox("Filter top 10 ads by", numeric_columns)
    with col2:
        sort_order = st.selectbox("Select sorting order", ["High to Low", "Low to High"])
    # Check if the selected column has valid data to display
    if filtered_column:
        ascending = sort_order == "Low to High"  # Determine sort order based on user selection
        # Sort and display top 10 ads
        top_10 = df.sort_values(by=filtered_column, ascending=ascending).head(10)
        st.write(f"Top 10 ads with {filtered_column}:")

        # Highlight the specified column dynamically using the imported function
        styled_top_10 = (
            top_10[columns]
            .style
            .apply(highlight_column, column_to_highlight=filtered_column, axis=1)
            # Format only numeric values
            .format(na_rep="-",  # Display a hyphen for NaN values
                    formatter=lambda x: f"{x:.2f}" if pd.api.types.is_numeric_dtype(x) else x)
        )
        st.dataframe(styled_top_10)

        # Select another column (matrix) to plot
        st.subheader("Select a column to plot")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            plot_column = st.selectbox("Select a column to plot", numeric_columns)

        # Initialize session state variables to control chart visibility
        if 'show_first_chart' not in st.session_state:
            st.session_state.show_first_chart = False
        if 'show_second_chart' not in st.session_state:
            st.session_state.show_second_chart = False

        # Check if plot_column is selected
        try:
            if plot_column:
                if plot_column == filtered_column:
                    st.write("Please select another column, as plotting cannot be the highlighted column.")

                # Create two columns for the charts
                col5, col6 = st.columns(2)
                # First chart logic
                with col5:
                    st.write(f"Chart of {plot_column} for top 10 ads")  # Title for the first chart

                    # Button to generate the first chart
                    if st.button("Generate Chart for Top 10 Ads"):
                        st.session_state.show_first_chart = True  # Set the flag to show the first chart
                        st.session_state.first_chart_data = top_10[plot_column]  # Store the data for the first chart

                    # If the first chart is to be shown, display it
                    if st.session_state.show_first_chart:
                        st.bar_chart(st.session_state.first_chart_data)  # Display the first chart

                # Second chart logic
                with col6:
                    st.write(f"Chart of {plot_column} vs. {filtered_column}")  # Title for the second chart

                    # Button to generate the second chart
                    if st.button("Generate Chart Comparing to Highlighted Column"):
                        st.session_state.show_second_chart = True  # Set the flag to show the second chart
                        # Prepare data for second chart
                        plot_data = top_10[[filtered_column, plot_column]].reset_index(drop=True)
                        st.session_state.second_chart_data = plot_data  # Store the data for the second chart

                    # If the second chart is to be shown, display it
                    if st.session_state.show_second_chart:
                        # Check if the column exists in the DataFrame before trying to plot
                        if filtered_column in st.session_state.second_chart_data.columns:
                            st.bar_chart(st.session_state.second_chart_data.set_index(filtered_column)[
                                             plot_column])  # Display the second chart
                        else:
                            st.error(f"Error: The column '{filtered_column}' does not exist in the data.")

        except ValueError as e:
            st.error(e)  # Display error message to the user

        except KeyError as e:
            # st.write(f"Check if the selected column {str(e)} is valid.")
            st.write("")

    # Linear Regression on Full Dataset
    st.subheader("Linear regression analysis")
    # Get only numeric columns for target and features
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()

    col1, col2 = st.columns([1, 3])
    with col1:
        # User selects target variable
        target = st.selectbox("Select Target Variable", numeric_columns, key="target_select")

        # Get features excluding the selected target
        features = [col for col in numeric_columns if col != target]
    with col2:
        # User selects features
        selected_features = st.multiselect("Select Features", features)

    # Button to perform linear regression if features are selected
    if st.button("Run Linear Regression"):
        if not selected_features:
            st.error("Please select at least one feature.")
        else:
            # Prepare data for regression
            x = df[selected_features]
            y = df[target]

            # Handle missing values (optional)
            x = x.fillna(x.mean())
            y = y.fillna(y.mean())

            # Standardize features
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(x)

            # Fit Linear Regression model
            model = LinearRegression()
            model.fit(x_scaled, y)

            # Get coefficients
            coef_df = pd.DataFrame({
                'feature': selected_features,
                'coefficient': model.coef_
            }).sort_values(by='coefficient', ascending=False)

            col1, col2 = st.columns(2)
            with col1:
                # Display coefficients
                st.write("Feature Importance (Coefficients)")
                st.write(coef_df.round(2))
            with col2:
                # Visualization: Bar chart of coefficients
                st.write("Feature Importance Visualization")
                st.scatter_chart(coef_df.set_index('feature')['coefficient'])

                # Hypothesis Testing using statsmodels
                X_with_const = sm.add_constant(x_scaled)  # Adding constant for intercept
                model_sm = sm.OLS(y, X_with_const).fit()  # Fit using statsmodels

            # Hypothesis Testing for each coefficient
            st.subheader("Hypothesis Testing for Coefficients")
            # Interpretation
            st.write("""
            if the p-value is less than 0.05, we reject the null hypothesis, 
            indicating that the coefficient is statistically significant. 
            """)

            st.write("""
            Otherwise, we fail to reject the null hypothesis, suggesting that there is not 
            enough evidence to conclude the coefficient significantly differs from zero, 
            indicating that the feature may not strongly influence the target variable.
            """)

            # Get coefficients, p-values, standard errors, and t-statistics
            p_values = model_sm.pvalues[1:]  # Excluding constant
            std_errors = model_sm.bse[1:]  # Standard errors for coefficients
            t_statistics = model_sm.tvalues[1:]  # T-statistics for coefficients
            conf_int = model_sm.conf_int().iloc[1:]  # Excluding constant

            # Create a DataFrame for hypothesis testing results with rounded values
            hypothesis_df = pd.DataFrame({
                'feature': selected_features,
                'coefficient': model_sm.params[1:],  # Exclude constant
                'p-value': p_values.round(6),
                'standard error': std_errors.round(6),
                't-statistic': t_statistics.round(6),
                'lower ci': conf_int.iloc[:, 0].round(6),
                'upper ci': conf_int.iloc[:, 1].round(6)
            })

            # Display hypothesis testing results
            st.write(hypothesis_df)

else:
    st.write("Waiting for file upload")

import streamlit as st
import pandas as pd
from datetime import datetime,timedelta
from streamlit import session_state as ss
def add_logo():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                background-image: url(https://kmkconsultinginc.com/wp-content/uploads/2020/12/KMK-Logo.png);
                background-repeat: no-repeat;
                padding-top: 120px;
                background-position: 20px 20px;
            }
            [data-testid="stSidebarNav"]::before {
                content: "Menu";
                margin-left: 20px;
                margin-top: 20px;
                font-size: 30px;
                position: relative;
                top: 100px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

add_logo()
# Main header text
main_header_text = "Data Upload"
# Display main header centered with color
st.markdown(f"<h1 style='text-align: center;color:#F12977;'>{main_header_text}</h1>", unsafe_allow_html=True)

st.write('---')
file_path = 'Calendarisation.xlsx'


# Function to export the input data format to Excel with dynamic column names
def export_input_format(num_rows, product_names, num_territories):
    # Create column names for products and territories
    sales_data = []
    for product in product_names:
        for territory in range(1, num_territories + 1):
            sales_data.append([product, f't{territory}',0,0,0])  # Assuming 123 for week sales for demonstration

    empty_df = pd.DataFrame(sales_data, columns=['Product', 'Territory', 'mm/dd/yyyy','mm/dd/yyyy','mm/dd/yyyy'])
    empty_df.fillna(0,inplace=True)
    
    # Export DataFrame to Excel
    with pd.ExcelWriter('input_format.xlsx') as writer:
        empty_df.to_excel(writer, sheet_name='Input_Format', index=False)
    st.success("The template has been created, Please click the Download Template button!")
    st.info("🤖 Please enter the dates in the columns and their corresponding sales below them.")
    # Provide a download link to the user
    with open('input_format.xlsx', 'rb') as f:
        data = f.read()
    st.download_button(
        label="Download Template",
        data=data,
        file_name='input_format.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    )

# Download Input Data Template section
with st.expander("Download Input Data Template 👇"):
    st.write("Please specify the number of products and territories.")
    num_columns = st.number_input("Number of Products", min_value=1, step=1, value=2)
    num_territories = st.number_input("Number of Territories", min_value=1, step=1, value=2)

    # Placeholder for product names input boxes
    product_name_placeholders = [st.empty() for _ in range(num_columns)]

    product_names = []
    for i, placeholder in enumerate(product_name_placeholders):
        product_name = placeholder.text_input(f"Product {i+1} name", f"Product {i+1}")
        product_names.append(product_name)
    export_input_format(num_columns, product_names, num_territories)
    

# Upload Excel File section
st.subheader("Upload Excel File:")
st.warning("Please use the input data template before uploading the file!")
uploaded_file = st.file_uploader(label="", type=["xlsx", "xls"])
df2 = pd.read_excel(file_path,sheet_name=0)

if uploaded_file is not None:
    # Read Excel file
    df1 = pd.read_excel(uploaded_file, sheet_name=0)
    # df2 = pd.read_excel(uploaded_file,sheet_name=1)        
    df = df1
    # Melt the DataFrame
    melted_df = df.melt(id_vars=['Product', 'Territory'], var_name='Dateperiod', value_name='Value')
    # Combine 'Product' and 'terr' columns into a single index
    melted_df['Dataperiod'] = melted_df['Product'] + ' ' + melted_df['Territory']

    # Drop unnecessary columns
    melted_df.drop(columns=['Product', 'Territory'], inplace=True)

    melted_df = melted_df.pivot(index='Dataperiod', columns='Dateperiod', values='Value').reset_index()
    melted_df.rename_axis(None, axis=1).rename_axis(None, axis=0)
    # Data preprocessing
    df1 = melted_df.T
    df1.columns = df1.iloc[0]
    df1 = df1[1:]
    df1.reset_index(inplace=True)
    df1.rename(columns={'Dateperiod': 'Dataperiod'}, inplace=True)
    df2 = df2.T
    df2.columns = df2.iloc[0]
    df2 = df2[1:]
    df2.reset_index(inplace=True)
    df2.rename(columns={'index': 'Dataperiod'}, inplace=True)

    df1 = df1.rename_axis(index=None, columns=None)
    df2 = df2.rename_axis(index=None, columns=None)
    df1.iloc[:, 1:] = df1.iloc[:, 1:].astype(float)
    df2.iloc[:, 1:] = df2.iloc[:, 1:].astype(float)

    # Store data in session state
    ss.file_uploaded = True
    ss.uploaded_file = df1
    ss.uploaded_file1 = df2
    ss.uploaded_filed = df
    
# Display uploaded file and summary
if hasattr(ss, 'file_uploaded') and ss.file_uploaded:
    # Display success message
    st.success("File has been uploaded.")
    # Display sales data (wide)
    # Highlight the maximum value in each column
    st.subheader('Sales data')
    st.dataframe(ss.uploaded_filed)
    # Summary section
    products = ss.uploaded_filed['Product'].nunique()
    territories = ss.uploaded_filed['Territory'].nunique()  # Excluding the 'Date' column
    start_date = ss.uploaded_file['Dataperiod'].min().strftime('%Y-%m-%d')
    # Convert string to datetime object
    dt_object = datetime.strptime(start_date, '%Y-%m-%d')
    dt_sns = dt_object.replace(microsecond=0)        
    ss['start_date']= dt_sns

    end_date = ss.uploaded_file['Dataperiod'].max().strftime('%Y-%m-%d')
    dt_object = datetime.strptime(end_date, '%Y-%m-%d')
    dt_ens = dt_object.replace(microsecond=0)        
    ss['end_date']=dt_ens
    
    future_start_date = ss.uploaded_file['Dataperiod'].max().strftime('%Y-%m-%d')
    dt_object = datetime.strptime(future_start_date, '%Y-%m-%d')
    dt_object += timedelta(days=7)  # Adding 7 days, approximately 1 week
    dt_fss = dt_object.replace(microsecond=0)
    ss['future_start_date'] = dt_fss

    future_end_date = ss.uploaded_file['Dataperiod'].max().strftime('%Y-%m-%d')
    dt_object = datetime.strptime(future_end_date, '%Y-%m-%d')
    dt_object += timedelta(days=90)  # Adding 90 days, approximately 3 months
    dt_fns = dt_object.replace(microsecond=0)
    ss['future_end_date'] = dt_fns
    
    ss['summary_data'] = pd.DataFrame({
        "Metrics": ["Number of Products", "Number of Territories", "Baseline Start Date", "Baseline End Date"],
        "Values": [products, territories, start_date, end_date]
    })
    
    # Display summary data with spinner
    with st.spinner("Loading summary data..."):
        # Once loading is complete, display the summary data
        st.write("### Summary")
        st.table(ss['summary_data'])
    st.write("### Data Validation")

    with st.spinner("Validating data..."):  # Display spinner while checking for outliers
        df1 = ss.uploaded_file
        # Check for NaNs and zeroes
        nan_zeroes_found = df1.isnull().values.any() or (df1 == 0).any().any()
    
        if nan_zeroes_found:
            st.warning("Missing values or zeros found in the data. Please fill them and reupload the file.")
        else:
            st.success("No missing values or zeros found in the data.")

            df1['Dataperiod'] = pd.to_datetime(df1['Dataperiod'])
            df1.sort_values(by='Dataperiod', inplace=True)
            # Calculate the differences between consecutive dates
            date_diffs = df1['Dataperiod'].diff().dt.days
            if date_diffs.nunique() == 1:
                st.success("The dates are consistent with {} days between them.".format(date_diffs.iloc[1]))
            else:
                st.error("The dates are inconsistent.")
                # Display the rows with inconsistent dates
                inconsistent_rows = df1[date_diffs != date_diffs.iloc[1]]
                st.write("Rows with inconsistent dates:")
                st.write(inconsistent_rows)

        
            outliers_found = False  # Flag to track if any outliers are found
            total_columns = len(df1.columns[1:])
            for i, product_column in enumerate(df1.columns[1:], 1):
                product_sales = df1[product_column]
                z_scores = (product_sales - product_sales.mean()) / product_sales.std()

                # Identify outliers using a threshold (e.g., z-score > 3 or z-score < -3)
                outliers = z_scores[(z_scores > 3) | (z_scores < -3)]

                if not outliers.empty:
                    outliers_found = True  # Set the flag to True if outliers are found
                    st.warning(f"Outliers detected in {product_column}: {outliers}")

                    # Ask the user if they want to normalize the outliers
                     # Getting the default value for the radio button
                    default_yes_or_no = ss.get('normalize_option', 'Yes')

                    # Defining options for the radio button
                    options = ["Yes", "No"]

                    # Setting the initial value if it exists in the options
                    if default_yes_or_no not in options:
                        default_yes_or_no = options[0]

                    # Creating a radio button for Yes or No with default value
                    normalize_option = st.radio(f"Do you want to normalize outliers in {product_column}?", options=options, index=options.index(default_yes_or_no))
                    ss.normalize_option = normalize_option

                    if normalize_option == 'Yes':
                        # Normalize outliers (replace them with the mean)
                        df1.loc[outliers.index, product_column] = product_sales.mean()
                        st.info(f"Outliers in {product_column} normalized.")

            # If no outliers are found in any column, display a single message
            if not outliers_found:
                st.success("No outliers found in the data.")

            if st.button('Next'):
                st.switch_page('pages/2_⚙️Model Parameters.py')



import streamlit as st
from datetime import datetime, timedelta
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
# Content for Model Parameters tab
# Main header text
main_header_text = "Modelling"
st.markdown(f"<h1 style='text-align: center;color:#30A1D8;'>{main_header_text}</h1>", unsafe_allow_html=True)
st.write('---')
if ss.uploaded_file is not None:
    # Initialize session state if it doesn't exist
    if 'parameters' not in ss:
        ss['parameters'] = {
            'Baseline Start Date': ss['start_date'],
            'Baseline End Date': ss['end_date'],
            'Forecast Start Date': ss['future_start_date'],
            'Forecast End Date': ss['future_end_date'],
            # 'Client_Forecast':0
            }
    if 'normalize_option' not in ss:
        ss['normalize_option'] = None
    if 'selected_models' not in ss:
        ss['selected_models'] = None
    if 'model_params' not in ss:
        ss['model_params'] = {
        'Exponential' : ['add','add',2,0],
        'Sarima' : [None,None]
        }
            
            
    df1 = ss.uploaded_file
    
    st.write("## Dates Selection")
    
    
    # Get the minimum and maximum date values from the DataFrame
    min_date = df1['Dataperiod'].min()
    max_date = df1['Dataperiod'].max()
    
    # Ensure default date is within the max_date
    
    # Display the date selection inputs side by side
    st.subheader("Baseline Period")
    b1,b2 = st.columns(2)
    
    baseline_start_date = b1.date_input("Start Date", min_value=min_date, max_value=max_date, value=ss.parameters['Baseline Start Date'])
    ss.parameters['Baseline Start Date'] = baseline_start_date
    baseline_end_date = b2.date_input("End Date", min_value=baseline_start_date, max_value=max_date, value=ss.parameters['Baseline End Date'])
    ss.parameters['Baseline End Date'] = baseline_end_date

    st.subheader("Forecast Period")
    a1,a2=st.columns(2)
    forecast_start_date = a1.date_input("Start Date", min_value=baseline_start_date, value=ss.parameters['Forecast Start Date'])
    ss.parameters['Forecast Start Date'] = forecast_start_date
    forecast_end_date = a2.date_input("End Date", min_value=forecast_start_date, value=ss.parameters['Forecast End Date'])
    ss.parameters['Forecast End Date'] = forecast_end_date
    # Client_Forecast = st.number_input("Product 1 Client Forecast ", min_value=0, value=ss.parameters['Client_Forecast'])
    # ss.parameters['Client_Forecast'] = Client_Forecast

   # Getting the default value for the radio button
    default_selected_type = ss.parameters.get('selected_type', 'Weekly')

    # Defining options for the radio button
    options = ["Weekly", "Quarterly"]

    # Setting the initial value if it exists in the options
    if default_selected_type not in options:
        default_selected_type = options[0]

    # Creating a radio button for Yes or No with default value
    selected_type = a1.radio("Select Forecast Type", options=options, index=options.index(default_selected_type))
    ss.parameters['selected_type'] = selected_type

    
    # Validation rules
    if forecast_start_date <= baseline_start_date or forecast_end_date <= baseline_start_date:
        st.error("Invalid forecast dates. Forecast dates should be greater than baseline start date.")
    elif baseline_end_date <= baseline_start_date:
        st.error("Invalid baseline end date. Baseline end date should be greater than baseline start date.")

    st.write('---')
    st.write("## Model Selection")
    available_models = ['Linear', 'Exponential', 'Upper Limit', 'Lower Limit', 'Prophet', 'Sarima', 'Gaussian']
    selected_models = st.multiselect("Select Forecast Models", available_models, default=ss['selected_models'])
    
    
    st.write('---')
    st.write("### Model Parameters Selection")
    # if 'Linear' in selected_models:
    #     st.write("No additional parameters needed for Linear model.")
    # if 'Gaussian' in selected_models:
    #     st.write("No additional parameters needed for Gaussian model.") 
    # if 'Prophet' in selected_models:
    #     st.write("No additional parameters needed for Prophet model.") 

    if 'Exponential' in selected_models:
        st.write("### Exponential Parameters")
        with st.container(border=True):
            with st.expander("Click here to know info about the parameters"):
                st.write("**Trend Options:**")
                st.write("- 'add': Adds a linear trend component to the model.")
                st.write("- 'mul': Multiplies the data by an exponentially weighted trend.")
                st.write("- None: No trend component is included in the model.")
                st.write("**Seasonal Options:**")
                st.write("- 'add': Adds seasonal variations to the model in an additive manner.")
                st.write("- 'mul': Incorporates seasonal variations by multiplying the data by seasonal factors.")
                st.write("- None: No seasonal component is included in the model.")
            c1,c2,c3 = st.columns(3)
            exponential_options = ['add', 'mul', 'None']
            trend_option = c1.selectbox(
                "Trend", 
                options=exponential_options,
                index = exponential_options.index(ss['model_params']['Exponential'][0])
            )
            seasonality_option = c2.selectbox(
                "Seasonality", 
                options=exponential_options,
                index = exponential_options.index(ss['model_params']['Exponential'][1])
            )
            seasonal_period = c3.number_input(
                "Seasonal Period (weeks)", 
                min_value=2, max_value=105,
                value = ss['model_params']['Exponential'][2]
            )
            confidence_interval = st.number_input("Confidence Interval (%)", min_value=0, max_value=100,step=5, value=ss['model_params']['Exponential'][3])
            

    
    if 'Sarima' in selected_models:
        st.write("### SARIMA Parameters")
        with st.container(border=True):
            with st.expander("Click here to know info about the parameters"):
                st.write("**Order (Non-seasonal components):**")
                st.write("- (p, d, q): Represents the non-seasonal ARIMA order.")
                st.write("- p: Autoregressive (AR) order, which represents the number of lagged observations included in the model.Typically ranging from 0 to 10 or higher, depending on the complexity of the time series data.")
                st.write("- d: Degree of differencing, which represents the number of times the data needs to be differenced to make it stationary.Any non negative interger")
                st.write("- q: Moving Average (MA) order, which represents the number of lagged forecast errors included in the model.Typically ranging from 0 to 10 or higher, depending on the complexity of the time series data.")
                st.write("**Seasonal Order (Seasonal components):**")
                st.write("- (P, D, Q, s): Represents the seasonal ARIMA order.")
                st.write("- P: Seasonal autoregressive order.")
                st.write("- D: Seasonal differencing order.")
                st.write("- Q: Seasonal moving average order.")
                st.write("- s: Seasonal period, i.e., the number of time periods in a season.")

            c4,c5=st.columns(2)
            try:
                order_input = c4.text_input("Order eg.1,1,1", placeholder='1,1,1', value=ss['model_params']['Sarima'][0])
                seasonal_order_input = c5.text_input("Seasonal Order eg.1,1,1,13", placeholder='1,1,1,13', value=ss['model_params']['Sarima'][1])
                
                if order_input is not None and seasonal_order_input is not None:
                    # Convert input strings to tuples of integers
                    order = tuple(map(int, order_input.strip().split(',')))
                    seasonal_order = tuple(map(int, seasonal_order_input.strip().split(',')))
                else:
                    st.warning('Please enter the parameters')
            except ValueError:
                # Catch the ValueError and display a message asking the user to select a model first
                st.warning('Please remove the brackets before entering again', icon="🚨")


                
                                        
    # Display the selected parameters
    st.write("### Submit Parameters")
    
    if st.button("Submit"):
        ss.submit_clicked = True
        # Display the selected parameters
        ss['selected_models'] = selected_models
        if not selected_models:
            st.info("ℹ️ Please select a forecast model for parameter selection.")
        else:
            ss['selected_model_params'] = ss['model_params']
        
        if 'Exponential' in selected_models:
            ss['model_params']['Exponential'][0] = trend_option
            ss['model_params']['Exponential'][1] = seasonality_option
            ss['model_params']['Exponential'][2] = seasonal_period
            ss['model_params']['Exponential'][3] = confidence_interval
        if 'Sarima' in selected_models:
            ss['model_params']['Sarima'][0] = order
            ss['model_params']['Sarima'][1] = seasonal_order
        st.switch_page('pages/3_📊Results.py')
    else:
        st.info("ℹ️ Please select the Forecasting model and click 'Process' to view the parameters.")

else:
    st.warning("⚠️ Please upload a file first to perform data validation.")


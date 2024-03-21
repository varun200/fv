import streamlit as st


st.set_page_config(
    
    page_title="Forecast Validation",
    layout="wide",
    initial_sidebar_state="auto",
    page_icon='📈'
)

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
# Set page width to wide
# Custom CSS for styling
custom_styles = {
    "container": {"padding": "0", "background-color": "transparent", "border": "none"},
    "nav-link": {"background-color": "transparent"},
    "nav-link-selected": {"padding": "10px", "background-color": "red"}
}

         


    
# Custom CSS for styling
st.markdown(
    """
    <style>
    .title {
        font-size: 40px; /* Increased font size */
        color: #ffffff; /* Changed font color to white */
        text-align: center;
        padding-bottom: 20px;
    }
    .description {
        font-size: 20px; /* Increased font size */
        color: #ffffff; /* Changed font color to white */
        text-align: justify;
        padding-bottom: 20px;
    }
    .section {
        font-size: 24px; /* Increased font size */
        color: #ffffff; /* Changed font color to white */
        padding-bottom: 10px;
    }
    .emoji {
        font-size: 24px;
        padding-right: 5px;
    }
    .line {
        border-top: 2px solid #ffffff; /* Changed border color to white */
        margin: 20px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Page title and description
st.markdown("<h1 class='title'>Instructions</h1>", unsafe_allow_html=True)
st.markdown(
    "\n\n"
    "---"  # Horizontal line
    "\n\n"
    "**Sections:**"
    "\n\n"
    "1. Upload: Upload your data files containing historical data to begin the forecasting process."
    "\n"
    "2. Model Parameters: Set the parameters for your forecasting model, including time periods, variables, and any additional specifications."
    "\n"
    "3. Results: View the results of your forecasting analysis, including predicted trends, confidence intervals, and actionable insights."
    "\n\n"
    "Start forecasting with ease and confidence. Let's predict the future together! 🚀"
)

#For the GUI part
import streamlit as st
import pickle
import numpy as np
from scipy.special import inv_boxcox



import warnings
warnings.filterwarnings('ignore')

#this value will be used to derive the value of the predicted Selling Price
boxcox_lambda = 0.20033253352979563

# Load the models from the .pkl file
#regression model
with open('random_forest_model_SP.pkl', 'rb') as file1:
    sp_model = pickle.load(file1)
#classification model
with open('xgb_classification_model_status.pkl', 'rb') as file2:
    status_model = pickle.load(file2)

#fixed values of Application, country, status, item_type and product_ref
application_list = [ '2.0','3.0','4.0','5.0','10.0','15.0','19.0', '20.0','22.0','25.0', '26.0', '27.0','28.0', '29.0','38.0','39.0','40.0','41.0', '42.0', '56.0', '58.0','59.0', '65.0','66.0',    '67.0', '68.0', '69.0','70.0', '79.0', '99.0'    
                   ]
country_list = [ '25.0','26.0', '27.0','28.0',   '30.0',  '32.0',  '38.0','39.0',  '40.0', '77.0', '78.0','79.0','80.0','84.0','89.0','107.0','113.0'            
                ]

status_list = [ 'Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM',
       'Wonderful', 'Revised', 'Offered', 'Offerable']

item_type_list = [ 'W', 'WI', 'S', 'PL', 'IPL', 'SLAWR', 'Others']

product_ref_list = ['611728','611733','611993','628112','628117','628377', '640400', '640405','640665',
                    '164141591','164336407','164337175',    '929423819','1282007633','1332077137', '1665572032','1665572374','1665584320','1665584642','1665584662','1668701376','1668701698','1668701718','1668701725','1670798778',      
          '1671863738', '1671876026', '1690738206', '1690738219',   
       '1693867550', '1693867563','1721130331','1722207579'         
                    ]

###### function definitions ######################
#To predict Selling Price once all values have been entered
def click_btn_SP():
    status_array = create_status_values(status) # create an array for the one hot encoded status columns based on user input
    item_array = create_item_type_values(item_type) # create an array for the one hot encoded item_type columns based on user input 
    country_array = create_country_values(country) # create an array for the one hot encoded country columns based on user input
    new_data = np.array([item_date, customer_id, application_id, product_ref,
       delivery_date,width,thickness,quantity]) # array of user inputs

    #add other data arrays to above array
    final_data = np.concatenate((new_data,status_array))
    final_data = np.concatenate((final_data,item_array))
    final_data = np.concatenate((final_data,country_array
                                ))
    final_data = np.array(final_data).reshape(1, -1) # array reshape being used as input is only one sample

    # st.write(final_data)
    # Make predictions using the loaded model
    predicted_price = sp_model.predict(final_data)
    pred_sp = inv_boxcox(predicted_price[0], boxcox_lambda)

    st.write("Predicted Selling Price:", pred_sp)
    

#To predict Status once all values have been entered
def click_btn_Status():
    item_array = create_item_type_values(item_type) # create an array for the one hot encoded item_type columns based on user input 
    country_array = create_country_values(country) # create an array for the one hot encoded country columns based on user input
    new_data = np.array([item_date, customer_id, application_id, product_ref,
       delivery_date,width,thickness,quantity,selling_price]) # array of user inputs

    #add other data arrays to above array
    final_data = np.concatenate((new_data,item_array))
    final_data = np.concatenate((final_data,country_array
                                ))
    final_data = np.array(final_data,dtype = object).reshape(1, -1) # array reshape being used as input is only one sample

    # Make predictions using the loaded model
    predicted_status = status_model.predict(final_data)
    
    if (predicted_status == 1):
        st.write("Predicted Status: Won")
    else:
        st.write("Predicted Status: Lost")




def create_item_type_values(item_type_input):
    # Define the possible categories 
    item_types = ['Others', 'PL', 'S', 'SLAWR', 'W', 'WI']

    
    # Initialize the one-hot encoded values to False (or 0)
    one_hot_encoded_values = np.array([0.0,0.0,0.0,0.0,0.0,0.0])

    # Set the column corresponding to the user input to True (or 1)
    for i in range(6):
        if item_type_input == item_types[i]:
            one_hot_encoded_values[i] = 1.0
    
    return one_hot_encoded_values

def create_status_values(status_input):
    # Define the possible categories 
    status_types = ['Lost', 'Not lost for AM',
       'Offerable', 'Offered', 'Revised',
       'To be approved', 'Won', 'Wonderful']    

    # Initialize the one-hot encoded values to False (or 0)
    one_hot_encoded_values = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])

    # Set the column corresponding to the user input to True (or 1)
    for i in range(8):
        if status_input == status_types[i]:
            one_hot_encoded_values[i] = True
    
    return one_hot_encoded_values
    
def create_country_values(country_input):
    # Define the possible categories 
    country_types = [25.0, 26.0, 27.0,
       28.0, 30.0, 32.0, 38.0, 39.0,
       40.0, 77.0, 78.0, 79.0, 80.0,
       84.0, 89.0, 107.0, 113.0]    

    # Initialize the one-hot encoded values to False (or 0)
    one_hot_encoded_values = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])

    # Set the column corresponding to the user input to True (or 1)
    for i in range(17):
        if country_input == country_types[i]:
            one_hot_encoded_values[i] = 1.0
    
    return one_hot_encoded_values
##############################################################################


#Setting the page layout
st.set_page_config(
    page_title="Industrial Copper Modeling",
    page_icon=":factory:",
    layout="wide",
    initial_sidebar_state="expanded")


#Create the first screen of the streamlit application, to display few radio buttons on the sidebar and for user inputs on the right.
header = st.container()

sidebar1 = st.sidebar

with header:
    st.subheader("Industrial Copper Modeling - Predicting Selling Price and Status")

with sidebar1:
    selection = sidebar1.radio("What's your choice of task?",[":house: Home",":1234:(Predict Selling Price)",":yin_yang:(Predict Status)"])

#Plain text information about the project and what it  intends to do
if selection == ":house: Home": 
    st.markdown("The copper industry deals with less complex data related to sales and pricing.")
    st.markdown("However, this data may suffer from issues such as skewness and noisy data, which can affect the accuracy of manual predictions.")
    st.markdown("Dealing with these challenges manually can be time-consuming and may not result in optimal pricing decisions.")
    st.markdown("A machine learning regression model can address these issues by utilizing advanced techniques such as data normalization, feature scaling, and outlier detection, and leveraging algorithms that are robust to skewed and noisy data.")
    st.markdown("Another area where the copper industry faces challenges is in capturing the leads.")
    st.markdown("A lead classification model is a system for evaluating and classifying leads based on how likely they are to become a customer.")
    st.markdown("This project hence aims to create a regression model to predict Selling Price given all other relevant parameters and also use a classification model to predict Status of the lead whether Won or Lost.")

# User input fields to predict Selling Price
if selection == ":1234:(Predict Selling Price)":
    col1,col2 = st.columns([3,3])
    with col1:
        item_date = st.number_input("Item Date: YYYYMMDD : min = 19950101.0, max = 20210401.0")
        customer_id = st.text_input("Customer ID: (Min: 0 & Max: 2147484000)")
        application_id = st.selectbox("Application ID:",application_list)
        product_ref = st.selectbox("Product Reference:",product_ref_list)
        delivery_date = st.number_input("Delivery Date: YYYYMMDD : Greater than Item Date")
        
    with col2:
        width = st.number_input("Width: min = 300, max = 1000")
        thickness = st.number_input ("Thickness: min = -2 , max = 4")
        quantity = st.number_input("Quantity: min = -1, max = 8 ")
        status = st.selectbox("Status:",status_list)
        item_type = st.selectbox("Item Type: ",item_type_list)
        country = st.selectbox("Country:",country_list)
    SP_btn = st.toggle("Predict Selling Price")

    if SP_btn:
        click_btn_SP()

# User input fields to predict Status
if selection == ":yin_yang:(Predict Status)":
    col3,col4 = st.columns([3,3])
    with col3:
        item_date = st.number_input("Item Date: YYYYMMDD : min = 19950101, max = 20210401")
        customer_id = st.text_input("Customer ID: (Min: 0 & Max: 2147484000)")
        application_id = st.selectbox("Application ID:",application_list)
        product_ref = st.selectbox("Product Reference:",product_ref_list)
        delivery_date = st.number_input("Delivery Date: YYYYMMDD : Greater than Item Date")

        

    with col4:
        width = st.number_input("Width: min = 300, max = 1000")
        thickness = st.number_input ("Thickness: min = -2 , max = 4")
        quantity = st.number_input("Quantity: min = -1, max = 8 ")
        selling_price = st.number_input("Selling Price: min = 9, max = 19 ")
        item_type = st.selectbox("Item Type: ",item_type_list)
        country = st.selectbox("Country:",country_list)

    Status_btn = st.toggle("Predict Status")

    if Status_btn:
        click_btn_Status()




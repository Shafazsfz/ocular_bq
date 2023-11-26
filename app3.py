import os
import streamlit as st
from streamlit_chat import message
import pandas as pd
from langchain import OpenAI
from langchain.agents import create_sql_agent, AgentType
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.utilities import SQLDatabase
from langchain.chat_models import ChatOpenAI
from datetime import datetime
from google.cloud import bigquery
from sqlalchemy import *
from sqlalchemy.engine import create_engine
from sqlalchemy.schema import *
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.agents import AgentExecutor
import tempfile

api_key = st.sidebar.text_input("Enter your API key:", type="password")

service_account_file = st.sidebar.file_uploader("Upload BigQuery Service Account file", type=['json'])

project = st.sidebar.text_input("Enter Project ID:","linked-368910")

dataset = st.sidebar.text_input("Enter Dataset Name:","thelook_ecommerce")

#table = "churn_table"

if service_account_file is not None:
    # Save the file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp_file:
        tmp_file.write(service_account_file.getvalue())
        tmp_file_path = tmp_file.name
    # Use the temporary file path in the SQLAlchemy URL
    sqlalchemy_url = f'bigquery://{project}/{dataset}?credentials_path={tmp_file_path}'


prefix_text = st.sidebar.text_area(
    "Enter prefix to pass:",
    """Given an input question, first create a syntactically correct sqlite query to run, then look at the results of the query and return the answer.
    note: Use `ILIKE %keyword%` in your SQL query to perform fuzzy text matching within text  or string datatype columns.
    
    Use the following for output format:
    Question: "Question here"
    SQLQuery: "SQL Query to run"
    SQLResult: "Result of the SQLQuery"
    Answer: "Final answer here"
    You should return the SQL query along with the result or response.

    Your output for user input should be like this format-

    ```     Question: "How many products are there?"
            SQLQuery: "Select count (distinct products) from orders"
            SQLResult: "1112"
            Answer: "There are 1112 products."
    ```""",height=600
    )
st.sidebar.write(f'You wrote {len(prefix_text)} characters.')


custom_table_info = {

   "fact_order_item": f""" CREATE TABLE `fact_order_item` (
                    order_id STRING,
                    order_item_id STRING,
                    marketplace STRING,
                    order_date TIMESTAMP,
                    quantity NUMERIC,
                    price NUMERIC	,
                    discounts NUMERIC,
                    shipping_pincode STRING,
                    state STRING,
                    city STRING,
                    sku_name STRING,
                    asin STRING,
                    amazon_parent_sku STRING,
                    fabric STRING,
                    product_name STRING,
                    collection STRING,
                    master_collection STRING,
                    product_type STRING,
                    weigh_slab INTEGER,
                    awb STRING,
                    shipping_status STRING,
                    simplified_status STRING,
                    fulfillment_channel STRING,
                    inner_consumption FLOAT	,
                    outer_consumption FLOAT	,
                    consumption_cost FLOAT	,
                    cost NUMERIC
                    )

    /*
    Note for the Table :-
    1.order_id : A unique identifier for each order.
    2.order_item_id : A unique identifier for each item within an order.
    3.marketplace: The platform or marketplace where the order was placed. Example :- Amazon, Shopify, Myntra, Flipkart
    4.order_date: The date and time when the order was placed in UTC format.
    5.quantity : The number of units of the item ordered.
    6.price : The price of a single unit of the item. The price in INR currency
    7.discounts : Any discounts applied to the order at an item level
    8.shipping_pincode : The postal code to which the order is shipped.
    9.state : The state where the order is shipped.
    10.city : The city where the order is shipped.
    11.sku_name : The stock keeping unit name, a unique identifier for each product.
    12.asin : Amazon Standard Identification Number, unique for products on Amazon.
    13.amazon_parent_sku : The parent SKU for products that have variations on Amazon.
    14.fabric : The type of fabric (if applicable) of the product.
    15.product_name : The name of the product. If a user asks for product use the sku_name column.
    16.collection : The collection to which the product belongs.
    17.master_collection : A broader collection category that the product belongs to.
    18.product_type : The type or category of the product.
    19.weigh_slab : A range of weight for shipping purposes.
    20.awb : Air Waybill number, used for tracking shipments.
    21.shipping_status : The current status of the shipment.
    22.simplified_status: A simplified or general status of the order. RTO( Return to Origin ), Returned, Return to Origin orders should not be considered in revenue calculations. For details or customer comments on the status look at fact_shipping table.
    23.fulfillment_channel : The channel through which an Amazon order was fullfilled.
    24.inner_consumption : Internal consumption of fabric .
    25.outer_consumption : External consumption of fabric .
    26.consumption_cost : Cost associated with consumption including inner or outer).
    27.cost: The total cost of procuring the product.
    28.Total revenue for a product is price into quantity minus the discount.
    29.Fact Order item has all details about orders.
    */""",


    "fact_shipping": f""" CREATE TABLE `fact_shipping` (
                  order_id STRING,
                  marketplace STRING,
                  placed_date DATETIME,
                  shipping_status STRING,
                  simplified_status STRING,
                  aging INTEGER,
                  awb STRING,
                  payment_method STRING,
                  pin_code STRING,
                  courier_partner STRING,
                  state STRING,
                  city STRING,
                  item_sku_code STRING,
                  amazon_parent_sku STRING,
                  items_quantity NUMERIC,
                  weight NUMERIC,
                  delivery_date TIMESTAMP,
                  return_reference_number STRING,
                  return_status STRING,
                  return_request_date DATE,
                  refund_amount NUMERIC,
                  return_quantity NUMERIC,
                  return_reason STRING,
                  simplified_return_reason STRING,
                  return_delivery_date DATE,
                  collection STRING,
                  master_collection STRING,
                  weigh_slab INTEGER
                 )
    /*
    Note for the Table :-
    1.order_id: A unique identifier for each order.
    2.marketplace : The name of the marketplace where the order was placed.  Example :- Amazon, Shopify, Myntra, Flipkart
    3.placed_date : The date and time when the order was placed.
    4.shipping_status : The current status of the shipping process (e.g., pending, shipped, delivered).
    5.simplified_status : A simplified or general status of the order. RTO( Return to Origin ) orders should not be considered in revenue calculations
    6.aging : The number of days since the order was placed or a specific event in the shipping process occurred.
    7.awb: Air Waybill number, which is used to track and identify shipments.
    8.payment_method : The method used for payment (e.g., credit card, PayPal, cash on delivery).
    9.pin_code : The postal code for the delivery address.
    10.courier_partner : The name of the courier or shipping partner handling the delivery.
    11.state: The state of the delivery address.
    12.city : The city of the delivery address.
    13.item_sku_code : The stock keeping unit code for the item, unique to each product.
    14.amazon_parent_sku : The parent SKU for products that have variations, specifically on Amazon.
    15.items_quantity : The quantity of items in the order.
    16.weight : The total weight of the order or shipment.
    17.delivery_date : The date and time when the order was delivered.
    18.return_reference_number : A unique identifier for a return, if applicable.
    19.return_status : The status of the return process (e.g., requested, processing, completed).
    20.return_request_date : The date when a return was requested.
    21.refund_amount : The amount to be refunded for the return.
    22.return_quantity : The quantity of items being returned.
    23.return_reason : The reason provided for the return or RTO ( Return to Origin ).
    24.simplified_return_reason : A simplified or general reason for the return.
    25.return_delivery_date : The date when the returned item was delivered back.
    26.collection : The collection or line to which the product belongs.
    27.master_collection : A broader collection category encompassing various individual collections.
    28.weigh_slab : A category or range of weight for shipping or return purposes.
    29. This table contains all information about shipping details , return date and return reasons.
    */""",
    }

    
# Initialize the LLMPredictor and other necessary components
def initialize_llm_predictor():

    db = SQLDatabase.from_uri(sqlalchemy_url, custom_table_info=custom_table_info)
    llm = ChatOpenAI(model_name='gpt-3.5-turbo',temperature=0)
    #llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
    #llm= OpenAI(model_name='gpt-3.5-turbo',temperature=0)

    agent = create_sql_agent(
        llm= llm,
        toolkit=SQLDatabaseToolkit(db=db, llm=llm),
        verbose=True,
        #agent_type=AgentType.OPENAI_FUNCTIONS,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        prefix=prefix_text,
        top_k=1000
    )
    return agent


if api_key and project and dataset and service_account_file:

    os.environ["OPENAI_API_KEY"] = api_key
    st.title("Ocular - Langchain - Text2Sql")

    # Initialize LLMPredictor
    query_engine = initialize_llm_predictor()

    def conversational_chat(query):
        result = query_engine.run(query)
        st.session_state['history'].append((query, result))
        return result

    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything."]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]
            
    #container for the chat history
    response_container = st.container()
        
    #container for the user's text input
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            
            user_input = st.text_input("Query:", placeholder="Ask question here", key='input')
            submit_button = st.form_submit_button(label='Send')
                
        if submit_button and user_input:
            output = conversational_chat(user_input)
                
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")

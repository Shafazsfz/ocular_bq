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


# custom_table_info = st.sidebar.text_area(
#     "Enter table info:",{
#     "orders":
#                     f"""CREATE TABLE "orders" (
#                         order_id INTEGER PRIMARY KEY, 
#                         order_date DATE, 
#                         shipping_date DATE, 
#                         delivery_date DATE, 
#                         return_date DATE, 
#                         order_status TEXT, 
#                         order_qty INTEGER, 
#                         customer_id INTEGER, 
#                         product TEXT, 
#                         price REAL, 
#                         warehouse_loc TEXT, 
#                         category TEXT, 
#                         prod_cost REAL, 
#                         brand TEXT, 
#                         customer_city TEXT, 
#                         customer_country TEXT, 
#                         traffic_source TEXT
#                         )

#     /*
#     Note: 
#     1. Revenue is calculated by multiplying 'price' and 'qty' columns.
#     2. A user or customer is new if 'customer_id' didn't exisit before the given 'order_date'.
#     3. Date format for 'order_date' , 'shipping_date', 'delivery_date', 'return_date' columns are dd-mm-yy .
#     4. The orders table is an orders table, representing all the orders placed, along with product information, customer information (customer ID) and the associated revenue with the order (qty of products and price).
#     5. Order_id column is the unique identifier of every order placed in the orders store.
#     6. Order_date is the date the order is placed in DD/MM/YY format.
#     7. Shipping_date is the date an order is shipped from orders facility to the customer in DD/MM/YY format.
#     8. Delivery_date is the date an order reaches the customer in DD/MM/YY format.
#     9. Return_date is non-null only when an order has been returned, do note, all returns should not be considered revenue earned. The return_date is the date customer has returned their product in DD/MM/YY format.
#     10. Order_status is a string field that can only have the values: Complete (Revenue to be considered), Shipped (Revenue to be considered), Processing (Revenue not to be considered), Cancelled (Revenue not to be considered), Returned (Revenue not to be considered).
#     11. Order_qty is the quantity of the product ordered in that order in integer format.
#     12. Customer_id is the unique identifier of a customer of orders in numeric format.
#     13. Product is string format of the exact product being ordered.
#     14. Price is numeric format of the USD price for one unit of the product.
#     15. Warehouse_loc is String format of the warehouse location from which the product is being shipped.
#     16. Category is string format of the category within which the product in the specified row falls.
#     17. Prod_cost is the cost of one unit of the product in USD, useful to calculate gross profit from an order.
#     18. Brand is the string format brand from which that product belongs to, eg. Coke is the brand of cold drinks that a large number of people drink.
#     19. Customer_city is the string format city from which the customer belongs.
#     20. Customer_country is the string format country from which the customer belongs.
#     21. Traffic_source is the string format of marketing channel from which the order came in.
#     22. Use `ILIKE %keyword% in your SQL query to perform fuzzy text matching within text  or string datatype columns.
#     """},height=600)

    
# Initialize the LLMPredictor and other necessary components
def initialize_llm_predictor():

    db = SQLDatabase.from_uri(sqlalchemy_url)
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
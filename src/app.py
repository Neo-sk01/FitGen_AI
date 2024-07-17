import os
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
import streamlit as st
import spacy
from transformers import pipeline

def init_database(user: str, host: str, port: str, database: str) -> SQLDatabase:
    db_uri = f"mysql+mysqlconnector://{user}:@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load transformers model for text generation
text_generator = pipeline("text2text-generation", model="google/t5-small-ssm-nq")

# Function to correct typos and understand user queries
def process_query(query):
    # Use spaCy for typo correction and text preprocessing
    doc = nlp(query)
    corrected_query = " ".join([token.text for token in doc])
    
    # Use transformers for text generation and understanding
    response = text_generator(f"Correct and clarify the following question: {corrected_query}")

    return response[0]['generated_text']

def get_sql_chain(db):
    template = """
    You are a personal trainer analyzing user data to provide tailored health, body workouts and fitness advice.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.

    Do not show the query to the user how you got the answer you will be providing. Write only simple english/natural language.
    
    If there is a typo in the user's question or you cannot understand the question, ask for clarification.
    If the question is outside the scope of the database schema, respond with "Sorry, I cannot answer that question."

    For example:
    User: What is my current weight?
    SQL Query: SELECT Weight FROM Users WHERE UserID;
    User: Show my recent workout history.
    SQL Query: SELECT * FROM UserProgress WHERE UserID ORDER BY Date DESC LIMIT 5;
    
    Your turn:
    
    User: {question}
    SQL Query:
    """

    # Process the user's question using NLP techniques
    def process_user_query(question, schema, chat_history):
        corrected_query = process_query(question)
        if "Sorry, I cannot answer that question" in corrected_query:
            return "Sorry, I cannot answer that question."
        return corrected_query

    prompt = ChatPromptTemplate.from_template(template)
    
    # llm = ChatOpenAI(model="gpt-4-0125-preview")
    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
    
    def get_schema(_):
        return db.get_table_info()
    
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )
    
def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    sql_chain = get_sql_chain(db)
    
    template = """
    You are a personal trainer analyzing user data to provide tailored health, body workouts and fitness advice.
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # llm = ChatOpenAI(model="gpt-4-0125-preview")
    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
    
    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: db.get_table_info(),
            response=lambda vars: db.run(vars["query"]),
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    })
    
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm your personal trainer. Ask me anything about your health goals."),
    ]

load_dotenv()

st.set_page_config(page_title="Chat with your Trainer", page_icon=":speech_balloon:")

st.title("Chat with your Personal Trainer")

with st.sidebar:
    st.subheader("Settings")
    st.write("This is a simple chat application using MySQL. Connect to the database and start chatting.")
    
    st.text_input("Host", value="localhost", key="Host")
    st.text_input("Port", value="3306", key="Port")
    st.text_input("User", value="root", key="User")
    
    st.text_input("Database", value="fitgen_ai", key="Database")
    
    if st.button("Connect"):
        with st.spinner("Connecting to database..."):
            db = init_database(
                st.session_state["User"],
                st.session_state["Host"],
                st.session_state["Port"],
                st.session_state["Database"]
            )
            st.session_state.db = db
            st.success("Connected to database!")
    
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

user_query = st.chat_input("Type a message...")
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)
        
    with st.chat_message("AI"):
        response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
        st.markdown(response)
        
    st.session_state.chat_history.append(AIMessage(content=response))
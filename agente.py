import os
import lancedb
import logging
import pandas as pd
import streamlit as st

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import LanceDB
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

embedding_model = OllamaEmbeddings(model="nomic-embed-text:latest")

# Función para obtener el embedding de un texto, la cual tambien reemplaza los saltos de linea del textp por espacios
def get_embedding(text):
  text = text.replace("\n", " ")
  embedding = embedding_model.embed_query(text)
  return embedding

def create_database():
  logger.info("Se crea u obtitne la base de datos hotel_reviews_db y la tabla hotel_reviews")
  # Conexión a la base de datos vectorial (LanceDB)
  vector_store = LanceDB(
    uri=os.path.join("data", "hotel_reviews_db"),
    embedding=embedding_model,
    table_name="hotel_reviews",
    mode="overwrite"
  )

  # Si la cantidad de registros en la tabla es diferente a la cantidad de filas en el dataframe, se procede a cargar los datos
  if(not vector_store.get_table()):
    # Dataset de los reviews de hoteles
    logger.info("Carga del dataset data/hotel_reviews.csv")
    data = pd.read_csv(os.path.join("data", "hotel_reviews.csv"))

    logger.info("Se procede a cargar los datos en la base de datos")

    # Se crea una columan en el dataset con la información combinada del dataset
    data["text"] = data.apply(lambda row: f"Name: {row['name']}. Review: {row['description']} Rating: {row['rating']}", axis=1)
    data["vector"] = data.description.apply(lambda x: get_embedding(x))

    vector_store.add_texts(
      texts=data.text,
      ids=data.id.values.tolist(),
      embeddings=data.vector.values.tolist()
    )

  return vector_store

prompt = ChatPromptTemplate.from_messages(
  [
    (
      "system",
      """
      Eres un agente turistico con muchos años de experiencia que impulsa el turismo en Colombia,
      trabajas para una agencia de turismo muy importante y reconocidad por su calidad en el servicio.
      Tu objetivo es recomendar hoteles en Colombia a los turistas que desean visitar el país,
      dando respuestas precisas y con los aspectos mas relevantes sobre los hoteles que recomiendes.
      """,
    ),
    MessagesPlaceholder("chat_history", optional=True),
    (
      "human",
      "{input}"
    ),
    MessagesPlaceholder("agent_scratchpad")
  ]
)

def get_recomendation(user_message: str):
  vector_store = create_database()
  retriever = vector_store.as_retriever()

  # Se crea el agente con el tool de la base de datos
  tool = create_retriever_tool(
    retriever=retriever,
    name="recomendar_hoteles",
    description="Buscar descripcion sobre hoteles de Colombia y recomendarlos a los turistas.",
  )

  tools = [tool]

  llm = ChatOpenAI(temperature=0)
  agent = create_openai_tools_agent(llm, tools, prompt)
  agent_executor = AgentExecutor(agent=agent, tools=tools)

  result = agent_executor.invoke(
    {
      "input": f"{user_message}"
    }
  )

  return result["output"]


# GUI
if "chat_messages" not in st.session_state:
  st.session_state.chat_messages = []

for chat_message in st.session_state.chat_messages:
  with st.chat_message(chat_message["role"]):
    st.markdown(chat_message["content"])

user_message = st.chat_input("Escribe algo...")

if user_message:
  with st.chat_message("user"):
    st.markdown(user_message)

  st.session_state.chat_messages.append({"role": "user", "content": user_message})

  response = get_recomendation(user_message)

  with st.chat_message("assistant"):
    st.markdown(response)

  st.session_state.chat_messages.append({"role": "assistant", "content": response})

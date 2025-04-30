import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_community import vectorstores
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import pgvector
from sqlalchemy import create_engine, text as sql_text
from dotenv import load_dotenv, dotenv_values
import os
from langchain.chat_models import init_chat_model

load_dotenv()

openai_token = os.getenv("OpenAI_API_Key")
if not os.environ.get("OPENAI_API_KEY"):  # If API token is not set in environment
    os.environ["OPENAI_API_KEY"] = openai_token
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
llm = init_chat_model("gpt-4o-mini", model_provider="openai")
query = "How many core credits we have to take?"

def retreieve_relavant_docs(query_embeddings, top_k = 5):

    connection_string = 'postgresql+psycopg://postgres:Aishu%40123@localhost:5432/postgres'
    collection_name = 'documents'
    embedding_string = f"[{', '.join(map(str, query_embeddings))}]"
    engine = create_engine(connection_string)
    sql = sql_text("""
        SELECT text
        from documents
        order by embedding <#> :query_embedding
        limit :top_k;
    """)
    with engine.connect() as conn:
        results = conn.execute(sql,{
            "query_embedding": embedding_string,
            "top_k": top_k
        }).fetchall()
    return results

def generate_response(context, question):
    prompt = """
        You are CourseBot, an intelligent assistant designed to help students understand course materials. You are given a context extracted from a course document and a student's question.
        Use only the context provided to answer the question.
        If the answer is not present in the context, respond with: "The answer to your question is not available in the provided course materials."
        Keep your answers clear, concise, and accurate.
        If the question is ambiguous or unclear, ask a clarifying question.

        Context:
        {context}

        Question:
        {question}
    """

    final_template = ChatPromptTemplate.from_template(prompt)
    final_prompt = final_template.invoke({"context":context, 'question':question})

    response = llm.invoke(final_prompt)
    return response.content

embedded_query = embeddings_model.embed_query(query)

context = "\n\n".join([row[0] for row in retreieve_relavant_docs(embedded_query)])

print(generate_response(context, query))


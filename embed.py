import pandas as pd
import numpy as np

#from sentence_transformers import SentenceTransformer
import fitz
from PIL import Image
import os
import io
import glob
from transformers import CLIPProcessor, CLIPModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv, dotenv_values  # Load environment variables from .env file
from datetime import datetime
from sqlalchemy import create_engine, text as sql_text
import psycopg2

load_dotenv()  # Load environment variables from a .env file

openai_token = os.getenv("OpenAI_API_Key")
if not os.environ.get("OPENAI_API_KEY"):  # If API token is not set in environment
    os.environ["OPENAI_API_KEY"] = openai_token
connection = psycopg2.connect(database = 'postgres', user = 'postgres',password = 'Aishu@123',host = 'localhost', port = '5432')
cursor = connection.cursor()

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
def extract_text_and_images(pdf_path, image_output_dir='pdf_images'):
    doc = fitz.open(pdf_path)
    texts = []
    images = []

    os.makedirs(image_output_dir, exist_ok=True)

    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        
        # Extract text
        texts.append(page.get_text())

        # Extract images
        images_info = page.get_images(full=True)
        for img_index, img in enumerate(images_info):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image = Image.open(io.BytesIO(image_bytes))

            image_path = os.path.join(image_output_dir, f"page{page_number+1}_img{img_index+1}.{image_ext}")
            image.save(image_path)
            images.append(image_path)
    
    return texts, images

def embed_pdf_texts(file):
    print("File exists:", os.path.exists(file))
    loader = PyPDFLoader(file)
    pages = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size  =1000, chunk_overlap = 10)
    chunked_documents = text_splitter.split_documents(pages)
    connection_string = 'postgresql+psycopg://postgres:Aishu%40123@localhost:5432/postgres'
    collection_name = 'documents'

    engine = create_engine(connection_string)

    file_name = file.split('\\')[-1]
    sql = f"delete from documents where source = '{file_name}'"
    cursor.execute(sql) 
    connection.commit()

    # Insert each chunk manually
    with engine.connect() as conn:
        for doc in chunked_documents:
            text = doc.page_content
            if isinstance(text, bytes):
                text = text.decode("utf-8", errors="replace").replace("\x00", "\ufffd")
            else:
                text = text.replace("\x00", "\ufffd")
            embedding = embeddings_model.embed_query(text)
            source = file_name
            added_at = datetime.utcnow()
            course_id = file_name.split('.')[0]


            insert_query = sql_text("""
                INSERT INTO documents (course_id, text, source, _added_at, embedding)
                VALUES (:course_id, :text, :source, :_added_at, :embedding)
            """)

            conn.execute(insert_query, {
                "course_id": course_id,
                "text": text,
                "embedding": list(embedding),  # Ensure it's a vector type
                "source": source,
                "_added_at": added_at
            })

        conn.commit()

def test_fitz_read(file):
    doc = fitz.open(file)
    for i, page in enumerate(doc):
        text = page.get_text()
        print(f"\n--- Page {i+1} ---\n{text[:500]}")  # print first 500 chars per page


path = 'Data/*.pdf'

for fname in glob.glob(path):
    print(fname)
    embed_pdf_texts(fname)
 
# import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings,OpenAIEmbeddings
from langchain.vectorstores import FAISS,qdrant, pinecone
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceHub
import qdrant_client
import requests
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModel, TextStreamer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.llms import HuggingFacePipeline
# from get_edge_cases import get_edge_cases
from langchain_core.messages import SystemMessage
import langchain.schema
import torch
import pandas as pd
# from extract_text_and_table import parse_pdf
import time

# Backend code

load_dotenv()

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
        
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        # separator=' ',
        chunk_size=1000,
        chunk_overlap=200,
        # length_function=len
    )

    chunks = text_splitter.split_text(text)

    with open("chunks.txt","w") as file:
        file.write(str(chunks[0]))

    return chunks

def get_vector_store(chunks,user_id):
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")    
    embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")

    # for openai embeddings
    # embeddings = OpenAIEmbeddings()

    # vectorstore = FAISS.from_texts(texts=chunks,embedding=embeddings)

    vectorstore = qdrant.Qdrant.from_texts(texts=chunks,embedding=embeddings,url=QDRANT_HOST,api_key=QDRANT_API_KEY,collection_name=user_id)

    # vectorstore = pinecone.from_texts(texts=chunks,embedding=embeddings,url = os.getenv("PINECONE_HOST"), api_key=os.getenv("PINECONE_API_KEY") ,collection_name=user_id)

    return vectorstore

def get_conversation_chain(vectorstore):
    

    # model_path = "casperhansen/mixtral-instruct-awq"

    model_path = "TheBloke/Mistral-7B-Instruct-v0.1-AWQ"

    # model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto").cuda()

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_params = {
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "max_new_tokens": 1000,
        "repetition_penalty": 1.1
    }

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        streamer=streamer,
        **generation_params
    )

    
    llm = HuggingFacePipeline(pipeline=pipe)
    # llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,        
            retriever=vectorstore.as_retriever(),
            memory=memory,      
    )


    return conversation_chain


# Storing the data in a vector database (Qdrant)    

def fetch_vector_store(user_id):
    client = qdrant_client.QdrantClient(
        os.getenv(QDRANT_HOST),
        api_key=os.getenv(QDRANT_API_KEY)
    )

    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")    

    vectorstore = qdrant.Qdrant(
        client=client,
        collection_name=user_id,
        embeddings=embeddings
    )

    try:
        collection_info = client.get_collection(collection_name=user_id)
        vectors_count = collection_info.vectors_count if collection_info else 0
        # print(vectors_count)

    except:
        vectors_count=0

    print(f"Vectors count: {vectors_count}")
    
    return vectorstore if vectors_count>0 else None

if __name__ == "__main__":
    
    HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_HOST = os.getenv("QDRANT_HOST")
    
    user_id = "new_id"

    # os.environ["QDRANT_COLLECTION_NAME"] = user_id

    
    # user_id = "USA_IBC_IPC"

    vectorstore = fetch_vector_store(user_id=user_id)

    if not vectorstore:

        print("User's database empty, adding new vectors")

        # get pdf text

        pdf_docs = ["sample.pdf"]

        raw_text = get_pdf_text(pdf_docs)

        # get text chunks   

        text_chunks = get_text_chunks(raw_text)

        # create vector store
        vectorstore = get_vector_store(text_chunks,user_id)

        print("vectorstore created")

        # save the vector store into user's vector database

        print(f"Saved vector into database with cluster name: {user_id}")

    # create conversation chain

    conversation = get_conversation_chain(vectorstore)

    # print(conversation)
    print("conversation initialized")
    print("Type 'exit()' to stop AI responses.")

    system_prompt = """"""    
    
    model_path = "TheBloke/Mistral-7B-Instruct-v0.1-AWQ"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    while True:

        user_question = input("User: ")
        if user_question=='exit()':
            break
        else:
            prompt_template=f'''<s>[INST] System_prompt: {system_prompt} {user_question} [/INST]'''
            start_time = time.time()
            response = conversation({'question':prompt_template})
            end_time = time.time()
            total_output_tokens = len(tokenizer.tokenize(response["answer"]))
            time_taken = end_time - start_time
            output_tokens_per_second = total_output_tokens / time_taken

            print(f"AI response: {response['answer']}\n\n{'*'*100}\n")

            print("Total time taken: ",time_taken)
            print("Tokens per second: ",output_tokens_per_second)
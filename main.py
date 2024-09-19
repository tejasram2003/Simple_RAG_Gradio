import gradio as gr
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import qdrant
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
import os
import torch
import time
from langchain.embeddings import HuggingFaceInstructEmbeddings
import qdrant_client


load_dotenv()

def get_pdf_text(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(chunks, user_id, QDRANT_HOST, QDRANT_API_KEY):
    embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
    vectorstore = qdrant.Qdrant.from_texts(texts=chunks, embedding=embeddings, url=QDRANT_HOST, api_key=QDRANT_API_KEY, collection_name=user_id)
    return vectorstore

def get_conversation_chain(vectorstore):
    model_path = "TheBloke/Mistral-7B-Instruct-v0.1-AWQ"
    model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, device_map="auto")
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
    
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, streamer=streamer, **generation_params)
    llm = HuggingFacePipeline(pipeline=pipe)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,        
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    return conversation_chain

def fetch_vector_store(user_id):
    client = qdrant_client.QdrantClient(
        os.getenv("QDRANT_HOST"),
        api_key=os.getenv("QDRANT_API_KEY")
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

# Gradio UI Functions
def process_pdf_and_initialize_chat(pdf, QDRANT_HOST, QDRANT_API_KEY):
    user_id = "new_id_1"  # Unique identifier for the user session
    
    # Process the PDF
    raw_text = get_pdf_text(pdf)
    text_chunks = get_text_chunks(raw_text)
    
    # Create vector store
    vectorstore = fetch_vector_store(user_id)

    if not vectorstore:
        vectorstore = get_vector_store(text_chunks, user_id, QDRANT_HOST, QDRANT_API_KEY)
    
    # Initialize conversation chain
    conversation_chain = get_conversation_chain(vectorstore)
    return conversation_chain

def chat_with_pdf(user_question, conversation_chain):
    prompt_template = f"<s>[INST] {user_question} [/INST]"
    response = conversation_chain({'question': prompt_template})
    
    return response["answer"]

# Initialize Gradio UI
def gradio_app():
    with gr.Blocks() as demo:
        QDRANT_HOST = os.getenv("QDRANT_HOST")
        QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

        conversation_chain = None  # Store the conversation chain

        with gr.Row():
            gr.Markdown("# Conversational AI ")

        with gr.Row():
            pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
        
        start_chat_button = gr.Button("Start Chat")
        
        with gr.Row():
            user_question_input = gr.Textbox(label="Your Question")
            submit_button = gr.Button("Submit")
            ai_response = gr.Textbox(label="AI Response", interactive=False)

        def start_chat(pdf):
            nonlocal conversation_chain
            conversation_chain = process_pdf_and_initialize_chat(pdf, QDRANT_HOST, QDRANT_API_KEY)
            return "Chat initialized. You can now ask questions!"

        def handle_question(user_question):
            if conversation_chain is None:
                return "Please upload a PDF and start the chat first.", "", ""
            
            response, time_taken, tokens_per_sec = chat_with_pdf(user_question, conversation_chain)
            return response, time_taken, tokens_per_sec

        start_chat_button.click(start_chat, inputs=[pdf_input], outputs=[ai_response])
        submit_button.click(handle_question, inputs=[user_question_input], outputs=[ai_response])

    demo.launch()

if __name__ == "__main__":
    gradio_app()
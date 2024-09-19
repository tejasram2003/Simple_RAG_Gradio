import gradio as gr
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import qdrant
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer
import qdrant_client
import torch

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


def process_pdf(pdf_file, user_id):
    if not pdf_file or not user_id:
        return "Please upload a PDF file and enter a user ID."
    
    # Process the PDF
    raw_text = get_pdf_text([pdf_file.name])
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vector_store(text_chunks, user_id)
    
    return f"PDF processed and vectors stored for user ID: {user_id}"

def chat_with_pdf(user_id, question, history):
    if not user_id:
        return "Please enter a user ID.", history
    
    vectorstore = fetch_vector_store(user_id)
    if not vectorstore:
        return "No data found for this user ID. Please upload a PDF first.", history
    
    conversation = get_conversation_chain(vectorstore)
    
    system_prompt = ""  # You can customize this if needed
    prompt_template = f'''<s>[INST] System_prompt: {system_prompt} {question} [/INST]'''
    
    response = conversation({'question': prompt_template})
    history.append((question, response['answer']))
    
    return response['answer'], history

# Gradio interface
def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# PDF Chat Interface")
        
        with gr.Tab("Upload PDF"):
            pdf_file = gr.File(label="Upload PDF")
            user_id_input = gr.Textbox(label="User ID")
            upload_button = gr.Button("Process PDF")
            upload_output = gr.Textbox(label="Upload Status")
            
            upload_button.click(process_pdf, inputs=[pdf_file, user_id_input], outputs=upload_output)
        
        with gr.Tab("Chat"):
            chat_user_id = gr.Textbox(label="User ID")
            chatbot = gr.Chatbot()
            msg = gr.Textbox(label="Your Question")
            chat_button = gr.Button("Ask")
            
            chat_button.click(chat_with_pdf, inputs=[chat_user_id, msg, chatbot], outputs=[msg, chatbot])
    
    demo.launch()

if __name__ == "__main__":
    HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_HOST = os.getenv("QDRANT_HOST")
    gradio_interface()
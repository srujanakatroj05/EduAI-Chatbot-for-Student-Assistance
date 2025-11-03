import os
import json
import pickle
import tempfile
import speech_recognition as sr
import streamlit as st
from dotenv import load_dotenv
#import pytesseract
from PIL import Image
from typing import List, Tuple
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts.chat import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

load_dotenv()

# Set up Groq API key
os.environ["GROQ_API_KEY"] = "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"


# Initialize LLaMA model
@st.cache_resource
def initialize_llama_model():
    return ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

# Initialize Gemma model using Google API
@st.cache_resource
def initialize_gemma_model():
    return ChatGroq(model_name="gemma2-9b-it", temperature=0)

# Load VectorStore dynamically based on user selection
@st.cache_resource
def load_vectorstore(index_dir: str, _embeddings) -> Tuple[FAISS, FAISS]:
    index_options = {
        0 : "./index_generated/Faculty_details/",
        1 : "./index_generated/Student_details/",
        2 : "./index_generated/Subjects/",
    }
    index1 = -1
    for i, dir in index_options.items():
        if index_dir == dir:
            index1 = i
    file = f"index_files{index1}/"
    vectorstore = FAISS.load_local(index_dir+file, embeddings=_embeddings, allow_dangerous_deserialization=True)
    with open(f"{index_dir}/retriever.pkl", "rb") as f:
        retriever_store = pickle.load(f)
    return vectorstore, retriever_store

# OCR function to extract text from image
def extract_text_from_image(image: Image.Image) -> str:
    text = pytesseract.image_to_string(image)
    return text.strip()

# Process query with LLaMA
def process_query_with_llama(query: str, documents: List[str], messages: List[dict], llm) -> str:
    if not documents:
        return "No relevant documents found."
    
    messages.append({'role': 'user', 'content': query})
    history_str = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages])
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's query based on the provided documents in strict JSON format."),
        ("user", f"Previous conversation:\n{history_str}\nQuery: {{query}}\nDocuments:\n{{documents}}"),
        ("system", """
         Your response must be a valid JSON object.
         You should greet the user if he/she wishes you.
         You should improvise the extracted content and return the improvised content as answer.
         I want you to be my assistant in answering all the questions by extraction of the data correctly from the input data.
        Example format: {{"response": "Your answer here"}}.
        Strictly avoid any additional text or comments outside the JSON object.
        """),
    ])
    
    parser = JsonOutputParser()
    chain = prompt_template | llm | parser

    try:
        result = chain.invoke({"query": query, "documents": "\n".join(documents), "messages": history_str})
        output = result.get("response", "Unable to generate a response.")
        messages.append({'role': 'assistant', 'content': output})
        return output
    except OutputParserException as e:
        st.error("Output parsing failed. Attempting to extract valid JSON...")
        llm_output = e.llm_output
        try:
            json_start = llm_output.find("{")
            json_end = llm_output.rfind("}") + 1
            valid_json = llm_output[json_start:json_end]
            parsed_output = json.loads(valid_json)
            return parsed_output.get("response", "Unable to extract a valid response.")
        except (json.JSONDecodeError, ValueError):
            st.error("Failed to parse JSON. Raw LLM output will be returned.")
            return f"Raw LLM Output:\n{llm_output}"

# Process query with Gemma model
def process_query_with_gemma(query: str, messages: List[dict], gemma_model) -> str:
    if not query:
        return "No query provided."
    
    messages.append({'role': 'user', 'content': query})
    history_str = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages])
    
    gemma_prompt_template = ChatPromptTemplate.from_messages([
        ("system", "Provide a concise and relevant answer to the user's query."),
        ("user", f"Previous conversation:\n{history_str}\nQuery: {{query}}"),
        ("system", """"
         Ensure the response is with quality of amount of content along with brief content; and addresses the query.
         You must not use the more than 300 tokens for generation.
         """),
    ])

    prompt = gemma_prompt_template.format(query=query)
    try:
        response = gemma_model.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        if not response_text.strip():
            response_text = "I couldn't generate a response at the moment."
        
        messages.append({'role': 'assistant', 'content': response_text})
        return response_text
    
    except Exception as e:
        error_message = f"Error processing query: {str(e)}"
        messages.append({'role': 'assistant', 'content': error_message})
        return error_message

# Voice input function
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak now.")
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            st.success(f"Recognized: {text}")
            return text
        except sr.UnknownValueError:
            st.warning("Could not understand audio.")
        except sr.RequestError:
            st.warning("Speech Recognition service is unavailable.")
        except sr.WaitTimeoutError:
            st.warning("No speech detected.")
    return ""

def clear_chat_history():
    st.session_state.messages = []

# Main Streamlit App
def main():
    st.set_page_config(page_title="EduAI: Chatbot for Student Assistance", layout="wide")

    st.markdown("""
        <style>
        html, body, [class*="css"]  {
            font-family: 'Segoe UI', 'Roboto', sans-serif;
        }
                </style>
    """, unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    st.sidebar.image("Mlrit.jpeg", use_container_width=True)
    
    index_options = {
        "Faculty": "./index_generated/Faculty_details/",
        "Students": "./index_generated/Student_details/",
        "Subjects": "./index_generated/Subjects/",
        "Subject_doubts": None
    }

    
    selected_index = st.sidebar.selectbox("Select Index Folder", list(index_options.keys()))

    uploaded_file = st.sidebar.file_uploader("Upload an image for OCR", type=["jpg", "png", "jpeg"])

    query = ""
    if st.sidebar.button("🎤 Use Voice Input", key="voice_input"):
        query = recognize_speech()
    st.sidebar.button("Clear Chat History", on_click=clear_chat_history)
    
    st.sidebar.header("Example Questions")
    example_questions = [
        "How can I contact the faculty?",
        "Who is the HOD of CSE-AI&ML?"
    ]
    
    for idx, question in enumerate(example_questions):
        if st.sidebar.button(question, key=f"example_{idx}"):
            query = question
    
    st.title("EduAI: Chatbot for Student Assistance")
    st.subheader("Type your query below and get department-specific answers!")
    
    # user_input = st.chat_input("Enter your query:")
    # query = user_input if user_input else query
    
    # Handle uploaded image
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_path = temp_file.name
            temp_file.write(uploaded_file.getbuffer())  # Save uploaded file to temp storage

        image = Image.open(temp_path)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Extract text using OCR
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        extracted_text = extract_text_from_image(image)
        extracted_text += "\n\n Extract the information related to the above extracted data based on Roll No./Name"
        
        query = extracted_text if extracted_text else query

        # Delete the temporary file after processing
        image.close()
        os.remove(temp_path)

    # Load the selected vector store dynamically
    user_input = st.chat_input("Enter your query:", key="user_query_input")
    query = user_input if user_input else query

    if query:
        if selected_index == "Subject_doubts":
            gemma_model = initialize_gemma_model()
            with st.spinner("Processing your query..."):
                process_query_with_gemma(query, st.session_state.messages,  gemma_model)
                for message in st.session_state.messages:
                    with st.chat_message(message['role']):
                        st.markdown(message['content'])
        else:
            index_dir = index_options[selected_index]
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore, retriever_store = load_vectorstore(index_dir, embeddings)
            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 20})
            results = retriever.get_relevant_documents(query)
            documents = [doc.page_content for doc in results]
            llm = initialize_llama_model()
            with st.spinner("Processing your query..."):
                process_query_with_llama(query, documents, st.session_state.messages, llm)
                for message in st.session_state.messages:
                    with st.chat_message(message['role']):
                        st.markdown(message['content'])
            # if documents:
            #     st.write("### Retrieved Documents:")
            #     for i, doc in enumerate(documents, 1):
            #         with st.expander(f"Document {i}"):
            #             st.write(doc)

if __name__ == "__main__":
    main()
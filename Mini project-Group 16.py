import os
import openai
import gradio as gr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from storeDocument import process_and_store_documents, load_documents_from_folder, split_documents, create_vector_store


# Directly passing the API key to the OpenAI model
#api_key="sk-<your-openai-api-key>"
#os.environ['OPENAI_API_KEY'] = api_key
openai.api_key = os.getenv('OPENAI_API_KEY')
pdf_doc_folder_path = r'./pwcdocument'  # Update this path to the correct folder
db_storage_path = r'./dbstorage'  # Update this path to the correct folder

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Always say "thanks for asking!" at the end of the answer.
{context}
Question: {query}
Helpful Answer:"""
QA_PROMPT_TEMPLATE =    PromptTemplate(input_variables=["context", "query"], template=template)

# User access control (admin and end-user roles)

user_access_control = {
    "admin": {
        "username": "admin",
        "password": "adminpassword",
        "access": ["./pwcdocument/pwc_employee_manual_.pdf", "./pwcdocument/pwc-mergers-acquisitions.pdf"]
    },
    "end-user": {
        "username": "user",
        "password": "userpassword",
        "access": ["./pwcdocument/pwc_employee_manual_.pdf"]
    }
}

vectordb = process_and_store_documents(pdf_doc_folder_path, db_storage_path)

# Initialize the OpenAI model using ChatOpenAI (with GPT-4 or the desired model)
llm = ChatOpenAI(model_name="gpt-4", temperature=0)  # Update the model name if needed


# Function to authenticate user
def authenticate_user(username, password):
    for role, user_data in user_access_control.items():
        if user_data['username'] == username and user_data['password'] == password:
            return role, user_data['access']
    return None, []

# Function for user login
def login(username, password):
    role, accessible_files = authenticate_user(username, password)
    if role:
        return (
            f"Login successful. Welcome {role}!",
            role,
            accessible_files,
            gr.update(visible=True)  # Show Query Document tab on success
        )
    return "Invalid username or password. Please try again.", None, [], gr.update(visible=False)


def get_query_response(query, role, accessible_files):
    if role is None:
        return "You must log in first."
    print(f"Applicable Documents {accessible_files}")
    if query:
        print(f"Received query: {query}")  # Debug log
        filter_criteria = {"source": {"$in": accessible_files}}
        # retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 5},filter={"source":'./pwcdocument/pwc_employee_manual_.pdf'}   )
        # docs = retriever.invoke(query)
        # for doc in docs:
        #     print(f"Metadata {doc.metadata} \n Content {doc.page_content}")
        
        docs = vectordb.similarity_search(
        query,
        k=5,
        filter=filter_criteria     # manually passing metadata, using metadata filter.
        )
        for doc in docs:
            print(f"Metadata {doc.metadata} ")
        
        return get_llm_response(query,docs)
    
    return "No query entered."

def get_llm_response(query, context):
      

    prompt = QA_PROMPT_TEMPLATE.format(context=context, query=query)
    print("\n Prompt: ", prompt)
    response = llm.invoke(prompt)
    # OpenAI model response
    
    return response.content
    
# Build the Gradio UI
def build_ui():
    with gr.Blocks() as demo:
        # Login Tab
        with gr.Tab("Login") as login_tab:
            gr.Markdown("**Login Form**")
            username = gr.Textbox(label="Username")
            password = gr.Textbox(label="Password", type="password")
            login_button = gr.Button("Login")
            output_login = gr.Textbox(label="Login Status")
            role, accessible_files = gr.State(), gr.State()
            
            # Query Document Tab - Initially hidden
            with gr.Tab("Query Document", visible=False) as query_tab:
                gr.Markdown("**Query the Document Knowledge Base**")
                query_input = gr.Textbox(label="Enter your query:")
                output_query = gr.Textbox(label="Query Response")
                submit_query = gr.Button("Submit Query")
                
                # Process the query submission after login
                submit_query.click(
                    fn=get_query_response, 
                    inputs=[query_input, role, accessible_files], 
                    outputs=[output_query]
                )

            # Login button logic for tab visibility
            login_button.click(
                fn=login,
                inputs=[username, password],
                outputs=[output_login, role, accessible_files, query_tab]
            )

    demo.launch(share=True)

# Run the UI
if __name__ == "__main__":
    build_ui()
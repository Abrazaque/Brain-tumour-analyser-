import base64
import streamlit as st
from mistralai import Mistral
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.agents import AgentExecutor
from langchain.vectorstores import Milvus
from langchain.embeddings import CohereEmbeddings
from langchain.prompts import PromptTemplate

def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        st.error(f"Error: The file {image_path} was not found.")
        return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def retrieve_relevant_documents(image_description, vectorstore):
    """Retrieve relevant documents related to the MRI description from the vectorstore."""
    query = image_description
    return vectorstore.similarity_search(query, k=5)

def initialize_ai_agent(api_key, vectorstore):
    llm = OpenAI(temperature=0)
    tools = [
        Tool(
            name="MRI Diagnosis Retrieval",
            func=lambda query: retrieve_relevant_documents(query, vectorstore),
            description="Retrieve relevant medical documents and diagnoses based on the MRI description"
        )
    ]
    
    agent = initialize_agent(tools, llm, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    return agent

def main():
    st.set_page_config(page_title="Advanced MRI Image Analysis", page_icon="🧠", layout="centered")
    st.title("🧠 Advanced MRI Image Analysis with Pixtral")

    api_key = '30l6CEgQprxyWVj4yoiDl0SSuCQBa9EE'
    model = "pixtral-large-latest"

    st.sidebar.header("Settings")
    api_key = st.sidebar.text_input("API Key", value='30l6CEgQprxyWVj4yoiDl0SSuCQBa9EE', type="password")
    model = st.sidebar.text_input("Model", value=model)

    if not api_key:
        st.warning("Please provide a valid API key in the settings sidebar.")
        return

    uploaded_file = st.file_uploader("Upload an MRI image (JPEG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        try:
            st.image(uploaded_file, caption="Uploaded MRI Image", use_column_width=True)
            base64_image = base64.b64encode(uploaded_file.read()).decode('utf-8')

            image_description = "A detailed MRI scan of the brain showing potential abnormalities."

            vectorstore = Milvus("milvus://localhost:19530", CohereEmbeddings())
            relevant_documents = retrieve_relevant_documents(image_description, vectorstore)

            #(retrieval + reasoning)
            agent = initialize_ai_agent(api_key, vectorstore)

            full_analysis_input = "\n".join([image_description] + [doc['content'] for doc in relevant_documents])

            with st.spinner("Analyzing the image, please wait..."):
                agent_response = agent.run(full_analysis_input)

            st.success("Analysis Completed")
            st.subheader("AI Diagnosis & Insights:")
            st.write(agent_response)

        except Exception as e:
            st.error(f"Error: {e}") 

if __name__ == "__main__":
    main()


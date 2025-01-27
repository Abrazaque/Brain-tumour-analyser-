# 🧠 Advanced MRI Image Analysis with Pixtral

This repository contains a Streamlit-based web application for analyzing MRI images using AI-powered tools. The application leverages advanced language models, embeddings, and vector databases to provide diagnostic insights and retrieve relevant medical documents.

## Features

- **Upload MRI Images**: Supports JPEG and PNG formats for MRI image uploads.
- **AI Diagnosis**: Utilizes OpenAI's GPT-based models to generate diagnostic insights for the uploaded image.
- **Relevant Document Retrieval**: Searches a vector database (Milvus) for documents related to the image description.
- **Interactive UI**: Intuitive Streamlit interface with sidebar configuration for API keys and model selection.

## How It Works

1. **Image Upload**: Users upload an MRI image.
2. **Image Analysis**: The application encodes the image and generates a description.
3. **Document Retrieval**: Relevant documents are fetched from the vector store using the description.
4. **AI Agent**: An AI agent combines the description and documents to generate insights.
5. **Output**: Diagnosis and related information are displayed in the UI.

## Prerequisites

- Python 3.7 or later
- A valid OpenAI API key
- Milvus setup for storing and retrieving embeddings
- Streamlit installed

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/your-username/advanced-mri-analysis.git
    cd advanced-mri-analysis
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Start the application:
    ```bash
    streamlit run app.py
    ```

## Configuration

- API Key: Input your OpenAI API key in the settings sidebar of the application.
- Model: Specify the model for generating insights (default: `pixtral-large-latest`).

## File Structure

- `app.py`: Main application script.
- `requirements.txt`: List of required Python libraries.

## Key Libraries

- [Streamlit](https://streamlit.io/) for the web interface.
- [LangChain](https://langchain.com/) for building the AI pipeline.
- [Milvus](https://milvus.io/) for vector-based document retrieval.
- [OpenAI](https://openai.com/) for GPT-based language models.

## Example Usage

1. Upload an MRI image via the interface.
2. Wait for the analysis to complete.
3. View the diagnostic insights and related documents.

## License

This project is licensed under the [MIT License](LICENSE).

# RAG Question Answering

This project provides a pipeline for answering questions based on the context of a PDF file using Retrieval-Augmented Generation (RAG).

The work was done using LangChain and a Groq API key:

  - https://python.langchain.com/v0.2/docs/introduction
  - https://console.groq.com/docs/quickstart

## Requirements

- See `requirements.txt` for the full list of dependencies.

## Usage

Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

Install the required packages:
```bash
pip install -r requirements.txt
```

To run the main script, use the following command:
```bash
python main.py --pdf_filepath path/to/your/pdf --query "Your question here"
```

**Note**: There must be a Groq API Key setup:

  - Create a `.env` file in the root directory.
  - Add your Groq API key: ```GROQ_API_KEY=your_groq_api_key```

## TO DO:
  - [ ] Manage multiple PDF files
  - [ ] Improve code generalization (_for example metadata information_)
  - [ ] Add Streamlit interactive interface

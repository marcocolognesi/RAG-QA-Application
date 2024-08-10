import os

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from utils import pdf_loader


def main():
    # Load Groq API Key
    load_dotenv()
    os.getenv('GROQ_API_KEY')

    # Define the PDF file path and the text splitter
    pdf_filepath = 'example.pdf'

    # Load the PDF file
    content = pdf_loader(pdf_filepath)

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50,
        separators=['.', ',', ' ', '!', '?', ';', ':']
    )

    document_list = []
    for entry in content:
        # Split the text
        text_split = text_splitter.split_text(entry['text'])
        document = entry['document']
        page = entry['page']
        # Create a Document object for each split
        for idx, split in enumerate(text_split):
            doc = Document(
                page_content=split,
                metadata={
                    'document': document,
                    'page': page,
                    'split': idx
                }
            )
            document_list.append(doc)

    # Define the vector store
    llm = ChatGroq(model='llama3-8b-8192', temperature=0)
    vector_store = FAISS.from_documents(
        documents=document_list,
        embedding=HuggingFaceEmbeddings(model='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    )
    retriever = vector_store.as_retriever()

if __name__ == "__main__":
    main()

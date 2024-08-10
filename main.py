import os
from argparse import ArgumentParser

from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from utils import pdf_loader, create_documents_from_chunks


def main_parser():
    # Initialize the argument parser
    parser = ArgumentParser(prog='RAG Question Answering',
                            description='Answer questions based on the context of a PDF file.')
    # Add arguments
    parser.add_argument('--pdf_filepath', type=str, required=True,
                        help='The path to the PDF file.')
    parser.add_argument('--llm_model', type=str, required=False,
                        default='llama3-8b-8192',
                        help='The LLM model to use for question answering.'
                             'See the list of available models at https://console.groq.com/docs/models')
    parser.add_argument('--query', type=str, required=True,
                        help='The size of the text chunks.')
    return parser


def main():
    # Parse the arguments
    parser = main_parser()
    args = parser.parse_args()

    # Load Groq API Key
    load_dotenv()
    os.getenv('GROQ_API_KEY')

    # Load the PDF file
    content = pdf_loader(args.pdf_filepath)

    # Define the text splitter and create Document objects
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50,
        separators=['.', ',', ' ', '!', '?', ';', ':']
    )
    document_list = create_documents_from_chunks(content=content, text_splitter=text_splitter)

    # Initialize vector store and retriever
    vector_store = FAISS.from_documents(
        documents=document_list,
        embedding=HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    )
    retriever = vector_store.as_retriever()

    # Initialize LLM model and prompt
    llm = ChatGroq(model=args.llm_model, temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                'system',
                '''
                Answer any use questions based solely on the context below:
                <context>
                {context}
                </context>
                
                Identify the language of the question and answer in the same language. For example,
                if the question is in English, the answer should be in English and same for other languages.
                '''
            ),
            (
                'human', '{input}'
            )
        ]
    )

    # Create RAG chain
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    # Invoke the RAG chain
    result = rag_chain.invoke({'input': args.query})
    print(f'=== Output ===\nQuery: {args.query}\nAnswer: {result['answer']}')


if __name__ == "__main__":
    main()

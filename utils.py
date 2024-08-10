from pathlib import Path

from pypdf import PdfReader
from langchain_community.docstore.document import Document


def cast_str_to_path(filepath: str | Path) -> Path:
    """Cast a string to a Path object."""
    return Path(filepath) if isinstance(filepath, str) else filepath


def pdf_loader(pdf_filepath: str | Path) -> list[dict]:
    """Load a PDF file and extract the text from each page.

    Args:
        pdf_filepath: The path to the PDF file.

    Returns:
        list[dict]: A list of dictionaries containing the text, document, and page number.
    """
    # Cast the input to a Path object
    pdf_filepath = cast_str_to_path(pdf_filepath)

    # Load the PDF file
    pdf_reader = PdfReader(pdf_filepath)

    # Extract the text from each page
    content = []
    for num_page in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[num_page]
        text = page.extract_text()
        processed_text = text.replace('\n', ' ')
        content.append(
            {
                'text': processed_text,
                'document': pdf_filepath.name,
                'page': num_page
            }
        )
    return content


def create_documents_from_chunks(content: list[dict], text_splitter) -> list[Document]:
    """Create Document objects from the extracted text content.

    First, it iterates through each entry in the input list, splitting the text using the provided text splitter.
    Then, it creates a Document object for each split, adding metadata about the source document and page number.

    Args:
        content (list[dict]): A list of dictionaries containing the text, name of source document and page number.
        text_splitter: The Langchain text splitter object.

    Returns:
        list[Document]: A list of Document objects.
    """
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
    return document_list

from pathlib import Path

from pypdf import PdfReader


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

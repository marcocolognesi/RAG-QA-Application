from utils import pdf_loader


def main():
    # Define the PDF file path and the text splitter
    pdf_filepath = 'example.pdf'

    # Load the PDF file
    content = pdf_loader(pdf_filepath)
    print(content)


if __name__ == "__main__":
    main()

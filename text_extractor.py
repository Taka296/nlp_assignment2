from pdf2image import convert_from_path
import pytesseract
import os

class TextExtractor:

    def __init__(self, input_directory='pdfs', output_directory='texts'):
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

    def extract_text(self):

        pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd

        # Define directory
        pdf_directory = self.input_directory
        output_directory = self.output_directory

        # Create directory if not exist
        self.create_directory(pdf_directory)
        self.create_directory(output_directory)

        # Get pdf file names
        pdf_file_names = [f.strip('.pdf') for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
        output_files = []

        # Extract texts from PDFs with OCR
        for file_name in pdf_file_names:
            pdf_file = file_name + '.pdf'
            output_file = file_name + '.txt'
            pdf_path = os.path.join(pdf_directory, pdf_file)
            output_path = os.path.join(output_directory, output_file)
            output_files.append(output_path)

            # If the text file is already created, then skip the process
            if os.path.exists(output_path):
                continue
            self.extract_left_side_text(pdf_path, output_path)

        return output_files

    def extract_left_side_text(self, pdf_path, output_txt_path):

        # Convert PDFs into images
        images = convert_from_path(pdf_path)
        extracted_texts = []

        for img in images:
            width, height = img.size

            # Crop the image because only the left side of PDF contains English
            left_half = img.crop((0, 0, width // 2, height))

            # Apply OCR
            text = pytesseract.image_to_string(left_half, lang="eng")
            extracted_texts.append(text)

        # Save the result as a text file
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(extracted_texts))

    def create_directory(self, directory):
        os.makedirs(directory, exist_ok=True)


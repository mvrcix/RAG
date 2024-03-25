import fitz
from PIL import Image
import os 
import sys
import io
import tabula
import pandas as pd
import pytesseract
from unstructured.partition.pdf import partition_pdf

# Replace sqlite3 with pysqlite3
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


path = '/home/vqa/masterthesis/ourproject/multimodal_rag/extracted_data'
pdf_path = "/home/vqa/masterthesis/ourproject/multimodal_rag/extracted_data/LLaVA.pdf"

def extract_and_save_images(pdf_path, output_dir):
    images = partition_pdf(
        filename=pdf_path,
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        image_output_dir_path=output_dir,)
    return images

# extract_and_save_images(pdf_path, path)

def extract_text_from_image(image):
    # Convert the image to grayscale
    grayscale_image = image.convert('L')
    # Use Tesseract OCR to extract text from the image
    text = pytesseract.image_to_string(grayscale_image)
    return text

def extract_text_elements(pdf_path):
    text_elements = []
    image_elements = []
    table_elements = []

    # Open the PDF file using PyMuPDF (fitz)
    pdf_document = fitz.open(pdf_path)

    for page_number in range(pdf_document.page_count):
        page = pdf_document.load_page(page_number)
        # Extract images from the page
        images = page.get_images(full=True)
        for image_index, image_info in enumerate(images, start=1):
            # Extract image data
            xref = image_info[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]

            # Convert the image bytes to a PIL Image
            pil_image = Image.open(io.BytesIO(image_bytes))
            # Extract text from the image using OCR
            image_text = extract_text_from_image(pil_image)
            image_elements.append({"type": "image", "text": image_text})

        # Extract text from the page that's not in the image
        if not page.get_images(full=True):
            page_text = page.get_text("text")
            if page_text.strip():  # Skip empty text pages
                text_elements.append({"type": "text", "text": page_text})
    # Extract tables:
        tables = tabula.read_pdf(pdf_path, pages=page_number + 1, multiple_tables=True)
        if tables:
            table_elements.extend(tables)

    return table_elements, text_elements, image_elements


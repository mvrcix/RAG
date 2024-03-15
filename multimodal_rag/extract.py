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

extract_and_save_images(pdf_path, path)

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




pdf_path = "/home/vqa/masterthesis/ourproject/multimodal_rag/extracted_data/LLaVA.pdf"
table_elements, text_elements, image_elements = extract_text_elements(pdf_path)

# print("Table elements:", table_elements)
# print("Text elements:", text_elements)
# print("Image elements:", image_elements)



# Original code 
# from unstructured.partition.pdf import partition_pdf
# path = "/home/vqa/masterthesis/ourproject/multimodal_rag/extracted_data/"

# raw_pdf_elements = partition_pdf(
#     filename=path + "output.pdf",
#     extract_images_in_pdf=True,
#     infer_table_structure=True,
#     chunking_strategy="by_title",
#     max_characters=4000,
#     new_after_n_chars=3800,
#     combine_text_under_n_chars=2000,
#     image_output_dir_path=path,)

# tables = []
# texts = []
# for element in raw_pdf_elements:
#     if "unstructured.documents.elements.Table" in str(type(element)):
#         tables.append(str(element))
#     elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
#         texts.append(str(element))




#___________________________________________________________________

# def remove_substrings(original_string, substrings_to_remove):
#     for substring in substrings_to_remove:
#         original_string = original_string.replace(substring, '')
#     return original_string


# def extract_content(path, file_path):
#     pdf_file = os.path.join(path, file_path)
    
#     # Create a PDF document object
#     doc = fitz.open(pdf_file)
    
#     # Initialize lists to store extracted data
#     table_list = []
#     text_list = []
#     image_list = []
    
#     # Extract tables
#     table_path = path + '/tables'
#     os.makedirs(table_path, exist_ok=True)
#     tables = tabula.read_pdf(pdf_file, pages='all', multiple_tables=True)

#     # Iterate through the extracted tables
#     for table_number, table in enumerate(tables):
#         # Convert each table to a CSV file
#         csv_filename = f'table_{table_number + 1}.csv'
#         csv_path = os.path.join(table_path, csv_filename)
#         table.to_csv(csv_path, index=False)
#         # print(f"Table {table_number + 1} converted and saved to {csv_path}")

#     # Iterate through CSV files in the folder
#     for filename in os.listdir(table_path):
#         if filename.endswith(".csv"):
#             csv_path = os.path.join(table_path, filename)
#             df = pd.read_csv(csv_path)
#             separator = ' '
            
#             # df_cleaned = df.applymap(lambda x: x.strip(' \t') if isinstance(x, str) else x)
#             csv_as_single_row_string = df.to_string(index=False, header=False, col_space=3, justify='left', na_rep='').replace('\n', separator)
#             csv_as_single_row_string = " ".join(csv_as_single_row_string.split())
#             with open(csv_path, 'w', newline='') as file:
#                 file.write(csv_as_single_row_string)
#             table_list.append(csv_as_single_row_string)

#     for page_number in range(doc.page_count):
#         page = doc[page_number]
        
#         # Extract text from the page (including text within images)
#         page_text = page.get_text("text")
#         text_list.append(page_text)
#         text_list = [s.replace('\n', ' ') for s in text_list]
#         text_list = [remove_substrings(s, table_list) for s in text_list]

#         # Extract images
#         images = page.get_images(full=True)
#         for img_index, img in enumerate(images):
#             img_index += 1
#             image_index = img[0]
#             base_image = doc.extract_image(image_index)
#             image_bytes = base_image["image"]
#             image = Image.open(io.BytesIO(image_bytes))
#             image_path = os.path.join(path, f"page_{page_number + 1}_image_{img_index}.png")
#             image.save(image_path)
#             image_list.append(image_path)
        
#     doc.close()
#     return table_list, text_list



# ___________________________________________________________






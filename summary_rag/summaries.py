from langchain_community.chat_models import ChatOllama
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from extract import extract_text_elements
from extract import extract_and_save_images
import os
import subprocess
import json
import re
from natsort import natsorted 

pdf_path = "/home/vqa/masterthesis/ourproject/summary_rag/summary_data/output.pdf"
path = "/home/vqa/masterthesis/ourproject/summary_rag/summary_data/summaries/"

table_elements, text_elements, image_text_elements = extract_text_elements(pdf_path)
extract_and_save_images(pdf_path, path)

# TEXT SUMMARIZATION

# summarizing tables and text 
def generate_summary(elements):
    summaries = []
    model = ChatOllama(temperature=0, model="llama2:7b-chat")

    for element in elements:
        text_splitter = CharacterTextSplitter()
        texts = text_splitter.split_text(element['text'])
        chunks = [Document(page_content=e) for e in texts if e != ""]
        chain = load_summarize_chain(model, chain_type='map_reduce')
        summaries.append(chain.run(chunks))

    return summaries

# IMAGES SUMMARIZATION

def summarize_images(img_path):

    # Loop through each image in the directory
    for img in natsorted(os.listdir(img_path)):
        print(img)
        if img.endswith(".jpg"):
            # Extract the base name of the image without extension
            base_name = os.path.splitext(img)[0]

            # Define the output file name based on the image name
            output_file = os.path.join(img_path, f"{base_name}.txt")

            # Execute the command and save the output to the defined output file
            command = [
                "python3",
                "/home/vqa/masterthesis/ourproject/runllava.py",
                "--i",
                os.path.join(img_path, img),
                "--p",
                prompt,
                "--o",
                output_file,
                ]

            subprocess.run(command)

# read each summary and clean it
def read_and_clean_summaries(image_path, delimiter):  
    image_summaries = []

    for file in natsorted(os.listdir(image_path)):
        # getting summary files - all the .txt files
        if file.endswith(".txt"):
            file_path = os.path.join(image_path, file)
            with open(file_path, "r") as file:
                summary = file.read()
                # Remove single newlines and multiple consecutive newlines
                summary = re.sub(r'\n+', '\n', summary)
                image_summaries.append(summary)

    return image_summaries


if __name__ == "__main__":

    image_text_summaries = generate_summary(image_text_elements)
    text_summaries = generate_summary(text_elements)
    table_summaries = generate_summary(table_elements)


    # extracted images directory
    img_path = "/home/vqa/masterthesis/ourproject/summary_rag/summary_data/summaries/"

    prompt = """You are an assistant whose task is to describe images for developing a Visual Question Answering tool.
    Provide a comprehensive description of the image, including all relevant details and elements like graphs, charts, diagrams, or textual information.
    Describe any notable features or patterns observed. Ensure that the description is clear, detailed, and covers all aspects of the image to facilitate understanding it."""

    # summarizing images
    summarize_images(img_path, prompt)

    delimiter = "~~~"

    image_summaries = read_and_clean_summaries(img_path, delimiter)

print('k', image_summaries)
print("Length of image_text_elements:", len(image_text_elements))
print("Length of image_text_summaries:", len(image_text_summaries))

# saving summaries to text files
def save_summaries(folder):
    with open(os.path.join(folder, "text_elements.json"), "w") as f:
        json.dump(text_elements, f)
    
    with open(os.path.join(folder, "table_elements.json"), "w") as f:
        json.dump(table_elements, f)

    with open(os.path.join(folder, "image_text_elements.json"), "w") as f:
        json.dump(image_text_elements, f)

    with open(os.path.join(folder, "text_summaries.txt"), "w") as f:
        f.writelines([summary + delimiter for summary in text_summaries])

    with open(os.path.join(folder, "image_text_summaries.txt"), "w") as f:
        f.writelines([summary + delimiter for summary in image_text_summaries])

    with open(os.path.join(folder, "table_summaries.txt"), "w") as f:
        f.writelines([summary + delimiter for summary in table_summaries])

    with open(os.path.join(folder, "image_summaries.txt"), "w") as f:
        f.writelines([summary + delimiter for summary in image_summaries])

output_folder = "/home/vqa/masterthesis/ourproject/summary_rag/summary_data/"
save_summaries(output_folder)

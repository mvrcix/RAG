# from extract import extract_content
from retriever import create_retriever
from decodeencode import is_base64, split_image_text_types
import base64
from operator import itemgetter
from langchain_core.messages import HumanMessage , SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
import os 
import sys 
from extract import extract_text_elements
from extract import extract_and_save_images
import os

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

pdf_path = "/home/vqa/masterthesis/ourproject/multimodal_rag/extracted_data/LLaVA.pdf"
path = '/home/vqa/masterthesis/ourproject/multimodal_rag/extracted_data/'

table_elements, text_elements, image_text_elements = extract_text_elements(pdf_path)
extract_and_save_images(pdf_path, path)

# path = "/home/vqa/masterthesis/ourproject/multimodal_rag/extracted_data/" 
# image_output_dir_path = os.path.join(path, "extracted_images")
# image_output_dir_path = "/home/vqa/masterthesis/ourproject/multimodal_rag/extracted_data/"
# os.makedirs(image_output_dir_path, exist_ok=True)
# filename = os.path.join(path, "LLaVA.pdf") 
# text_list, image_list = extract_content(filename, image_output_dir_path)


# path = "/home/vqa/masterthesis/ourproject/multimodal_rag/extracted_data/"
# file_path = 'LLaVA.pdf'
# table_list, text_list = extract_content(path, file_path)

texts = []
for dictionary in text_elements + image_text_elements:
    if 'text' in dictionary:
        texts.append(dictionary['text'])


retriever = create_retriever(path, texts) 

def prompt_func(data_dict):
    # Joining the context texts into a single string
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    print(formatted_texts)
    messages = []

    # Adding image(s) to the messages if present
    if data_dict["context"]["images"]:
        image_message = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{data_dict['context']['images'][0]}"
            },
        }
        # image_url = f"data:image/jpeg;base64,{data_dict['context']['images'][0]}"
        # image_message = {
        #     "type": "image_url",
        #     "image_url": image_url,
        # }
        messages.append(image_message)

    # Adding the text message for analysis
    text_message = {
        "type": "text",
        "text": (
            "Provide a detailed answer to the user question based on the provided context."
            f"User question: {data_dict['question']}\n\n"
            "Context:\n"
            f"{formatted_texts}"
        ),
    }

    messages.append(text_message)
    return [HumanMessage(content=messages)]




# model = ChatOpenAI(temperature=0, model="gpt-4-vision-preview", max_tokens=1024)
# model = ChatOllama(model="llava:13b", temperature=0)
model = ChatOllama(model="llava")
# model = ChatOllama(model="llama2:7b-chat")



# RAG pipeline
chain = (
    {
        "context": retriever | RunnableLambda(split_image_text_types),
        "question": RunnablePassthrough(),
    }
    | RunnableLambda(prompt_func)
    | model
    | StrOutputParser()
)


from IPython.display import HTML, display


def plt_img_base64(img_base64):
    # Create an HTML img tag with the base64 string as the source
    image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'

    # Display the image by rendering the HTML
    display(HTML(image_html))


docs = retriever.get_relevant_documents("chicken nuggets", k=2)
for doc in docs:
    page = doc.page_content
    if is_base64(doc.page_content):
        plt_img_base64(doc.page_content)
    # else:
    #     print(doc.page_content)


print(chain.invoke("Explain the chicken nugget picture."))


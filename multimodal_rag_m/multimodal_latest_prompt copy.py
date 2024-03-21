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

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# path = "/home/vqa/masterthesis/ourproject/multimodal_rag/extracted_data/" 
# image_output_dir_path = os.path.join(path, "extracted_images")
# image_output_dir_path = "/home/vqa/masterthesis/ourproject/multimodal_rag/extracted_data/"
# os.makedirs(image_output_dir_path, exist_ok=True)
# filename = os.path.join(path, "LLaVA.pdf") 
# text_list, image_list = extract_content(filename, image_output_dir_path)



from unstructured.partition.pdf import partition_pdf
path = "/home/vqa/masterthesis/ourproject/multimodal_rag/extracted_data/"


raw_pdf_elements = partition_pdf(
    filename=path + "LLaVA.pdf",
    extract_images_in_pdf=True,
    infer_table_structure=True,
    chunking_strategy="by_title",
    max_characters=4000,
    new_after_n_chars=3800,
    combine_text_under_n_chars=2000,
    image_output_dir_path=path,)

tables = []
texts = []
for element in raw_pdf_elements:
    if "unstructured.documents.elements.Table" in str(type(element)):
        tables.append(str(element))
    elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
        texts.append(str(element))

# path = "/home/vqa/masterthesis/ourproject/multimodal_rag/extracted_data/"
# file_path = 'LLaVA.pdf'
# table_list, text_list = extract_content(path, file_path)

retriever = create_retriever(path, texts) 


def prompt_func(data_dict):
    # Joining the context texts into a single string
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = []

    # Adding image(s) to the messages if present
    if data_dict["context"]["images"]:
        image_message = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{data_dict['context']['images'][0]}"
            },
        }
        messages.append(image_message)

    # Adding the text message for analysis
    text_message = {
        "type": "text",
        "text": (
            "Try your best to answer the user question based on the provided context."
            f"User-provided keywords: {data_dict['question']}\n\n"
            "Context:\n"
            f"{formatted_texts}"
        ),
    }

    messages.append(text_message)
    return [HumanMessage(content=messages)]


model = ChatOllama(model="llava")

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
print('length', len(docs))

for doc in docs:
    page = doc.page_content
    print('this....', doc.page_content)
    if is_base64(doc.page_content):
        plt_img_base64(doc.page_content)
    # else:
    #     print(doc.page_content)


print(chain.invoke("Explain the chicken nugget picture."))



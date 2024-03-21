import os
from langchain_community.vectorstores import Chroma
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
import os
import sys

from extract import extract_text_elements
from extract import extract_and_save_images


__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


def create_retriever(path, text_list):
    vectorstore = Chroma(collection_name="multimodaldata", embedding_function=OpenCLIPEmbeddings())
    store = InMemoryStore()  
    id_key = "doc_id"
    retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key=id_key,)
    image_uris = sorted([os.path.join(path, image_name) for image_name in os.listdir(path) if image_name.endswith(".jpg")])
    vectorstore.add_images(uris=image_uris)
    vectorstore.add_texts(texts=text_list)
    retriever = vectorstore.as_retriever()

    return retriever, vectorstore



__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

pdf_path = "/home/vqa/masterthesis/ourproject/multimodal_rag/extracted_data/LLaVA.pdf"
path = '/home/vqa/masterthesis/ourproject/multimodal_rag/extracted_data/'

table_elements, text_elements, image_text_elements = extract_text_elements(pdf_path)
extract_and_save_images(pdf_path, path)

texts = []
for dictionary in text_elements + image_text_elements:
    if 'text' in dictionary:
        texts.append(dictionary['text'])


retriever, vectorstore = create_retriever(path, texts) 
vectorstore.get()

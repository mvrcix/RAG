import os
from langchain_community.vectorstores import Chroma
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
import os
import sys

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


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


def create_retriever(path, text_list):
    vectorstore = Chroma(collection_name="multimodaldata", embedding_function=OpenCLIPEmbeddings())
    # The storage layer for the parent documents
    store = InMemoryStore()  # <- Can we extend this to images
    id_key = "doc_id"
    # The retriever (empty to start)
    retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key=id_key,)

    # Get image URIs with .jpg extension only
    image_uris = sorted([os.path.join(path, image_name) for image_name in os.listdir(path) if image_name.endswith(".jpg")])

    # Add images
    vectorstore.add_images(uris=image_uris)
    data_dict['context']['images'][0]
    # Add documents
    vectorstore.add_texts(texts=text_list)
    # Make retriever
    retriever = vectorstore.as_retriever()
    
    return retriever

retriever = create_retriever(path, texts) 
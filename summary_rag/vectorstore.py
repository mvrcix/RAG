import uuid
import os
import json
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema.document import Document
from datasets import Dataset
from langchain_community.llms import Ollama

output_folder = "/home/vqa/masterthesis/ourproject/summary_rag/summary_data"

def read_elements(output_folder, filename):
    with open(os.path.join(output_folder, filename), "r") as json_file:
        return json.load(json_file)

def read_summaries(output_folder, filename, delimiter):
    with open(os.path.join(output_folder, filename), "r") as f:
        return [s.strip() for s in f.read().split(delimiter) if s.strip()]

delimiter = "~~~"
text_elements = read_elements(output_folder, "text_elements.json")
table_elements = read_elements(output_folder, "table_elements.json")
image_text_elements = read_elements(output_folder, "image_text_elements.json")
image_text_elements = [i['text'] for i in image_text_elements]
text_summaries = read_summaries(output_folder, "text_summaries.txt", delimiter)
image_text_summaries = read_summaries(output_folder, "image_text_summaries.txt", delimiter)
table_summaries = read_summaries(output_folder, "table_summaries.txt", delimiter)
image_summaries = read_summaries(output_folder, "image_summaries.txt", delimiter)

print('wow', image_summaries)
# The vectorstore to use to index the child chunks
vectorstore = Chroma(collection_name="summaries", embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
# The storage layer for the parent documents
store = InMemoryStore()  # <- Can we extend this to images
id_key = "doc_id"
# The retriever (empty to start)
retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key=id_key,)

def add_data(text_summaries, image_text_summaries, table_summaries, text_elements, image_text_elements, table_elements, cleaned_img_summary, retriever):
    # Add texts
    if text_summaries:
        doc_ids = [str(uuid.uuid4()) for _ in text_elements]
        summary_texts = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(text_summaries)]
        retriever.vectorstore.add_documents(summary_texts)
        retriever.docstore.mset(list(zip(doc_ids, text_elements)))

    # Add image texts
    if image_text_summaries:
        doc_ids = [str(uuid.uuid4()) for _ in image_text_elements]
        summary_texts = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(image_text_summaries)]
        retriever.vectorstore.add_documents(summary_texts)
        retriever.docstore.mset(list(zip(doc_ids, image_text_elements)))

    # Add tables
    if table_summaries:
        table_ids = [str(uuid.uuid4()) for _ in table_elements]
        summary_tables = [
            Document(page_content=s, metadata={id_key: table_ids[i]})
            for i, s in enumerate(table_summaries)]
        retriever.vectorstore.add_documents(summary_tables)
        retriever.docstore.mset(list(zip(table_ids, table_elements)))

    # Add images
    if image_summaries:
        img_ids = [str(uuid.uuid4()) for _ in cleaned_img_summary]
        summary_img = [
            Document(page_content=s, metadata={id_key: img_ids[i]})
            for i, s in enumerate(cleaned_img_summary)]
        retriever.vectorstore.add_documents(summary_img)
        retriever.docstore.mset(list(zip(img_ids, cleaned_img_summary)))  

    return retriever


retriever = add_data(text_summaries, image_text_summaries, table_summaries, text_elements, image_text_elements, table_elements, image_summaries, retriever)


from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama


# Prompt template
template = """Answer the question based only on the following context, which can include text and tables:
{context}
Question: {question}
Helpful Answer:
"""
prompt = PromptTemplate.from_template(template)

# Option 1: LLM
# model = ChatOllama(model="llama2:7b-chat")
# Option 2: Multi-modal LLM
# model = LLaVA
from langchain.llms import Ollama
llm = Ollama(model="llama2",
                verbose=True)
print(f"Loaded LLM model {llm.model}")

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser())


# from langchain.chains import RetrievalQA
# chain = RetrievalQA.from_chain_type(llm,
#                                   chain_type="stuff",
#                                   chain_type_kwargs={"prompt": prompt},
#                                   retriever= retriever
#  


question = "Which battery does the CD player use?"
answer = chain.invoke("Which battery does the CD player use?")
print(answer)
print('lol', retriever.get_relevant_documents(question))




# EVALUATION


from ragas.evaluation import evaluate

# from ragas.llms import LangchainLLM
# vllm = LangchainLLM(llm=llm)

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
) 

questions = ['Where is the CLASS 1LASER PRODUCT?', 'What should I pay attention to when I use headphones with this CD Player?', 'How to slide the compartment door?','How to make the headphone connection?', 'How can I use the open switch?', 'How to enter the next track of the CD?', 'When will the low battery symbol appear?','Can you list the names of play modes?']
ground_truths = [['The CLASS 1 LASER PRODUCT label is located on the bottom of the player.'], ['You should best avoid extreme volume.'], ['You can use the positive and negative battery terminals.'], ['You can insert the headphone plug in the jack on the side of the CD payer.'], ['You can slide the switch to the right to open the CD  compartment. Open indicates the compartment door is open.'], ['You can press this button to  advance to the next track when a CD is being played.'], ['When about 10 minutes of playing time remain, the low  battery symbol appears and flashes until the player is turned OFF or the battery runs out of power.'], ['Yes, there are five play modes: Normal, Intro, Shuffle, Repeat and Repeat All mode.']]
answers = []
contexts = []

for question in questions:
    answer = str(chain.invoke(question))
    answers.append(answer)
    context = []
    docs = retriever.get_relevant_documents(question)
    contexts.append(docs)
  
print(contexts)

data = {
    "question": questions,
    "answer": answers,
    "ground_truths": ground_truths,
    "contexts": contexts,
}

# convert dict to dataset
dataset = Dataset.from_dict(data)
print(dataset)

# # eval_result = faithfulness_chain(dataset)

from ragas.metrics import faithfulness
faithfulness.llm = llm
faithfulness.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

result = evaluate(
    dataset = dataset, 
    metrics=[
        # context_precision,
        # context_recall,
        faithfulness,
        # answer_relevancy,
    ],
    # llm = model,
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
)
from ragas import evaluate

result = evaluate(
    dataset = dataset,
  metrics=[faithfulness]
)

df = result.to_pandas()
print(df)
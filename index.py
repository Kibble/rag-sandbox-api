from flask import Flask, request
# from openai import OpenAI
from dotenv import load_dotenv
import os
# import chromadb

# import bs4
from langchain import hub
from langchain_chroma import Chroma
# from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.vectorstores import VectorStoreRetriever
# from dataclasses import Field
from typing import List


load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

# Get a pre-constructed RAG prompt from prompthub
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

llm = ChatOpenAI(model="gpt-4o-mini")

def load_rules(filepath, context, size, overlap, errata=[]):
    rules = PyMuPDFLoader(filepath).load()
    
    if (len(errata) > 0):
        for e in errata:
            rules.append(e)
    for r in rules:
        r.metadata['source'] = context
    vectorstore = Chroma.from_documents(
        documents=RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
            .split_documents(rules),
        embedding=OpenAIEmbeddings())
    return vectorstore

vs = {}

# RAG start
sts_errata = [
    Document(
        page_content='Poison is an effect that removes HP over time. As it is not damage, it does not trigger effects caused by taking damage. Poison is tracked with poison tokens.',
        metadata={'source': 'sts'})
]
sts_vs = load_rules('./sts.pdf', 'sts', 1000, 200, sts_errata)
# RAG end

class FilteredRetriever(VectorStoreRetriever):
    vectorstore: VectorStoreRetriever
    search_type: str = "similarity"
    # search_kwargs: dict = Field(default_factory=dict)
    filter_prefix: str
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        results = self.vectorstore.get_relevant_documents(query=query)
        return [doc for doc in results if doc.metadata['source'].startswith(self.filter_prefix)]

def get_chain(retriever):
    llm = ChatOpenAI(model='gpt-4o-mini')
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# Define the /rag API endpoint
@app.route('/sts', methods=['POST'])
def sts():
    filtered_retriever = FilteredRetriever(
        vectorstore=sts_vs.as_retriever(),
        filter_prefix='sts'
    )
    query = request.form['query']
    llm = ChatOpenAI(model="gpt-4o-mini")
    chain = (
        {"context": filtered_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain.invoke(query)

@app.route('/test', methods=['POST'])
def test():
    size = request.form['chunk_size']
    overlap = request.form['chunk_overlap']
    rules = PyMuPDFLoader('./sts.pdf').load()
    query = request.form['query']
    llm = ChatOpenAI(model='gpt-4o-mini')
    vectorstore = Chroma.from_documents(
        documents=RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
            .split_documents(rules),
        embedding=OpenAIEmbeddings())
    chain = (
        {"context": vectorstore.as_retriever() | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain.invoke(query)


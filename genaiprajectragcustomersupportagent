!pip install langchain
!pip install -qU langchain-openai
!pip install "langchain<0.1.0"
!pip install unstructured
!pip install faiss-cpu
!pip install transformers
#import the library
import torch
from transformers import DPRQuestionEncoder,DPRQuestionEncoderTokenizer,DPRContextEncoder,DPRContextEncoderTokenizer
from transformers import pipeline,AutoModelForSeq2SeqLM,AutoTokenizer
import faiss
import numpy as np
#load the huggingface model
question_encoder=DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
question_tokenizer=DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
context_encoder=DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
context_tokenizer=DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
rag_model=AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base')
rag_tokenizer=AutoTokenizer.from_pretrained('google/flan-t5-base')
import os
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
loader = DirectoryLoader('/content/knowledge_base/', glob="**/*.txt")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = splitter.split_documents(documents)
#with open(docs[0], 'r') as f:
#    content = f.read()
#print(content)
content = docs[0].page_content
doc_embeddings=[]
inputs=context_tokenizer(content,return_tensors='pt',padding=True)
embedding=context_encoder(**inputs).pooler_output.detach().numpy()
doc_embeddings.append(embedding)
print(doc_embeddings)
doc_embeddings = np.array(doc_embeddings)
doc_embeddings.shape[1]
doc_embeddings=np.vstack(doc_embeddings)
#create faiss index for fast retrieval
dimension=doc_embeddings.shape[1]
faiss_index=faiss.IndexFlatL2(dimension)
faiss_index.add(doc_embeddings)
#query processing and retrieval process
def retrieve_top_k(query,k=2):
  query_inputs=question_tokenizer(query,return_tensors='pt')
  query_embeddings=question_encoder(**query_inputs).pooler_output.detach().numpy()
  distances,indices=faiss_index.search(query_embeddings,k)
  retrieved_docs=[content[i] for i in indices[0]]
  return retrieved_docs
#generate the response using RAG
def generate_response(query):
  retrieved_docs=retrieve_top_k(query,k=2)
  context=" ".join(retrieved_docs)
  inputs=rag_tokenizer(f"Question:{query} context:{context}",return_tensors='pt')
  output=rag_model.generate(**inputs)
  response=rag_tokenizer.decode(output[0],skip_special_tokens=True)
  return response
#question answer bot
def chat():
  print('Hi Ask me anything or type stop to end the conversation')
  while True:
    query=input('you: ')
    if query.lower()=='stop':
      print('Goodbye')
      break
    response=generate_response(query)
    print(f'GPT: {response}')
chat()
#docs[0]
#for doc in docs:
#  print(doc.metadata)

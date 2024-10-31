# loading pdf file. 
import openai
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings 
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI 
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser 


my_api_key = os.getenv("OPENAI_API_KEY") 

# Loading the external data and creating a vector space out of that. 

pdf_loader = PyPDFLoader(file_path='../../data/pilot-manual-787.pdf')
documents = pdf_loader.load()
splitter = RecursiveCharacterTextSplitter(
    separators=["\n", ".", "!", "?", ";"],
    chunk_size=1000,
    chunk_overlap=200
)
# the only change is calling split_documents() instead of split_text()
chunks = splitter.split_documents(documents)

embedding_model = OpenAIEmbeddings(
    api_key=my_api_key,
    model="text-embedding-3-small"
)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model
)

# Designing a retriever to retrieve the most similar chunks.
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs = {"k":2}
)

# Designing a prompt template. 
prompt = ChatPromptTemplate.from_template("""
Use the following pieces of context to answer the question at the end.
If you don't know the answer, say that you don't know.
Context:{context}
Question:{question}
""")

# Instantiating the LLM model 

# Instantiate the ChatOpenAI model
llm = ChatOpenAI(model="gpt-4", openai_api_key=my_api_key)

# Define messages
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is cost index?"}
]

# Get the response
response = llm.invoke(messages)
print(response.content)


# creating a LCEL - langchain expression language based retrieval chain. 
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt 
    | llm
    | StrOutputParser()
)

result = chain.invoke("how to do performance initialization?")
print(result)



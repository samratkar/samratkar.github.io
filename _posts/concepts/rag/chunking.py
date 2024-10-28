# from langchain_text_splitters import CharacterTextSplitter

# # splitting text into chunks
# text_splitter = CharacterTextSplitter(
#     separator="\n",
#     chunk_size=100,
#     chunk_overlap=10
# )
# text = '''This is a longer paragraph. It contains multiple sentences to demonstrate how the text splitter works. 
# The text splitter will break this paragraph into smaller chunks based on the specified chunk size and overlap. 
# This is useful for processing large texts in manageable pieces. Let's see how it performs with this example.'''
# chunks = text_splitter.split_text(text)
# print (chunks)
# print (len(chunk) for chunk in chunks)

# from langchain_text_splitters import RecursiveCharacterTextSplitter
# splitter = RecursiveCharacterTextSplitter(
#     separators=["\n", ".", "!", "?", ";"],
#     chunk_size=100,
#     chunk_overlap=10
# )
# # chunks = splitter.split_text(text)
# # print (chunks)
# # print (len(chunk) for chunk in chunks)

# # splitting documents into chunks. 

# # loading pdf file. 
# from langchain_community.document_loaders import PyPDFLoader
# pdf_loader = PyPDFLoader(file_path='data/pilot-manual-787.pdf')
# documents = pdf_loader.load()
# splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# # the only change is calling split_documents() instead of split_text()
# chunks = splitter.split_documents(documents)
# print (chunks)
# print (len(chunk) for chunk in chunks)
# print (f"Total number of chunks: {len(chunks)}")

# loading and chunking python files 
# loading python files 
# from langchain_community.document_loaders import PythonLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
# loader=PythonLoader('embedding-pdf.py')
# python_data = loader.load()
# print(python_data[0])
# # chunking without context
# # python_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=10)
# # chunking with context 
# python_splitter = RecursiveCharacterTextSplitter.from_language(
#         language = Language.PYTHON, chunk_size=150, chunk_overlap=10)
# chunks = python_splitter.split_documents(python_data)
# for i, chunk in enumerate (chunks[:3]):
#     print(f"Chunk {i+1}:\n{chunk.page_content}\n")

# Advanced splitting methods 
# 1. the previous splits are not context based. ignore context of surrounding texts. 
# 2. splits are made using characters rather than # tokens.  If we split documents using characters rather than tokens, we risk retrieving chunks and creating a retrieval prompt that # exceeds the maximum amount of text the model can process at once, also called the model context window. All language models break texts into tokens. These are smaller unit of texts, for processing. 







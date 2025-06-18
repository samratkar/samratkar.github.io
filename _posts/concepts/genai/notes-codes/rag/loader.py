# # loading a csv file. 
# from langchain_community.document_loaders.csv_loader import CSVLoader
# csv_loader = CSVLoader(file_path='data/master_flight_data-1.csv')
# documents = csv_loader.load()
# print (documents)

# # loading pdf file. 
# from langchain_community.document_loaders import PyPDFLoader
# pdf_loader = PyPDFLoader(file_path='data/PMDG_737_MSFS_Tutorial.pdf')
# documents = pdf_loader.load()
# print (documents)

# # loading html files
# from langchain_community.document_loaders import UnstructuredHTMLLoader
# html_loader = UnstructuredHTMLLoader(file_path='')
# documents = html_loader.load()
# print (documents)

# # loading markdown files. 
# from langchain_community.document_loaders import UnstructuredMarkdownLoader
# loader = UnstructuredMarkdownLoader(file_path='data/teen-study.md')
# markdown_content = loader.load()
# print(markdown_content[0])


# loading python files 
from langchain_community.document_loaders import PythonLoader
loader=PythonLoader('embedding-pdf.py')
python_data = loader.load()
print(python_data[0])

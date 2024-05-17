from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


embedding = HuggingFaceEmbeddings()
def load_data(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    return pages

print("loading data")
data = load_data("deeplearningwithpython.pdf")


print("loading embeddings in vector db")
db=FAISS.from_documents(data,embedding)
print("saving index")
db.save_local('faiss')
print("process completed")


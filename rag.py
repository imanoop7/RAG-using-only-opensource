from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings



embedding=HuggingFaceEmbeddings()
def get_response(question,context):
    prompt="""
    You are good assistant, will provide summary on {context} based on {question}
    """
    model = Ollama(model='phi3')
    response= model.invoke(prompt)
    return response

question = input()
new_db = FAISS.load_local("faiss", embedding,allow_dangerous_deserialization=True)
res = new_db.similarity_search(question)

output = get_response(question, res[0])

print(output)



    


    
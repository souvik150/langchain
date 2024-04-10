from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import langchain
from redundant_filter_retriever import RedundantFilterRetriever

# langchain.debug = True

load_dotenv()
 
chat = ChatOpenAI()
embeddings = OpenAIEmbeddings()

db = Chroma(
  persist_directory="emb",
  embedding_function=embeddings
)

# retriever = db.as_retriever()

# chain = RetrievalQA.from_chain_type(
#   llm=chat,
#   retriever=retriever,
#   chain_type="stuff"
# )

# print("----------------------------------------------------")
# result = chain.run("Tell me about a computer bug?")

# print(result)

retriever = RedundantFilterRetriever(
    embeddings=embeddings,
    chroma=db
)

chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=retriever,
    chain_type="stuff"
)

print("----------------------------------------------------")
result = chain.run("What is an interesting fact about the English language?")

print(result)

from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

document = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

embeddings = HuggingFaceEmbeddings(model='sentence-transformers/all-MiniLM-L6-v2')

query = " Tell me about Virat Kohli"

doc_emb = embeddings.embed_documents(document)
query_emb = embeddings.embed_query(query)

"""Values passed inside cosine similarity function should be 2D vectors"""
scores = cosine_similarity([query_emb],doc_emb)[0]

print(list(enumerate(scores)))

index, score = sorted(list(enumerate(scores)),key= lambda x:x[1])[-1]

print(query)
print(document[index])
print("Similarity score is = ", score)

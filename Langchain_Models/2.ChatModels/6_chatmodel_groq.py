from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-120b",temperature=0,max_tokens=300)

result = model.invoke("What is the capital of India")

print(result.content)
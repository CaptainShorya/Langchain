from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model =ChatGroq(model="openai/gpt-oss-120b")

chat_history = [
    SystemMessage(content="You are a helpful assitant")
]

while True:
    user_input = input("You : ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input == "exit":
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI : ", result.content)

print(chat_history)
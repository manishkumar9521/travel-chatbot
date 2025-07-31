import os
from dotenv import load_dotenv

load_dotenv()

if os.getenv("OPEN_ROUTER"):
    os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
    os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_ROUTER")
else:
    print("Open-router API key not found!")
    exit()

from langchain.chat_models import init_chat_model

model = init_chat_model(
    temperature=0.7,
    model="deepseek/deepseek-chat-v3-0324:free",
    model_provider="openai"
)

from langchain.schema import HumanMessage
response = model.invoke([HumanMessage(content="Hi, how are you?")])

print(response.content)
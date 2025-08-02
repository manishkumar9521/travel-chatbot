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

# model="llama3.2", model_provider="ollama"
model = init_chat_model(
    temperature=0.7,
    model="deepseek/deepseek-chat-v3-0324:free",
    model_provider="openai",
    max_tokens=50
)

from langchain.schema import HumanMessage, AIMessage, SystemMessage
system_prompt = SystemMessage(
    content="You are a helpful, friendly travel assistant and who in a very short, concise and accurate manner. " \
    "You also creates travel packages for the users. If needed you uses tools to answer questions and "
    "if you don't know the answer just say it.")

from html_design import loading_html
# Define chatbot function
from openai import RateLimitError
def chat_with_bot(query, chat_history):
    messages = [system_prompt]

    # Add history (if any)
    for msg in chat_history:
        messages.append((HumanMessage if msg["role"] == "user" else AIMessage)(content=msg["content"]))

    # # Add current input
    messages.append(HumanMessage(content=query))
    
    # # Get model response
    try:
        is_streaming = True
        chunks = []
        for chunk in model.stream(messages):
            chunks.append(chunk.content or "")  # Avoid NoneType issues
            meta = getattr(chunk, "response_metadata", {})
            if (meta.get("finish_reason") == "stop" or meta.get("done") is True) and is_streaming:
                is_streaming = False
            yield "".join(chunks) + (loading_html if is_streaming else "")

    except RateLimitError as e:
        # Friendly message for user
        yield e.message
    
    except Exception as e:
        # General fallback in case of other errors
        yield f"⚠️ An unexpected error occurred: {str(e)}"

import gradio as gr
demo = gr.ChatInterface(
    fn=chat_with_bot,
    type="messages",
    textbox=gr.Textbox(
        placeholder="Ask me any travel-related question.",
        container=False,
        scale=7
    ),
    title="🧳 Travel Assistant Chatbot (LangChain + OpenRouter + Gradio)",
    description="Ask any question about the place where you want to visit.",
    theme="ocean",
    examples=["Hello", "Plan a trip", "Suggest me a picnic location near me"],
)
demo.launch()

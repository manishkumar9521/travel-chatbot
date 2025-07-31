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

from langchain.schema import HumanMessage, SystemMessage
system_prompt = SystemMessage(content="You are a helpful, friendly travel assistant.")

# Define chatbot function
def chat_with_bot(user_input, chat_history):
    messages = [system_prompt]
    
    # Add history (if any)
    for human, ai in chat_history:
        messages.append(HumanMessage(content=human))
        messages.append(HumanMessage(content=ai))  # Gradio doesn't track AIMessage; reuse HumanMessage
    
    # Add current input
    messages.append(HumanMessage(content=user_input))
    
    # Get model response
    response = model.invoke(messages)
    
    chat_history.append((user_input, response.content))
    return chat_history, chat_history

import gradio as gr
with gr.Blocks() as demo:
    gr.Markdown("## 🧳 Travel Assistant Chatbot (LangChain + OpenRouter + Gradio)")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Type your message here...")
    clear = gr.Button("Clear")

    state = gr.State([])

    msg.submit(
        chat_with_bot,
        [msg, state],
        [chatbot, state]
    )

    clear.click(
        lambda: ([], []),
        None,
        [chatbot, state]
    )

demo.launch()

print(response.content)
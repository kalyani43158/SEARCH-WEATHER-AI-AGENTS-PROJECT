import streamlit as st

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


def Chat_node(state: ChatState):
    response = llm.invoke(state["messages"])
    return {
        "messages": [response]
    }


checkpointer = MemorySaver()

graph = StateGraph(ChatState)

graph.add_node("Chat Node", Chat_node)

graph.add_edge(START, "Chat Node")
graph.add_edge("Chat Node", END)

Chatbot = graph.compile(
    checkpointer=checkpointer
)


st.set_page_config(
    page_title="LangGraph Chatbot",
    page_icon="🤖"
)

st.title("🤖 LangGraph Chatbot")


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])


user_message = st.chat_input("Type your message...")


if user_message:

    st.session_state.messages.append({
        "role": "user",
        "content": user_message
    })

    with st.chat_message("user"):
        st.write(user_message)


    config = {
        "configurable": {
            "thread_id": "1"
        }
    }


    response = Chatbot.invoke(
        {
            "messages": [
                HumanMessage(content=user_message)
            ]
        },
        config=config
    )


    ai_response = response["messages"][-1].content


    st.session_state.messages.append({
        "role": "assistant",
        "content": ai_response
    })


    with st.chat_message("assistant"):
        st.write(ai_response)
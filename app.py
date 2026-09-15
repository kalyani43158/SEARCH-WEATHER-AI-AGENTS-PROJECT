import os
import requests
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain.tools import tool
from langchain.agents import create_agent


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Smart AI Agent",
    page_icon="🤖",
    layout="wide"
)


# =====================================================
# LOAD ENVIRONMENT VARIABLES
# =====================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")


# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #0f172a,
        #111827,
        #1e293b
    );
}

/* Main title */
.title {
    text-align: center;
    font-size: 48px;
    font-weight: bold;
    margin-top: 10px;
    background: linear-gradient(
        90deg,
        #60a5fa,
        #a78bfa,
        #f472b6
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 18px;
    margin-bottom: 30px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0f172a;
}

/* Cards */
.card {
    background: rgba(30,41,59,0.8);
    border: 1px solid #334155;
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 15px;
}

/* Status */
.status {
    background: #052e16;
    color: #86efac;
    padding: 10px;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 20px;
}

/* Buttons */
.stButton > button {
    width: 100%;
    border-radius: 10px;
    background: #1e293b;
    color: white;
    border: 1px solid #475569;
}

.stButton > button:hover {
    border-color: #60a5fa;
    color: #60a5fa;
}

/* Chat input */
[data-testid="stChatInput"] {
    border-radius: 15px;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    border-radius: 15px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# HEADER
# =====================================================

st.markdown(
    '<div class="title">🤖 Smart AI Agent</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Intelligent Assistant powered by Groq + Tavily + WeatherStack'
    '</div>',
    unsafe_allow_html=True
)


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown("## 🤖 AI Agent")

    st.markdown(
        '<div class="status">🟢 Agent Online</div>',
        unsafe_allow_html=True
    )

    st.markdown("### 🛠️ Tools")

    st.markdown("""
    <div class="card">
        <h4>🔎 Web Search</h4>
        <p>Search latest information from the web.</p>
    </div>

    <div class="card">
        <h4>🌤️ Weather</h4>
        <p>Get current weather information.</p>
    </div>

    <div class="card">
        <h4>🧠 Groq AI</h4>
        <p>Fast intelligent AI responses.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")

    st.caption("Built using Streamlit & LangChain")


# =====================================================
# WEATHER TOOL
# =====================================================

@tool
def get_weather(city: str) -> str:
    """Fetch current weather information for a city."""

    api_key = os.getenv("WEATHERSTACK_API_KEY")

    if not api_key:
        return "WEATHERSTACK_API_KEY is not configured."

    try:

        url = (
            "http://api.weatherstack.com/current"
            f"?access_key={api_key}"
            f"&query={city}"
        )

        response = requests.get(
            url,
            timeout=10
        )

        data = response.json()

        if "current" not in data:
            return f"Could not fetch weather for {city}"

        current = data["current"]

        return (
            f"City: {city}\n"
            f"Temperature: {current['temperature']}°C\n"
            f"Weather: {current['weather_descriptions'][0]}\n"
            f"Humidity: {current['humidity']}%\n"
            f"Wind Speed: {current['wind_speed']} km/h"
        )

    except Exception as e:

        return f"Weather error: {str(e)}"


# =====================================================
# TAVILY SEARCH
# =====================================================

search_tool = TavilySearch(
    max_results=3
)


# =====================================================
# TOOLS
# =====================================================

tools = [
    search_tool,
    get_weather
]


# =====================================================
# CREATE AI AGENT
# =====================================================

@st.cache_resource
def create_ai_agent():

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=GROQ_API_KEY
    )

    agent = create_agent(
        model=llm,
        tools=tools
    )

    return agent


# =====================================================
# SESSION STATE
# =====================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =====================================================
# WELCOME SCREEN
# =====================================================

if len(st.session_state.messages) == 0:

    st.markdown("""
    <div class="card">

    <h2>👋 Welcome!</h2>

    <p>
    I'm your <b>Smart AI Agent</b>.
    Ask me anything and I can decide whether to use
    web search or weather information.
    </p>

    <br>

    <b>Try these examples:</b>

    <ul>
        <li>🔎 What is the latest news about AI?</li>
        <li>🌤️ What is the current weather in Ongole?</li>
        <li>🧠 Explain artificial intelligence.</li>
        <li>🔎 Search for machine learning tutorials.</li>
    </ul>

    </div>
    """, unsafe_allow_html=True)


# =====================================================
# DISPLAY CHAT HISTORY
# =====================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# =====================================================
# CHAT INPUT
# =====================================================

prompt = st.chat_input(
    "💬 Ask your AI Agent anything..."
)


# =====================================================
# PROCESS QUESTION
# =====================================================

if prompt:

    # User message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):

        st.markdown(prompt)


    # AI response
    with st.chat_message("assistant"):

        with st.spinner("🤔 AI Agent is thinking..."):

            try:

                agent = create_ai_agent()

                response = agent.invoke({
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                })

                answer = response["messages"][-1].content

                st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:

                error = f"""
❌ **Something went wrong**

`{str(e)}`
"""

                st.error(error)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error
                })

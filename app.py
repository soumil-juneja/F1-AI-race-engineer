import streamlit as st
import os
from dotenv import load_dotenv

# 1. Load environment variables (API Keys)
load_dotenv()

# 2. LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 3. Import your fully decoupled logic!
from tools import F1_TOOLS

# --- UI SETUP ---
st.set_page_config(page_title="F1 Pit Wall AI", page_icon="🏎️", layout="wide")

neumorphic_css = """
<style>
    /* Global Font and Background */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Roboto', system-ui, -apple-system, sans-serif !important;
        background-color: #1a1a1a !important;
        color: #FFD700 !important;
    }
    
    .stApp {
        background-color: #1a1a1a !important;
    }

    /* Headers */
    h1, h2, h3 {
        color: #E10600 !important;
        border-bottom: none;
        padding-bottom: 10px;
        font-weight: 800 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }

    /* Chat Input Box */
    div[data-testid="stChatInput"] {
        background-color: #1a1a1a !important;
        border: none !important;
        border-radius: 20px !important;
        box-shadow: inset 5px 5px 10px #101010, inset -5px -5px 10px #242424 !important;
        padding: 5px;
    }
    div[data-testid="stChatInput"] textarea {
        color: #FFD700 !important;
    }

    /* Chat Messages */
    div[data-testid="stChatMessage"] {
        background-color: #1a1a1a !important;
        border: none !important;
        border-radius: 20px !important;
        padding: 20px !important;
        margin-bottom: 20px !important;
        box-shadow: 8px 8px 16px #101010, -8px -8px 16px #242424 !important;
    }

    /* Avatars */
    div[data-testid="stChatMessageAvatarUser"], div[data-testid="stChatMessageAvatarAssistant"] {
        border-radius: 50% !important;
        box-shadow: 3px 3px 6px #101010, -3px -3px 6px #242424 !important;
    }
    
    /* Code blocks / data outputs */
    pre, code {
        background-color: #151515 !important;
        color: #D4AF37 !important;
        border: none !important;
        border-radius: 10px !important;
        box-shadow: inset 3px 3px 6px #0b0b0b, inset -3px -3px 6px #1f1f1f !important;
    }
</style>
"""
st.markdown(neumorphic_css, unsafe_allow_html=True)

st.title("🏎️ F1 Pit Wall Dashboard")
st.markdown("**STATUS: ACTIVE** | Query race results, driver stints, or 2026 regulations.")

# --- AGENT SETUP ---
# Initialize the Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0)

# Create the System Prompt (Giving the AI its persona and context)
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert F1 Race Engineer and Strategist. 
                  IMPORTANT: You HAVE access to a live data feed via the 'get_live_summary' tool. 
                  When asked "Who is leading?" or "What's happening?", ALWAYS call 'get_live_summary' first.
                  
                  IRONCLAD GUARDRAILS:
                  0. If 'get_live_summary' returns data, use it to answer live questions.
                  1. THE 'I DON'T KNOW' RULE: If a tool returns a CRITICAL SYSTEM ERROR, you MUST say "I don't have that data yet." However, if a tool says a driver recorded 0 laps or did not participate, you should answer the user confidently with "No, they did not participate."
                  2. NO SUBSTITUTIONS: If the user asks for 2026 data and it is unavailable, DO NOT provide 2024 or 2025 data instead. Just say you don't have it.
                  3. STRICT GROUNDING: You must ONLY use the exact numbers returned by your tools. Do NOT invent, guess, or simulate pit stops, lap times, or winners.
                  4. DEFAULT YEAR: If the user does not specify a year, assume 2026. If 2026 fails, tell them to specify a historical year."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Build the Agent using your imported F1_TOOLS registry
agent = create_tool_calling_agent(llm, F1_TOOLS, prompt)
agent_executor = AgentExecutor(agent=agent, tools=F1_TOOLS, verbose=True)

# --- CHAT STATE MANAGEMENT ---
# Initialize chat memory if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing messages on screen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- USER INPUT & AI RESPONSE ---
if user_input := st.chat_input("E.g., How many pit stops did Hamilton make in the 2024 British GP?"):
    
    # 1. Add user message to state and display it
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Generate and display AI response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing telemetry and data..."):
            try:
                # Pass the input and the chat history to the agent
                response = agent_executor.invoke({
                    "input": user_input,
                    "chat_history": st.session_state.messages[:-1]
                })
                output = response["output"]

                # If LangChain returns a list of blocks, extract just the text
                if isinstance(output, list):
                    output = "".join([chunk.get('text', '') for chunk in output])
                elif not isinstance(output, str):
                    output = str(output)

                st.markdown(output)
                
                # Save AI response to state
                st.session_state.messages.append({"role": "assistant", "content": output})
            except Exception as e:
                st.error(f"An error occurred: {e}")

# python -m streamlit run app.py
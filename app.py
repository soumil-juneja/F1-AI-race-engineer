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
st.title("🏎️ F1 AI Race Engineer")
st.markdown("Ask me about race results, driver stints, or 2026 regulations.")

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
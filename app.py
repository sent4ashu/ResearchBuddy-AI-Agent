"""Streamlit UI for ResearchBuddy.
Run with: streamlit run app.py 
"""
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from agent.agent_core import build_agent
from agent.memory import WindowedMemory


#----------------------------- Page config and UI setup -----------------------------
st.set_page_config(page_title="ResearchBuddy AI Agent", page_icon="🤖", layout="centered")

#----------------------------- State intitialization -----------------------------
if "executor" not in st.session_state:
    st.session_state.executor = build_agent()  # create the agent executor once and reuse

if "memory" not in st.session_state:
    st.session_state.memory = WindowedMemory(max_turns=5)  # create the memory manager once and reuse

if "ui_messages" not in st.session_state:
    # seperate from agent memory, we will keep it growing for UI Display purposes. 
    # Agent memory is Windowed and in rolling manner keeps last N turns only. 
    st.session_state.ui_messages = []  # list of (is_user: bool, content: str) for rendering the chat history

#----------------------------- Sidebar: info + controls -----------------------------

with st.sidebar:
    st.title("ResearchBuddy AI Agent")
    st.markdown("""A ReAct agent with web search, wikipedia and a calculator.""")
    st.markdown("**How it works**")
    st.markdown(
        "- Ask Anything; ResearchBuddy will decide when to use tools (web search, wikipedia, calculator) \n"
        "- Expand 'Resourcing trace' under each option to see what it did \n"
        "- The agent remembers the last 5 turns of conversation."
    )
    st.divider()
    if st.button("Clear Conversation"):
        st.session_state.memory.clear()
        st.session_state.ui_messages = []
        st.rerun()  # refresh the page to clear the chat history display
    st.caption(f"Memory : {len(st.session_state.memory)} messages held.")
    
    # ---------------------------Main Chat Interface-----------------------------
st.title("Ask ResearchBuddy")

# Replay existing conversation on each rerun
for msg in st.session_state.ui_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # If this was an assistent turn with a tool trace, show it collapsed.
        if msg["role"] == "assistant" and msg.get("trace"):
            with st.expander("🔧 Reasoning trace"): 
                for i, step in enumerate(msg["trace"], 1):
                    tool_name = step["tool"]
                    tool_input = step["tool_input"]
                    tool_output = step["tool_output"]
                    st.markdown(f"**Step {i}: '{tool_name}'**")
                    st.code(f"Input:{tool_input}\nOutput:{tool_output}", language="text")

# ------------------------------ Handle new user input -----------------------------
if user_input := st.chat_input("Type your question here..."):
    # Display user's message immediately
    st.session_state.ui_messages.append({"role": "user", "content": user_input, "trace": None})
    with st.chat_message("user"):
        st.markdown(user_input)


    # Call the agent show a spinner while waiting for the response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.executor.invoke({"input": user_input, "chat_history": st.session_state.memory.get_messages()})

        answer = result["output"]  # the agent's final answer after reasoning and tool use
        st.markdown(answer)

        # Extract the intermediate steps into a UI-friendly shape.
        # intermediate steps is a list of (AgentAction, Output_str) tuples.
        trace = []
        for action, output in result.get("intermediate_steps", []):
            trace.append({
                "tool": action.tool,
                "tool_input": action.tool_input,
                "tool_output": str(output)
            })
        
        if trace:
            with st.expander("🔧 Reasoning trace"):
                for i, step in enumerate(trace, 1):
                    st.markdown(f"**Step {i}: '{step['tool']}'**")
                    st.code(f"Input:{step['tool_input']}\nOutput:{step['tool_output']}", language="text")

    # Persist this turn - both into UI history and agent memory
    st.session_state.ui_messages.append({"role": "assistant", "content": answer, "trace": trace})
    st.session_state.memory.add_turn(HumanMessage(content=user_input), AIMessage(content=answer))
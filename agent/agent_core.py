"""
Baseline chatboat - plain LLM with convesation memory
"""
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from agent.llm import get_llm

SYSTEM_PROMPT="""You are a ResearchBuddy, a helpfulresearch assistent.
your job is to answer the user's question clearly and concisely.

Important rules:
- If you do not know something or are not sure, say so plainly. Do not invent facts.
- If a question requires current information (news, prices, today's weather, recent events),
  acknowledge that you cannot look things up yet and explain what you would need to answer it.
- Keep answers focused. No filler like 'Great question!'.
"""

def run_chat():
    llm = get_llm()
    
    # Conversation history starts with system prompt
    # Every turn, we append users message and AI reply to this history
    history = [SystemMessage(content=SYSTEM_PROMPT)]

    print("=" * 60)
    print("ResearchBuddy (baseline - no tool yet)")
    print("Type 'quit' or 'exit' to stop. Type 'reset' to clear the history.")
    print("=" * 60)

    while True:
        user_input = input("\nYou: ").strip()

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break
        if user_input.lower() == "reset":
            history = [SystemMessage(content=SYSTEM_PROMPT)]
            print("Conversation history reset.")
            continue
    
    # Append user's message to history
        history.append(HumanMessage(content=user_input))

    # Send full history to LLM 
        response = llm.invoke(history)

    # Append AI's response to history
        history.append(AIMessage(content=response.content))

        print(f"\nResearchBuddy: {response.content}")

if __name__ == "__main__":
    run_chat()
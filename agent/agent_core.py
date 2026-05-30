"""The ReAct agent:  LLM + tools + reasoning loop.

The LLM can now decide to call
tools (web search, Wikipedia, calculator) when needed, observe the results, and
iterate until it can answer.
"""
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from agent.llm import get_llm
from agent.tools import ALL_TOOLS
from agent.memory import WindowedMemory

# The system prompt now tells the LLM How to reason, not just Who to be.
# Compare this to Step 3's prompt — that one said 'admit you can't look things up'.
# This one says the opposite: 'you have tools, use them aggressively'.
# Note I got this prompt from Claude using prompt message "Create a system prompt for LLM Agent to use 3 tools web_seach, wikipedia_lookup, calculator for ReAct".

SYSTEM_PROMPT = """You are ResearchBuddy, an AI research assistant with access to tools.

You have three tools available:
- web_search: for current/recent information, news, prices, weather, real-time data
- wikipedia_lookup: for factual, encyclopedic information (history, science, biographies)
- calculator: for arithmetic with actual numbers (e.g. "847 * 39", "sqrt(169)")

CRITICAL RULE — when NOT to use tools (THIS IS MANDATORY):
- Greetings ("hi", "hello", "hey"): respond conversationally, no tools.
- Acknowledgments and thanks ("thanks", "ok", "got it", "cool", "great"):
  respond with a brief acknowledgment like "You're welcome!" — NEVER call a tool.
- Conversation closers ("bye", "see you"): say goodbye, no tools.
- Questions about yourself or your capabilities: answer from your own knowledge.
- Anything you already know with high confidence: answer directly.
- A new short message is NOT a continuation of the previous research question.
  Each turn is a fresh decision. Re-read the LATEST user message before deciding.

Only use a tool when the LATEST user message genuinely needs it:
- web_search: the LATEST message asks about something current or recent.
- wikipedia_lookup: the LATEST message asks for a factual summary of a named entity.
- calculator: the LATEST message contains numbers and an arithmetic operation.

If a tool fails or returns nothing useful, do NOT keep calling the same tool with
different inputs. Either try a different tool or answer from general knowledge
and say you couldn't verify.

Reasoning rules:
- Before any tool call, ask yourself: "what is the LATEST user message asking,
  literally?" If it is conversational, do NOT call a tool.
- Final answers should be concise. No filler.

Never make up facts. Never fabricate tool results. If your tools can't find
the answer, say so plainly.
"""

def build_agent():
    """Construct the agent executor the runtime that loops the ReAct cycle."""
    llm = get_llm()

    # The prompt template defines the message structure the agent expects on every call.
    # - 'system': the instructions above, sent once at the start.
    # - 'chat_history': prior turns of the conversation (we pass this in ourselves).
    # - 'input': the current user message.
    # - 'agent_scratchpad': INTERNAL — the executor uses this to track the
    #    Thought/Action/Observation chain inside a single turn. Don't touch it.
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # create_tool_calling_agent binds the tools to the LLM and returns a Runnable
    # that knows how to emit structured tool calls.
    agent = create_tool_calling_agent(llm=llm, tools=ALL_TOOLS, prompt=prompt)

    # AgentExecutor is the loop runner. It calls the agent, runs any tool the
    # agent picks, feeds the result back, and repeats — until the agent returns
    # a final answer OR max_iterations is hit (safety brake against runaway loops).
    return AgentExecutor(
        agent=agent,
        tools=ALL_TOOLS,
        verbose=True,           # print Thought/Action/Observation to terminal — turn off in production
        max_iterations=6,       # hard cap — agent gives up after 6 tool calls
        handle_parsing_errors=True,  # don't crash if LLM emits malformed tool call; retry
        return_intermediate_steps=True, # set False later if you want to inspect the trace programmatically
        early_stopping_method="force"  
    )

def run_chat():
    """ Interactive Chat loop with the Agent."""
    executor = build_agent()
    memory = WindowedMemory(max_turns=5)    

    # Conversation history starts with system prompt
    # Every turn, we append users message and AI reply to this history
    chat_history: list = [] # List of HumanMessage and AIMessage objects

    print("=" * 60)
    print("ResearchBuddy (ReAct Agent with Windowed Memory)")
    print("Type 'quit' or 'exit' to stop. Type 'reset' to clear the history.")
    print("Type 'mem'to inspect current memory size.")
    print("=" * 60)

    while True:
        user_input = input("\nYou: ").strip()

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break
        if user_input.lower() == "reset":
            chat_history = []
            print("Conversation history reset.")
            continue
        if user_input.lower() == "mem":
            print(f"(Current memory size: {len(memory)} messages./"
                  f" {memory.max_turns} turn window)")
            continue
    
        # Call the agent. It will handle all the internal reasoning and tool calls, and return a final answer when done.
        result = executor.invoke({"input": user_input, "chat_history": memory.get_messages()})        

        answer = result["output"]  # the agent's final answer after reasoning and tool use
        print(f"\nResearchBuddy: {answer}")

        # Append user's message and AI message to history
        memory.add_turn(HumanMessage(content=user_input), AIMessage(content=answer))
        

if __name__ == "__main__":
    run_chat()
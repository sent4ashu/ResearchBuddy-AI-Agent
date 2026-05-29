"""Conversational memory management for agents- windowed buffer

Keeps only last N turns of conversation to bound context size.
A 'turn' is one human message + one agent message (which may include tool calls and observations).
"""
from langchain_core.messages import BaseMessage
class WindowedMemory:
    """Hold last N turn of conversations between human and agent."""

    def __init__(self, max_turns: int = 5):
        """
        Args:
            max_turns: how many turns of conversation to keep in memory. A turn is one human message + one agent message.
        """
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")    
        self.max_turns = max_turns
        self._messages: list[BaseMessage] = []
    
    def add_turn(self, human_msg: BaseMessage, ai_msg: BaseMessage) -> None:
        """Append a complete turn and trim to window size."""
        self._messages.append(human_msg)
        self._messages.append(ai_msg)
        self._trim()

    def _trim(self) -> None:
        """Drop oldest message except last N turns (2N messages)."""
        max_messages = self.max_turns * 2
        if len(self._messages) > max_messages:
            self._messages = self._messages[-max_messages:]
    
    def get_messages(self) -> list[BaseMessage]:
        """Get the current list of messages in memory."""
        return list(self._messages)  # return a copy to prevent external mutation
    
    def clear(self) -> None:
        """Clear all messages from memory."""
        self._messages = [] 

    def __len__(self) -> int:
        """Number of messages currently in memory."""
        return len(self._messages)
    
if __name__ == "__main__":
      from langchain_core.messages import HumanMessage, AIMessage
      m = WindowedMemory(max_turns=2)
      for i in range(5):
          m.add_turn(HumanMessage(content=f"Q{i}"), AIMessage(content=f"A{i}"))
      print(f"After 5 turns with window=2: {len(m)} messages held")
      for msg in m.get_messages():
          print(f"  {type(msg).__name__}: {msg.content}")
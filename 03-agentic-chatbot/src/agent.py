"""
Core agent loop: retrieve relevant memory, let the model decide whether
a tool call is needed (via Groq's OpenAI-compatible function calling),
execute the tool if requested, then produce a final answer. Writes a
compressed summary of the turn back to memory.

Uses Groq's free-tier API (no billing required) -- the Groq Python SDK
mirrors OpenAI's client interface, so this is the same tool-calling
contract as OpenAI's API (structured tool_calls, not prompt parsing).
"""
import json

from groq import Groq

from memory import MemoryStore
from tools import TOOL_REGISTRY

client = Groq()  # reads GROQ_API_KEY from the environment


def _tool_schemas() -> list[dict]:
    schemas = []
    for name, spec in TOOL_REGISTRY.items():
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": spec["description"],
                    "parameters": {
                        "type": "object",
                        "properties": {"input": {"type": "string"}},
                        "required": ["input"],
                    },
                },
            }
        )
    return schemas


class Agent:
    def __init__(self, cfg: dict, memory_store: MemoryStore):
        self.cfg = cfg
        self.memory = memory_store

    def respond(self, session_id: str, message: str, history: list[dict]) -> str:
        m = self.cfg["memory"]
        recalled = self.memory.retrieve(message, session_id, m["top_k_retrieve"], m["similarity_threshold"])
        memory_context = "\n".join(f"- {fact}" for fact in recalled)

        system_prompt = self.cfg["agent"]["system_prompt"]
        if memory_context:
            system_prompt += f"\n\nRelevant facts you remember about this user:\n{memory_context}"

        messages = [{"role": "system", "content": system_prompt}] + history + [
            {"role": "user", "content": message}
        ]

        response = client.chat.completions.create(
            model=self.cfg["model"],
            messages=messages,
            tools=_tool_schemas(),
            tool_choice="auto",
        )
        choice = response.choices[0].message

        if choice.tool_calls:
            messages.append(choice)
            max_calls = self.cfg["agent"]["max_tool_calls_per_turn"]
            # Every tool_call_id here MUST get a matching "role": "tool"
            # message below, or the follow-up call fails -- so calls past
            # our cap still get a placeholder response, never silently dropped.
            for i, call in enumerate(choice.tool_calls):
                tool_name = call.function.name
                if i >= max_calls:
                    result = "Skipped -- tool call limit reached for this turn."
                else:
                    args = json.loads(call.function.arguments)
                    result = TOOL_REGISTRY[tool_name]["fn"](args["input"])
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": result}
                )
            follow_up = client.chat.completions.create(model=self.cfg["model"], messages=messages)
            final_answer = follow_up.choices[0].message.content
        else:
            final_answer = choice.content

        _FILLER = {"thanks", "thank you", "ok", "okay", "cool", "got it", "sounds good"}
        if message.strip().lower() not in _FILLER and len(message.strip()) > 3:
            self.memory.add(f"User asked: {message} | Agent answered: {final_answer[:200]}", session_id)

        return final_answer
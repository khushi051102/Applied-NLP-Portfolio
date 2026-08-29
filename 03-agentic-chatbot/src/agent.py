"""
Core agent loop: retrieve relevant memory, let the model decide whether
a tool call is needed (via OpenAI function calling), execute the tool
if requested, then produce a final answer. Writes a compressed summary
of the turn back to memory.
"""
import json

from openai import OpenAI

from memory import MemoryStore
from tools import TOOL_REGISTRY

client = OpenAI()


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
        # Retrieve relevant long-term memory for this session and inject it
        # as extra context -- this is what lets the agent "remember" things
        # from previous sessions that aren't in the current `history`.
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

        # Tool-call routing: the model itself decides if a tool is needed
        # (returns tool_calls) vs. answering directly -- no hardcoded if/else
        # on keywords, which is the difference between a real agent and a
        # rules engine wearing an LLM costume.
        if choice.tool_calls:
            messages.append(choice)
            for call in choice.tool_calls[: self.cfg["agent"]["max_tool_calls_per_turn"]]:
                tool_name = call.function.name
                args = json.loads(call.function.arguments)
                result = TOOL_REGISTRY[tool_name]["fn"](args["input"])
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": result}
                )
            follow_up = client.chat.completions.create(model=self.cfg["model"], messages=messages)
            final_answer = follow_up.choices[0].message.content
        else:
            final_answer = choice.content

        # Write a compact summary back to memory rather than the raw turn --
        # keeps the memory store from filling up with filler ("thanks", "ok").
        self.memory.add(f"User asked: {message} | Agent answered: {final_answer[:200]}", session_id)

        return final_answer

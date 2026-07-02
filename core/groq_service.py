import os
import json
from groq import Groq
from types import SimpleNamespace

class GroqService:
    def __init__(self, model: str):
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError("Error: GROQ_API_KEY cannot be empty. Update .env")
        self.client = Groq(api_key=api_key)
        self.model = model

    def add_user_message(self, messages: list, message):
        # Handle tool results (Anthropic format -> Groq format)
        if isinstance(message, list) and len(message) > 0 and isinstance(message[0], dict) and message[0].get("type") == "tool_result":
            for part in message:
                messages.append({
                    "role": "tool",
                    "tool_call_id": part["tool_use_id"],
                    "content": part["content"]
                })
        else:
            # Normal user message
            content = getattr(message, "content", message)
            messages.append({
                "role": "user",
                "content": content
            })

    def add_assistant_message(self, messages: list, message):
        msg = {"role": "assistant"}
        
        # message is our MockMessage here
        content_text = ""
        if hasattr(message, "content") and isinstance(message.content, list):
            for block in message.content:
                if block.type == "text":
                    content_text += getattr(block, "text", "")
        
        if content_text:
            msg["content"] = content_text
        else:
            msg["content"] = ""
            
        if hasattr(message, "groq_tool_calls") and message.groq_tool_calls:
            msg["tool_calls"] = message.groq_tool_calls
            
        messages.append(msg)

    def text_from_message(self, message):
        if not hasattr(message, "content") or not isinstance(message.content, list):
            return ""
        return "\n".join([block.text for block in message.content if block.type == "text"])

    def chat(self, messages, system=None, temperature=1.0, stop_sequences=[], tools=None, thinking=False, thinking_budget=1024):
        groq_tools = None
        if tools:
            groq_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["input_schema"]
                    }
                } for t in tools
            ]
        
        groq_messages = []
        if system:
            groq_messages.append({"role": "system", "content": system})
        
        for m in messages:
            if isinstance(m, dict):
                content = m["content"]
                # If content is Anthropic blocks
                if isinstance(content, list):
                    content = "\n".join([b["text"] for b in content if b.get("type") == "text"])
                groq_messages.append({"role": m["role"], "content": content})
                if "tool_calls" in m:
                    groq_messages[-1]["tool_calls"] = m["tool_calls"]
                if "tool_call_id" in m:
                    groq_messages[-1]["tool_call_id"] = m["tool_call_id"]
            else:
                groq_messages.append(m)

        params = {
            "model": self.model,
            "messages": groq_messages,
            "temperature": temperature,
        }
        if groq_tools:
            params["tools"] = groq_tools

        response = self.client.chat.completions.create(**params)
        choice = response.choices[0]
        
        stop_reason = choice.finish_reason
        if stop_reason == "tool_calls":
            stop_reason = "tool_use"
            
        content_blocks = []
        if choice.message.content:
            content_blocks.append(SimpleNamespace(type="text", text=choice.message.content))
            
        groq_tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                # Add to anthropic style content blocks
                content_blocks.append(SimpleNamespace(
                    type="tool_use",
                    id=tc.id,
                    name=tc.function.name,
                    input=json.loads(tc.function.arguments)
                ))
                # Keep original for appending to groq_messages
                groq_tool_calls.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                })

        class MockMessage:
            def __init__(self, content, stop_reason, groq_tool_calls):
                self.content = content
                self.stop_reason = stop_reason
                self.groq_tool_calls = groq_tool_calls

        return MockMessage(content_blocks, stop_reason, groq_tool_calls)

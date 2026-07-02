import asyncio
import sys
import os
from dotenv import load_dotenv
from contextlib import AsyncExitStack

from mcp_client import MCPClient
from core.claude import Claude
from core.groq_service import GroqService

from core.cli_chat import CliChat
from core.cli import CliApp

load_dotenv()

def get_service():
    print("Select a model to use:")
    print("1. claude-sonnet-4-5")
    print("2. llama-3.3-70b-versatile")
    choice = input("Enter 1 or 2: ").strip()
    
    if choice == "2":
        groq_model = "llama-3.3-70b-versatile"
        # groq_model = "llama-3.1-8b-instant"
        return GroqService(model=groq_model)
    else:
        # Default to Claude
        claude_model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5")
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
        assert claude_model, "Error: CLAUDE_MODEL cannot be empty. Update .env"
        assert anthropic_api_key, "Error: ANTHROPIC_API_KEY cannot be empty. Update .env"
        return Claude(model=claude_model)

async def main():
    claude_service = get_service()

    server_scripts = sys.argv[1:]
    clients = {}

    command, args = (
        ("uv", ["run", "mcp_server.py"])
        if os.getenv("USE_UV", "0") == "1"
        else ("python", ["mcp_server.py"])
    )

    async with AsyncExitStack() as stack:
        doc_client = await stack.enter_async_context(
            MCPClient(command=command, args=args)
        )
        clients["doc_client"] = doc_client

        for i, server_script in enumerate(server_scripts):
            client_id = f"client_{i}_{server_script}"
            client = await stack.enter_async_context(
                MCPClient(command="uv", args=["run", server_script])
            )
            clients[client_id] = client

        chat = CliChat(
            doc_client=doc_client,
            clients=clients,
            claude_service=claude_service,
        )

        cli = CliApp(chat)
        await cli.initialize()
        await cli.run()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())

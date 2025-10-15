import os
import json
import asyncio
from contextlib import AsyncExitStack
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


load_dotenv()


class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.deepseek = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
        self.anthropic = Anthropic()

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server
        
        Args:
            server_script_path:  Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_javascript = server_script_path.endswith('.js')
        if not (is_python or is_javascript):
            raise ValueError("Server script must be a .py or .js file")
        
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None,
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        await self.session.initialize()
        
        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])
    
    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        response = await self.session.list_tools()
        available_tools = [{ 
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]

        print(">>> Available tools:", json.dumps(available_tools, indent=2))
        # Initial Claude API call
        # response = self.deepseek.chat.completions.create(
        #     model="deepseek-chat",
        #     max_tokens=1000,
        #     stream=False,
        #     messages=messages,
        #     tools=available_tools
        # )
        response = self.anthropic.messages.create(
            model="claude-3-7-sonnet-latest",
            messages=messages,
            max_tokens=1000,
            stream=False,
            tools=available_tools,
        )
        # response = self.deepseek.completions
        print(f">>> Deepseek response: {response.choices[0].message.content}")

        # Process response and handle tool calls
        tool_results = []
        final_text = []

        for content in response.choices:
            if content.type == 'text':
                final_text.append(content.text)
            elif content.type == 'tool_use':
                tool_name = content.name
                tool_args = content.input
                
                # Execute tool call
                result = await self.session.call_tool(tool_name, tool_args)
                tool_results.append({"call": tool_name, "result": result})
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                # Continue conversation with tool results
                if hasattr(content, 'text') and content.text:
                    messages.append({
                      "role": "assistant",
                      "content": content.text
                    })
                messages.append({
                    "role": "user", 
                    "content": result.content
                })

                # Get next response from Claude
                # response = self.deepseek.chat.completions.create(
                #     model="deepseek-chat",
                #     max_tokens=1000,
                #     stream=False,
                #     messages=messages,
                # )
                response = self.anthropic.messages.create(
                    model="claude-3-7-sonnet-latest",
                    messages=messages,
                    max_tokens=1000,
                    stream=False,
                    tools=available_tools,
                )

                final_text.append(response.content[0].text)

        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() == 'quit':
                    break
                    
                response = await self.process_query(query)
                print("\n" + response)
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)
        
    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        print("Connected to server.")

        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())
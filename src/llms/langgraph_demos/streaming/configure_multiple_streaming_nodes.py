"""

Reference: https://langchain-ai.github.io/langgraph/how-tos/stream-multiple/#setup
"""
from typing import Literal

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.constants import Send


@tool
def get_weather(city: Literal["nyc", "sf"]):
    """Use this to get weather information."""
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"
    else:
        raise AssertionError("Unknown city")


tools = [get_weather]

model = ChatOpenAI(temperature=0)
graph = create_react_agent(model, tools)

inputs = {"messages": [("human", "what's the weather in sf")]}


async def execute():
    async for event, chunk in graph.astream(inputs, stream_mode=["updates", "debug"]):
        print(f"Receiving new event of type: {event}...")
        print(chunk)
        print("\n\n")


if __name__ == '__main__':
    import asyncio
    asyncio.run(execute())

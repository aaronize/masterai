from typing import TypedDict, Annotated
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent

toolkit = SQLDatabaseToolkit(db=None, llm=None)
tools = toolkit.get_tools()

agent_executor = create_react_agent()
agent_executor.invoke()

def reduce_list(left: list | None, right: list | None) -> list:
    if not left:
        left = []
    if not right:
        right = []
    return left + right


class ChildState(TypedDict):
    name: str
    path: Annotated[list[str], reduce_list]


child_builder = StateGraph(ChildState)


if __name__ == '__main__':
    pass

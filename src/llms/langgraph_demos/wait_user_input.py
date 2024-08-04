"""

Reference: https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/wait-user-input/
"""
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel

from src.llms.langgraph_demos.tools import search


class AskHuman(BaseModel):
    """Ask the human a question"""
    question: str


class HumanConfirmAgent(object):

    def __init__(self):
        tools = [search]
        self.tool_node = ToolNode(tools)
        self.model = ChatOpenAI().bind_tools(tools + [AskHuman])

        self.app = self.build()

    def should_continue(self, state: MessagesState):
        """"""
        messages = state["messages"]
        last_message = messages[-1]

        if not last_message.tool_calls:
            return "end"
        elif last_message.tool_calls[0]["name"] == "AskHuman":
            return "ask_human"
        else:
            return "continue"

    def call_model(self, state: MessagesState):
        """"""
        messages = state["messages"]
        response = self.model.invoke(messages)

        return {"messages": [response]}

    def ask_human(self, state: MessagesState):
        """"""
        pass

    def build(self):
        """"""
        workflow = StateGraph(MessagesState)

        workflow.add_node("agent", self.call_model)
        workflow.add_node("action", self.tool_node)
        workflow.add_node("ask_human", self.ask_human)

        workflow.add_edge(START, "agent")

        workflow.add_conditional_edges(
            "agent",
            self.should_continue,
            {
                "continue": "action",
                "ask_human": "ask_human",
                "end": END,
            }
        )

        workflow.add_edge("action", "agent")

        workflow.add_edge("ask_human", "agent")

        memory = MemorySaver()

        return workflow.compile(checkpointer=memory, interrupt_before=["ask_human"])


if __name__ == '__main__':
    agent = HumanConfirmAgent()

    config = {"configurable": {"thread_id": "2"}}
    input_message = HumanMessage(
        content="Use the search tool to ask the user where they are, then look up the weather there"
    )

    for event in agent.app.stream({"messages": [input_message]}, config, stream_mode="values"):
        print(">>>n")
        event["messages"][-1].pretty_print()

    tool_call_id = agent.app.get_state(config).values["messages"][-1].tool_calls[0]["id"]
    tool_message = [
        {"tool_call_id": tool_call_id, "type": "tool", "content": "san francisco"}
    ]

    agent.app.update_state(config, {"messages": tool_message}, as_node="ask_human")
    print(">>>> g:", agent.app.get_state(config).next)

    for event in agent.app.stream(None, config, stream_mode="values"):
        event["messages"][-1].pretty_print()




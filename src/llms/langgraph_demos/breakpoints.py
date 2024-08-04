"""

Reference: https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/breakpoints/#agent
"""
from typing import Type

from IPython.display import Image, display
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from src.llms.langgraph_demos.tools import search


class AgentWithBreakpoints(object):

    def __init__(self):
        self.engine = None
        tools = [search]
        self.tool_node = ToolNode(tools)
        self.model = ChatOpenAI(temperature=0).bind_tools(tools)

        self.build_engine()

    def should_continue(self, state: MessagesState):
        """

        :param state:
        :return:
        """
        messages = state["messages"]
        last_message = messages[-1]

        if not last_message.tool_calls:
            return "end"
        else:
            return "continue"

    def call_model(self, state: MessagesState):
        """

        :param state:
        :return:
        """
        messages = state["messages"]
        response = self.model.invoke(messages)

        return {"messages": [response]}

    def build_engine(self):
        """"""
        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", self.call_model)
        workflow.add_node("action", self.tool_node)

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            self.should_continue,
            {
                "continue": "action",
                "end": END,
            }
        )
        workflow.add_edge("action", "agent")

        memory = MemorySaver()

        self.engine = workflow.compile(checkpointer=memory, interrupt_before=["action"])
        # display(Image(self.engine.get_graph().draw_mermaid_png()))

    # def display_engine_graph(self):
    #     display(Image(self.engine))


if __name__ == '__main__':
    agent = AgentWithBreakpoints()

    thread = {"configurable": {"thread_id": "3"}}
    inputs = [HumanMessage(content="search for the weather in sf now")]
    for event in agent.engine.stream({"messages": inputs}, thread, stream_mode="values"):
        event["messages"][-1].pretty_print()

    # 上面会在执行action前停止，下面继续调用才会继续执行
    for event in agent.engine.stream(None, thread, stream_mode="values"):
        event["messages"][-1].pretty_print()

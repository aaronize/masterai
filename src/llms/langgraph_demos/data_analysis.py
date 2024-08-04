from typing import List, Union, Any, Literal, Type

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseLLM
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint import MemorySaver
from langgraph.constants import END
from langgraph.graph import MessagesState, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import ToolNode, create_react_agent

from llm_cases.data_analyser.analyse_tools.database import QueryMongoTool
from llm_cases.data_analyser.prompts import SYSTEM_PROMPT_T


class AnalysisAgent:
    def __init__(self, model: BaseLLM = None, tools: List[BaseTool] = None):
        self.model = model
        self.tools = tools
        self.engine: Type[CompiledGraph, None] = None

        # set default tools
        if not tools:
            self.tools = [QueryMongoTool()]

        # system prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT_T),
            MessagesPlaceholder(variable_name="messages")
        ])

        if not self.model:
            # self.model = OpenAI(model_name="gpt-4", temperature=0)
            # chat
            self.model = prompt | ChatOpenAI(model="gpt-4").bind_tools(self.tools)

        self.build_engine()

        # Graph create_react_agent doc: https://langchain-ai.github.io/langgraph/reference/prebuilt/#create_react_agent
        # self.agent = create_react_agent(model=self.model, tools=self.tools, messages_modifier=prompt)

    def build_engine(self):
        """
        构建workflow
        :return:
        """
        tool_node = ToolNode(self.tools)

        workflow = StateGraph(MessagesState)

        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", tool_node)

        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", self._should_continue)
        # workflow.add_conditional_edges("agent", self._routing)

        workflow.add_edge("tools", "agent")

        check_pointer = MemorySaver()
        self.engine = workflow.compile(checkpointer=check_pointer)

    def _call_model(self, state: MessagesState):
        """
        调用模型
        :param state:
        :return:
        """
        messages = state["messages"]
        response = self.model.invoke(messages)

        return {"messages": [response]}

    @staticmethod
    def _should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
        """
        决定是否继续
        :param state:
        :return:
        """
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tools"

        return END

    @staticmethod
    def _routing(state: MessagesState) -> Literal["tools", "continue", "__end__"]:
        """

        :param state:
        :return:
        """
        last_message = state["messages"][-1]

        if last_message.tool_calls:
            return "tools"
        if "FINAL ANSWER" in last_message.content:
            return END

        return "continue"

    def invoke(self, message: str) -> Union[dict[str, Any], Any]:
        """
        调用
        :param message:
        :return:
        """

        input_msg = {"messages": [HumanMessage(content=message)]}
        response = self.engine.invoke(input_msg, config={"configurable": {"thread_id": 42}})

        return response["messages"][-1].content


def data_analysis(task_id: int, message: str) -> str:
    if not task_id:
        return ""

    agent = AnalysisAgent()

    res = agent.invoke(f"ID为{task_id}的任务下, {message}")

    return res


if __name__ == '__main__':
    res = data_analysis(task_id=170782, message="是否存在未加密的通信？")
    print(res)
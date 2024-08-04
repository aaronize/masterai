from typing import TypedDict, Annotated

from langgraph.graph import StateGraph


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

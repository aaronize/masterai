"""

"""
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class Add(BaseModel):
    """Add two integers together."""

    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")




class Multiply(BaseModel):
    """Multiply two integers together."""

    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")


tools = [Add, Multiply]


def main():
    """"""
    # llm = ChatOpenAI(model="gpt-4")
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    llm_with_tools = llm.bind_tools(tools)
    output = llm_with_tools.invoke(input="what is 2+4?")
    print(output)


if __name__ == '__main__':
    main()

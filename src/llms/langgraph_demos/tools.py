from langchain_core.tools import tool


@tool
def search(query: str):
    """Call to surf the web."""
    print(f">>> query kw: {str} in search tool")
    return [
        "It's sunny in San Francisco, but you better look out if you're a Gemini 😈."
    ]
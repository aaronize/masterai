import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def get_deepseek_resp():
    """"""
    ds = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
    respone = ds.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "user",
                "content": "What is the capital of France?"
            }
        ],
        stream=False,
    )

    print(">>> Deepseek response:", respone.choices[0].message.content)


if __name__ == "__main__":
    get_deepseek_resp()
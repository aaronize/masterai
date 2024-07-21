"""

Reference: https://docs.chainlit.io/examples/openai-sql
"""

from openai import AsyncOpenAI

import chainlit as cl

# from src.utils.env import OPENAI_API_KEY

cl.instrument_openai()
client = AsyncOpenAI()

template = """SQL tables (and columns):
* Customers(customer_id, signup_date)
* Streaming(customer_id, video_id, watch_date, watch_minutes)

A well-written SQL query that {input}:
```"""


settings = {
    "model": "gpt-3.5-turbo",
    "temperature": 0,
    "max_tokens": 500,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "stop": ["```"],
}


@cl.set_starters
async def starters():
    return [
        cl.Starter(
            label=">50 minutes watched",
            message="Compute the number of customers who watched more than 50 minutes of video this month."
        )
    ]


@cl.on_message
async def main(message: cl.Message):
    stream = await client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": template.format(input=message.content),
            }
        ], stream=True, **settings
    )

    msg = await cl.Message(content="", language="sql").send()

    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await msg.stream_token(token)

    await msg.update()


if __name__ == '__main__':
    pass

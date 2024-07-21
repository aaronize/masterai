"""

Reverence: https://github.com/langchain-ai/rag-from-scratch/blob/main/rag_from_scratch_5_to_9.ipynb
"""
from operator import itemgetter

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_openai import ChatOpenAI

from src.llms.rag_from_scratch.prompts import RAG_QUESTION_COMPOSITION_TEMPLATE, RAG_DECOMPOSITION_TEMPLATE, \
    RAG_QA_PAIRS_TEMPLATE
from src.llms.rag_from_scratch.vector_store import get_retriever


# Retriever
retriever = get_retriever()

prompt_decomposition = ChatPromptTemplate.from_template(RAG_QUESTION_COMPOSITION_TEMPLATE)

llm = ChatOpenAI(temperature=0)

# decomposition chain
generate_queries_decomposition = (
    prompt_decomposition |
    llm |
    StrOutputParser() |
    (lambda x: x.split("\n"))
)

# Run
question = "What are the main decomposition of an LLM-powered autonomous agent system?"
questions = generate_queries_decomposition.invoke({"question": question})

print("decomposition questions: ", questions)

# Prompt
decomposition_prompt = ChatPromptTemplate.from_template(RAG_DECOMPOSITION_TEMPLATE)


def format_qa_pair(question, answer):
    """
    Format Q and A pair.
    :param question:
    :param answer:
    :return:
    """
    formatted_string = ""
    formatted_string += f"Question: {question}\nAnswer: {answer}\n\n"

    return formatted_string.strip()


# llm
llm = ChatOpenAI(temperature=0)

q_a_pairs = ""
for q in questions:
    rag_chain = (
        {"context": itemgetter("question") | retriever,
         "question": itemgetter("question"),
         "q_a_pairs": itemgetter("q_a_pairs")} |
        decomposition_prompt |
        llm |
        StrOutputParser()
    )

    answer = rag_chain.invoke({"question": q, "q_a_pairs": q_a_pairs})
    print(">>>>", answer)
    q_a_pair = format_qa_pair(q, answer)
    q_a_pairs = q_a_pairs + "\n---\n" + q_a_pair


prompt_rag = hub.pull("rlm/rag-prompt")
"""
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""


def retrieve_and_rag(question: str, prompt_rag: ChatPromptTemplate, sub_question_generator_chain: RunnableSerializable):
    """
    RAG on each sub-question.
    :param question:
    :param prompt_rag:
    :param sub_question_generator_chain:
    :return:
    """
    sub_questions = sub_question_generator_chain.invoke({"question": question})

    rag_results = []
    for sub_question in sub_questions:
        retrieved_docs = retriever.invoke(sub_question)

        answer = (prompt_rag | llm | StrOutputParser()).invoke({"context": retrieved_docs, "question": sub_question})

        rag_results.append(answer)

    return rag_results, sub_questions


# Wrap the retrieval and RAG process in a RunnableLambda for integration into a chain.
answers, questions = retrieve_and_rag(question, prompt_rag, generate_queries_decomposition)


def format_qa_pairs(questions, answers):
    """
    Format Q and A pairs.
    :param questions:
    :param answers:
    :return:
    """
    formatted_string = ""
    for i, (question, answer) in enumerate(zip(questions, answers), start=1):
        formatted_string += f"Question: {i}: {question}\nAnswer: {i}: {answer}\n\n"

    return formatted_string.strip()


context = format_qa_pairs(questions, answers)
prompt = ChatPromptTemplate.from_template(RAG_QA_PAIRS_TEMPLATE)

final_rag_chain = (
    prompt |
    llm |
    StrOutputParser()
)


if __name__ == '__main__':
    res = final_rag_chain.invoke({"context": context, "question": question})
    print(">>> final result:", res)

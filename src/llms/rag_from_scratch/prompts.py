"""
Prompts

"""

# RAG general
RAG_TEMPLATE = """Answer the following question based on this context:

{context}

Question: {question}"""

# RAG Multi query
RAG_MULTI_QUERY_TEMPLATE = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""

# RAG-Fusion
RAG_FUSION_TEMPLATE = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output (4 queries):"""


# RAG Decomposition
RAG_QUESTION_COMPOSITION_TEMPLATE = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
Generate multiple search queries related to: {question} \n
Output (3 queries):"""

RAG_DECOMPOSITION_TEMPLATE = """Here is the question you need to answer:

\n --- \n {question} \n --- \n

Here is any available background question + answer pairs:

\n --- \n {q_a_pairs} \n --- \n

Here is additional context relevant to the question: 

\n --- \n {context} \n --- \n

Use the above context and any background question + answer pairs to answer the question: \n {question}
"""

#
RAG_QA_PAIRS_TEMPLATE = """Here is a set of Q+A pairs:

{context}

Use these to synthesize an answer to the question: {question}
"""

# RAG HyDE prompt
RAG_HYDE_TEMPLATE = """Please write a scientific paper passage to answer the question
Question: {question}
Passage:"""

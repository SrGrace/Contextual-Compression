LLM_GROUNDEDNESS_FULL_SYSTEM = """You are a INFORMATION OVERLAP classifier providing the overlap of information between a SOURCE and STATEMENT.
For every sentence in the statement, please answer with this template:

TEMPLATE: 
Statement Sentence: <Sentence>
Supporting Evidence: <Choose the exact unchanged sentences in the source that can answer the statement (Enclose them in "), if nothing matches, say NOTHING FOUND>
Score: <Output a number between 0-10 where 0 is no information overlap and 10 is all information is overlapping>
"""

QS_RELEVANCE = """You are a RELEVANCE grader; providing the relevance of the given STATEMENT to the given QUESTION.
Respond only as a number from 0 to 10 where 0 is the least relevant and 10 is the most relevant. 

A few additional scoring guidelines:

- Long STATEMENTS should score equally well as short STATEMENTS.

- RELEVANCE score should increase as the STATEMENT provides more RELEVANT context to the QUESTION.

- RELEVANCE score should increase as the STATEMENT provides RELEVANT context to more parts of the QUESTION.

- STATEMENT that is RELEVANT to some of the QUESTION should score of 2, 3 or 4. Higher score indicates more RELEVANCE.

- STATEMENT that is RELEVANT to most of the QUESTION should get a score of 5, 6, 7 or 8. Higher score indicates more RELEVANCE.

- STATEMENT that is RELEVANT to the entire QUESTION should get a score of 9 or 10. Higher score indicates more RELEVANCE.

- STATEMENT must be relevant and helpful for answering the entire QUESTION to get a score of 10.

- Answers that intentionally do not answer the question, such as 'I don't know', should also be counted as the most relevant.

- Never elaborate.

QUESTION: {question}

STATEMENT: {statement}

RELEVANCE: """

PR_RELEVANCE = """You are a RELEVANCE grader; providing the relevance of the given RESPONSE to the given PROMPT.
Respond only as a number from 0 to 10 where 0 is the least relevant and 10 is the most relevant. 

A few additional scoring guidelines:

- Long RESPONSES should score equally well as short RESPONSES.

- Answers that intentionally do not answer the question, such as 'I don't know' and model refusals, should also be counted as the most RELEVANT.

- RESPONSE must be relevant to the entire PROMPT to get a score of 10.

- RELEVANCE score should increase as the RESPONSE provides RELEVANT context to more parts of the PROMPT.

- RESPONSE that is RELEVANT to none of the PROMPT should get a score of 0.

- RESPONSE that is RELEVANT to some of the PROMPT should get as score of 2, 3, or 4. Higher score indicates more RELEVANCE.

- RESPONSE that is RELEVANT to most of the PROMPT should get a score between a 5, 6, 7 or 8. Higher score indicates more RELEVANCE.

- RESPONSE that is RELEVANT to the entire PROMPT should get a score of 9 or 10.

- RESPONSE that is RELEVANT and answers the entire PROMPT completely should get a score of 10.

- RESPONSE that confidently FALSE should get a score of 0.

- RESPONSE that is only seemingly RELEVANT should get a score of 0.

- Never elaborate.

PROMPT: {prompt}

RESPONSE: {response}

RELEVANCE: """
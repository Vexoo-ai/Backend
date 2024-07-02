from app.api.v1.core.format import *
from app.api.v1.web_crawler.search import * 
import app.api.v1.web_crawler.search as serp_calls
import re
from dotenv import load_dotenv
import os
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from datetime import datetime

def configure():
    load_dotenv()

import re
from tenacity import retry, stop_after_attempt, wait_fixed

import os
import json
from datetime import datetime
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from dotenv import load_dotenv

load_dotenv()

def call_freshprompt(model, question, check_premise=False, verbose=False):
    # Environment setup
    endpoint = os.getenv('endpoint')
    api_key = os.getenv('api_key')
    client = MistralClient(api_key=api_key, endpoint=endpoint)
    model = "azureai"
    # Search parameters
    search_data = call_search_engine(question)
    temperature = 0.3
    max_tokens = 1000  # Increased for longer, more comprehensive responses
    
    num_organic_results = 20
    num_related_questions = 8
    num_questions_and_answers = 8
    num_retrieved_evidences = 20

    # Prepare source data
    sources = []
    for idx, result in enumerate(search_data.get('organic_results', [])[:num_organic_results]):
        sources.append({
            'id': idx + 1,
            'title': result.get('title', ''),
            'link': result.get('link', ''),
            'snippet': result.get('snippet', '')
        })

    sources_json = json.dumps(sources)

    # Construct the prompt
    current_date = datetime.now().strftime("%Y-%m-%d")
    suffix = f"""Provide a comprehensive answer to the following question. Include source attributions for each statement:

Question: {question}

Your response should:
1. Thoroughly address all aspects of the question
2. For each statement or piece of information, include a superscript number reference at the end, e.g. "This is a fact.[1]"
3. Use multiple sources where appropriate, e.g. "This is a complex idea[1][2][3]"
4. If coding is involved, provide complete, executable code solutions with explanations
5. Break down complex concepts into understandable parts
6. Address potential challenges, limitations, or considerations

Here are the sources you can reference:
{sources_json}

At the end of your response, include a "Sources:" section listing all the sources used, numbered corresponding to their superscript references.

Answer: """

    freshprompt_question = freshprompt_format(
        question,
        search_data,
        suffix,
        num_organic_results,
        num_related_questions,
        num_questions_and_answers,
        num_retrieved_evidences,
    )

    system_message = f"""You are 'VEXO', an advanced AI research assistant. Your capabilities include:

1. Providing comprehensive and detailed answers to questions.
2. Including source attributions for each statement using superscript numbers, e.g. [1], [2], etc.
3. Using [0] for general knowledge not from provided sources.
4. Analyzing complex topics from multiple angles.
5. Generating code and technical explanations when relevant.
6. Considering ethical implications and potential biases in information.

When responding:
- Ensure each statement has a source attribution.
- Use multiple sources where appropriate, e.g. "This is a complex idea[1][2][3]"
- Provide thorough explanations, breaking down complex concepts.
- If coding is involved, include complete, executable code with explanations.
- Address potential challenges, limitations, or considerations.
- At the end of your response, include a "Sources:" section listing all referenced sources.

Your goal is to provide the most informative, accurate, and well-sourced response possible. Knowledge cutoff: {current_date}."""

    messages = [
        ChatMessage(role="system", content=system_message),
        ChatMessage(role="user", content=freshprompt_question)
    ]

    # Make the API call
    chat_response = client.chat(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        safe_mode=False
    )

    answer = chat_response.choices[0].message.content

    # Check if sources are included, if not, generate them
    if "Sources:" not in answer:
        sources_prompt = f"Based on the following answer, generate a 'Sources:' section listing all referenced sources:\n\n{answer}"
        sources_response = client.chat(
            model=model,
            messages=[ChatMessage(role="user", content=sources_prompt)],
            temperature=0.2,
            max_tokens=500,
            safe_mode=False
        )
        sources_section = sources_response.choices[0].message.content
        answer += "\n\n" + sources_section

    return (answer, freshprompt_question, num_related_questions)



def call_freshprompt_stream(model, question, check_premise=False, verbose=False):
    print(f"Processing query: {question}")
    
    search_data = serp_calls.call_search_engine(question)
    temperature = 0.3  # Lowered for more consistent outputs
    max_tokens = 1000  # Increased for longer responses

    num_organic_results = 20
    num_related_questions = 8
    num_questions_and_answers = 8
    num_retrieved_evidences = 20

    suffix = f"""Provide a comprehensive and detailed response to the following query. If the query involves coding or technical instructions, include complete, executable code snippets and step-by-step explanations:

Query: {question}

Your response should:
1. Directly and thoroughly address all aspects of the query
2. For general topics: Provide in-depth explanations, examples, and relevant context
3. For coding tasks: 
   - Offer complete, executable code solutions
   - Explain the code in detail, including the reasoning behind design choices
   - Consider edge cases and potential optimizations
   - Provide usage examples and expected outputs
4. For technical explanations: Break down complex concepts into understandable parts
5. For problem-solving: Outline a clear approach, including alternatives if applicable
6. Address potential challenges, limitations, or considerations
7. Suggest related areas for further exploration or learning

IMPORTANT: Ensure your response is comprehensive, accurate, and directly applicable to the user's needs. For code, prioritize correctness, efficiency, and readability. Always conclude your response with a summary or closing statement."""

    freshprompt_question = freshprompt_format(
        question, search_data, suffix,
        num_organic_results, num_related_questions,
        num_questions_and_answers, num_retrieved_evidences
    )
    
    print(f"Formatted freshprompt question: {freshprompt_question}")
    
    client = MistralClient(api_key=os.getenv('api_key'), endpoint=os.getenv('endpoint'))
    model = "azureai"

    current_date = datetime.now().strftime("%Y-%m-%d")
    system_message = f"""You are VEXOO, an advanced AI assistant capable of handling a wide range of tasks, including complex coding and technical challenges. You were built by Aditya Vardhan. Your capabilities include:

1. Comprehensive Knowledge: Vast understanding across various fields, including programming, science, mathematics, and general topics.
2. Code Generation: Ability to write efficient, correct, and readable code in multiple programming languages.
3. Technical Problem-Solving: Skill in breaking down and solving complex technical issues.
4. Detailed Explanations: Capacity to explain intricate concepts and code in an understandable manner.
5. Adaptability: Flexibility to switch between different types of tasks seamlessly.
6. Creative Solutions: Innovative approaches to unique or challenging problems.
7. Best Practices: Knowledge of industry standards and best practices in software development and other technical fields.

When responding:
- For coding tasks: Provide complete, executable code solutions. Include detailed explanations, consider edge cases, and suggest optimizations.
- For technical queries: Offer step-by-step explanations, use analogies when helpful, and break down complex topics.
- For general questions: Give comprehensive, well-structured responses with relevant examples and context.
- For problem-solving: Outline clear approaches, consider alternatives, and discuss pros and cons.

Always strive for accuracy, clarity, and practicality in your responses. Tailor your language and depth to the complexity of the query.

Your goal is to provide responses that not only answer the immediate question but also enhance the user's understanding and ability to apply the information.

CRITICAL: Always conclude your response with a clear summary or closing statement to ensure completeness.

Knowledge cutoff: {current_date}"""

    messages = [
        ChatMessage(role="system", content=system_message),
        ChatMessage(role="user", content=freshprompt_question)
    ]
    
    def generate_response():
        full_response = ""
        chat_response = client.chat_stream(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            safe_mode=False
        )
        
        for chunk in chat_response:
            content = chunk.choices[0].delta.content
            full_response += content
            yield content

        # Check if the response seems incomplete or lacks a conclusion
        if not re.search(r'(conclusion|summary|in summary|to summarize|in conclusion)', full_response.lower()) or \
           not re.search(r'(\.\s*$|\}\s*$)', full_response.strip()):
            completion_message = "\n\nTo conclude: "
            summary = client.chat(
                model=model,
                messages=[
                    ChatMessage(role="system", content="Complete the following response by providing a concise summary of the main points discussed. Ensure all code blocks are closed and the explanation is finalized:"),
                    ChatMessage(role="user", content=full_response + completion_message)
                ],
                max_tokens=500,
                temperature=0.2
            )
            yield completion_message + summary.choices[0].message.content

    for content in generate_response():
        yield content

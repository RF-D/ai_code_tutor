import os
import streamlit as st
from typing import List, Dict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from utils.llm_manager import LLMManager
from dotenv import load_dotenv
from streamlit_ace import st_ace
import random
import time
import logging
import re


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

load_dotenv()

LLMManager.initialize_ollama_models()

st.set_page_config(page_title="Learn to Code with AI", page_icon="📡")


CODE_EVALUATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            """You are an AI teaching assistant specialized in evaluating Python code. Your task is to analyze the submitted code and evaluate it against the given practice question. Evaluate the code based on:
1. Correctness: Does it solve the problem posed in the practice question?
2. Syntax correctness
3. Functionality
4. Code style and best practices
5. Concept understanding

Practice Question: {practice_question}

Provide feedback in the following format:
- Overall assessment: (Correct/Partially Correct/Incorrect)
- Correctness: (Does the code correctly solve the problem posed in the practice question?)
- Comments: (Detailed explanation of what the code does well or where it falls short)
- Suggestions: (Specific ideas for improvement)"""
        ),
        HumanMessagePromptTemplate.from_template(
            "Here's the code to evaluate:\n\n{code}\n\nPlease provide your evaluation."
        ),
    ]
)

PRACTICE_QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            """You are an AI teaching assistant specialized in creating Python programming practice questions. Your task is to generate a practice question based on the given topic or concept. The question should be:
1. Clear and concise
2. Appropriate for the user's skill level
3. Focused on a specific Python concept or skill
4. Accompanied by a brief explanation of what the question is testing

Provide the practice question in the following format:
Topic: (The main Python concept being tested)
Question: (The actual practice question)
Explanation: (A brief explanation of what the question is testing and why it's important)
Skill Level: (The skill level this question is appropriate for)
Justification: (Why this question is appropriate for the given skill level)"""
        ),
        HumanMessagePromptTemplate.from_template(
            "Please generate a practice question for the following topic: {topic}. The user's skill level is: {skill_level}."
        ),
    ]
)


def generate_topic_suggestions():
    topic_prompt = """Generate 5 diverse and interesting Python topic suggestions for practice questions. 
    Each suggestion should cover a different area of Python programming, such as:
    1. Basic concepts (e.g., data types, control structures)
    2. Data structures and algorithms
    3. Object-oriented programming
    4. File handling and I/O
    5. Libraries and modules (e.g., NumPy, Pandas, Matplotlib)
    6. Web development (e.g., Flask, Django)
    7. Data processing and analysis
    8. Networking and APIs
    9. Testing and debugging
    10. Advanced Python features (e.g., decorators, generators)

    Ensure that the suggestions are specific enough to create focused questions but broad enough to allow for 
    various difficulty levels. Format the response as a numbered list."""

    suggestions = question_gen_llm.invoke(topic_prompt)
    if isinstance(suggestions, AIMessage):
        return suggestions.content.split("\n")
    elif isinstance(suggestions, str):
        return suggestions.split("\n")
    else:
        raise ValueError(f"Unexpected type for suggestions: {type(suggestions)}")


# Helper function to parse the AI's response
def parse_practice_question(response):
    parsed = {
        "topic": "",
        "question": "",
        "explanation": "",
        "skill_level": "",
        "justification": "",
    }

    # Use regex to find each section
    patterns = {
        "topic": r"Topic:\s*(.*?)(?=Question:|$)",
        "question": r"Question:\s*(.*?)(?=Explanation:|$)",
        "explanation": r"Explanation:\s*(.*?)(?=Skill Level:|$)",
        "skill_level": r"Skill Level:\s*(.*?)(?=Justification:|$)",
        "justification": r"Justification:\s*(.*?)$",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            parsed[key] = match.group(1).strip()

    # Fill in any missing fields
    for key in parsed:
        if not parsed[key]:
            parsed[key] = f"[No {key.replace('_', ' ').title()} provided]"

    return parsed


def debug_input(x):
    print(f"Input to LLM: {x}")
    return x


def setup_sidebar():
    st.sidebar.title("Python Learning Assistant Configuration")

    # LLM selection for Code Evaluation
    code_eval_provider = st.sidebar.selectbox(
        "AI for Code Evaluation", list(LLMManager.get_provider_models().keys())
    )
    code_eval_model = st.sidebar.selectbox(
        "Model for Code Evaluation",
        LLMManager.get_models_for_provider(code_eval_provider),
    )

    st.sidebar.write("---")

    # LLM selection for Practice Question Generation
    question_gen_provider = st.sidebar.selectbox(
        "AI for Question Generation", list(LLMManager.get_provider_models().keys())
    )
    question_gen_model = st.sidebar.selectbox(
        "Model for Question Generation",
        LLMManager.get_models_for_provider(question_gen_provider),
    )

    return (
        code_eval_provider,
        code_eval_model,
        question_gen_provider,
        question_gen_model,
    )


# Streamlit UI components
st.title("Python Learning Assistant")

# Initialize session state
if "code_input" not in st.session_state:
    st.session_state.code_input = ""
if "temp_code_input" not in st.session_state:
    st.session_state.temp_code_input = ""
if "editor_key" not in st.session_state:
    st.session_state.editor_key = 0
if "code_evaluation" not in st.session_state:
    st.session_state.code_evaluation = ""
if "topic_input" not in st.session_state:
    st.session_state.topic_input = ""
if "skill_level" not in st.session_state:
    st.session_state.skill_level = "Beginner"
if "practice_question" not in st.session_state:
    st.session_state.practice_question = ""
if "current_practice_question" not in st.session_state:
    st.session_state.current_practice_question = None
if "question_generated" not in st.session_state:
    st.session_state.question_generated = False
if "new_question_generated" not in st.session_state:
    st.session_state.new_question_generated = False

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to", ["Practice Questions", "Code Evaluation", "Solution Assistant"]
)


code_eval_provider, code_eval_model, question_gen_provider, question_gen_model = (
    setup_sidebar()
)

# Load the LLMs after the sidebar setup
code_eval_llm = LLMManager.load_llm(code_eval_provider, code_eval_model)
question_gen_llm = LLMManager.load_llm(question_gen_provider, question_gen_model)

# Set up the chain for practice question generation
practice_question_chain = (
    PRACTICE_QUESTION_PROMPT
    | RunnableLambda(debug_input)
    | question_gen_llm
    | StrOutputParser()
)

# Set up the chain for code evaluation
code_evaluation_chain = CODE_EVALUATION_PROMPT | code_eval_llm | StrOutputParser()


def generate_hint(question, practice_question, code_input=None):
    hint_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                """You are an AI teaching assistant specialized in providing hints for Python programming questions. 
            Your task is to provide a helpful hint based on the practice question and the user's specific question. 
            If the user has provided code, use it to give more specific hints. 
            The hint should guide the user towards the solution without giving it away completely."""
            ),
            HumanMessagePromptTemplate.from_template(
                """Practice Question: {practice_question}
            
            {code_section}
            
            User's Question: {question}
            
            Please provide a helpful hint:"""
            ),
        ]
    )

    code_section = ""
    if code_input:
        code_section = f"""User's Current Code:
            ```python
            {code_input}
            ```
            """

    hint_chain = hint_prompt | question_gen_llm | StrOutputParser()

    return hint_chain.invoke(
        {
            "practice_question": practice_question,
            "code_section": code_section,
            "question": question,
        }
    )


def update_code_input():
    st.session_state.code_input = st.session_state.temp_code_input


# Modify the generate_practice_question function
def generate_practice_question():
    if st.session_state.topic_input:
        with st.spinner("Generating practice question..."):
            logging.info(f"Topic input: {st.session_state.topic_input}")
            logging.info(f"Skill level: {st.session_state.skill_level}")
            try:
                raw_response = practice_question_chain.invoke(
                    {
                        "topic": st.session_state.topic_input,
                        "skill_level": st.session_state.skill_level,
                    }
                )
                logging.info(f"Raw response from Ollama: {raw_response}")

                # Check if the response is a string or an AIMessage
                if isinstance(raw_response, str):
                    response_text = raw_response
                elif hasattr(raw_response, "content"):
                    response_text = raw_response.content
                else:
                    raise ValueError(f"Unexpected response type: {type(raw_response)}")

                st.session_state.current_practice_question = parse_practice_question(
                    response_text
                )
                logging.info(
                    f"Parsed question: {st.session_state.current_practice_question}"
                )
                st.session_state.question_generated = True
                st.session_state.new_question_generated = True
                st.session_state.code_input = ""  # Clear the code input
                st.session_state.hint_messages = []  # Clear the hint chat history

                # Display parsed response
                if "current_practice_question" != st.session_state.practice_question:
                    st.empty()  # Clear previous content
                st.subheader("Current Practice Question")
                for key, value in st.session_state.current_practice_question.items():
                    if value:  # Only display non-empty fields
                        st.markdown(f"**{key.capitalize()}:** {value}")
                    else:
                        st.warning(f"Missing {key} in the generated question.")
            except Exception as e:
                logging.error(f"Error generating practice question: {str(e)}")
                st.error(
                    f"An error occurred while generating the practice question: {str(e)}"
                )
    else:
        st.warning("Please enter a topic or concept.")


if page == "Practice Questions":
    st.header("Generate Practice Questions")

    # Display only the most recent practice question
    if st.session_state.question_generated:
        st.subheader("Current Practice Question")
        for key, value in st.session_state.current_practice_question.items():
            if value:
                st.markdown(f"**{key.capitalize()}:** {value}")

    # Add a button to generate topic suggestions
    if st.button("Suggest Topics"):
        suggestions = generate_topic_suggestions()
        st.session_state.topic_suggestions = suggestions
        st.subheader("Suggested Topics:")
        for suggestion in suggestions:
            st.write(suggestion)

    # Topic input (now with an option to use suggestions)
    if "topic_suggestions" in st.session_state and st.session_state.topic_suggestions:
        selected_topic = st.selectbox(
            "Select a suggested topic or enter your own:",
            ["Enter your own"] + st.session_state.topic_suggestions,
        )
        if selected_topic == "Enter your own":
            st.session_state.topic_input = st.text_input(
                "Enter the Python topic or concept you want to practice:",
                value=st.session_state.topic_input,
            )
        else:
            # Extract the core topic from the selected suggestion
            core_topic = selected_topic.split(":", 1)[-1].strip()
            st.session_state.topic_input = core_topic
    else:
        st.session_state.topic_input = st.text_input(
            "Enter the Python topic or concept you want to practice:",
            value=st.session_state.topic_input,
        )

    # Skill level selection
    st.session_state.skill_level = st.select_slider(
        "Select your skill level:",
        options=["Beginner", "Intermediate", "Advanced"],
        value=st.session_state.skill_level,
    )

    # Generate question button
    if st.button("Generate Practice Question"):
        generate_practice_question()

elif page == "Code Evaluation":
    st.header("Code Evaluation")

    # Display current practice question
    if st.session_state.question_generated:
        st.subheader("Current Practice Question")
        st.write(st.session_state.current_practice_question["question"])
    else:
        st.warning(
            "No practice question generated yet. Please go to the Practice Questions page to generate a question first."
        )

    # Code input area using st_ace
    st.session_state.temp_code_input = st_ace(
        value=st.session_state.code_input,
        language="python",
        theme="dracula",
        show_gutter=True,
        auto_update=True,
        wrap=False,
        key="code_editor",
    )

    update_code_input()

    # Evaluation button and results outside the container
    if st.button("Evaluate Code", key="custom_apply", type="primary"):
        if st.session_state.temp_code_input and st.session_state.question_generated:
            with st.spinner("Evaluating code..."):
                evaluation_input = {
                    "code": st.session_state.temp_code_input,
                    "practice_question": st.session_state.current_practice_question[
                        "question"
                    ],
                }
                st.session_state.code_evaluation = code_evaluation_chain.invoke(
                    evaluation_input
                )
            st.subheader("Evaluation Result")
            st.write(st.session_state.code_evaluation)
        elif not st.session_state.question_generated:
            st.warning("Please generate a practice question first.")
        else:
            st.warning("Please enter some code to evaluate.")
elif page == "Solution Assistant":
    st.header("Solution Assistant")

    # Display current practice question
    if st.session_state.question_generated:
        st.subheader("Current Practice Question")
        st.write(st.session_state.current_practice_question["question"])
    else:
        st.warning(
            "No practice question generated yet. Please go to the Practice Questions page to generate a question first."
        )

    chat_container = st.container()

    # Initialize hint chat history if it doesn't exist
    if "hint_messages" not in st.session_state:
        st.session_state.hint_messages = []

    # Chat input
    hint_prompt = st.chat_input("Ask for a hint", key="hint_input")

    if hint_prompt:
        # Append to chat history
        st.session_state.hint_messages.append({"role": "user", "content": hint_prompt})

        if st.session_state.question_generated:
            hint = generate_hint(
                hint_prompt,
                st.session_state.current_practice_question["question"],
                st.session_state.code_input if st.session_state.code_input else None,
            )
            st.session_state.hint_messages.append(
                {"role": "assistant", "content": hint}
            )
        else:
            error_message = "Please make sure you have a practice question generated before asking for a hint."
            st.session_state.hint_messages.append(
                {"role": "assistant", "content": error_message}
            )

    # Display chat history in the container
    with chat_container:
        for message in st.session_state.hint_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

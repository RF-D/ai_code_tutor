from sys import stdout

# Langchain imports
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOllama
from langchain_mistralai.chat_models import ChatMistralAI


# Local imports for Ollama
import os
import subprocess
import time
import atexit
import psutil


# Streamlit is needed for type hinting and accessing session state
import streamlit as st


class LLMManager:
    provider_models = {
        "Anthropic": [
            "claude-3-haiku-20240307",
            "claude-3-sonnet-20240229",
            "claude-3-5-sonnet-20240620",
            "claude-3-opus-20240229",
        ],
        "OpenAI": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        "Groq": [
            "llama-3.1-70b-versatile",
            "llama-3.1-405b-reasoning",
            "mixtral-8x7b-32768",
            "llama3-groq-70b-8192-tool-use-preview",
        ],
        "Mistral": [
            "mistral-large-latest",
            "open-mixtral-8x22b",
            "codestral-latest",
            "open-codestral-mamba",
            "open-mistral-nemo",
        ],
        "Ollama": [],
    }
    ollama_process = None

    @staticmethod
    def start_ollama_server():
        # Check if Ollama is already running
        for proc in psutil.process_iter(["name"]):
            if proc.info["name"] == "ollama":
                print("Ollama is already running.")
                return

        print("Starting Ollama server...")
        try:
            # Start Ollama server
            process = subprocess.Popen(
                ["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            # Wait for the server to start (adjust the sleep time if needed)
            time.sleep(5)

            # Check if the process is running
            if process.poll() is None:
                print("Ollama server started successfully.")
            else:
                print("Failed to start Ollama server.")
                stdout, stderr = process.communicate()
                print(f"Error: {stderr.decode()}")
        except FileNotFoundError:
            print(
                "Ollama executable not found. Make sure Ollama is installed and in your PATH."
            )
        except Exception as e:
            print(f"An error occurred while starting Ollama: {str(e)}")

    @staticmethod
    def stop_ollama_server():
        for proc in psutil.process_iter(["name"]):
            if proc.info["name"] == "ollama":
                proc.terminate()
                proc.wait()
                print("Ollama server stopped.")
                return
        print("Ollama server was not running.")

    @staticmethod
    def initialize_ollama_models():
        LLMManager.start_ollama_server()
        LLMManager.provider_models["Ollama"] = LLMManager.get_installed_ollama_models()

    @staticmethod
    def get_installed_ollama_models():
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            if result.returncode == 0:
                models = []
                for line in result.stdout.split("\n")[1:]:  # Skip the header line
                    if line.strip():
                        model_name = line.split()[0]
                        models.append(model_name)
                return models
            else:
                print(f"Error running 'ollama list': {result.stderr}")
                return []
        except FileNotFoundError:
            print(
                "Ollama command not found. Make sure Ollama is installed and in your PATH."
            )
            return []

    @staticmethod
    @st.cache_resource
    def load_llm(provider: str, model: str):
        if provider == "Ollama":
            LLMManager.initialize_ollama_models()  # Refresh Ollama models before loading

        providers = {
            "Anthropic": lambda: ChatAnthropic(
                model=model, temperature=0.8, streaming=True
            ),
            "OpenAI": lambda: ChatOpenAI(model=model, temperature=0.7),
            "Groq": lambda: ChatGroq(
                model_name=model, temperature=0.9, max_tokens=2048
            ),
            "Ollama": lambda: ChatOllama(
                model=model,
                temperature=0.8,
                presence_penalty=0.2,
                n_sentence_context=2,
                streaming=True,
            ),
            "Mistral": lambda: ChatMistralAI(
                model=model, temperature=1, streaming=True
            ),
        }
        if provider not in providers:
            raise ValueError(f"Unsupported provider: {provider}")
        if model not in LLMManager.provider_models[provider]:
            raise ValueError(f"Unsupported model {model} for provider {provider}")
        return providers[provider]()

    @staticmethod
    def get_provider_models():
        return LLMManager.provider_models

    @staticmethod
    def get_models_for_provider(provider: str):
        return LLMManager.provider_models.get(provider, [])

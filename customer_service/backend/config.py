OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL_NAME = "phi3:mini"  # The name of your local model in Ollama

ollama_config_list = [
    {
        "model": OLLAMA_MODEL_NAME,
        "base_url": OLLAMA_BASE_URL,
        "api_type": "ollama",
    }
]


openai_config_list = [
    {
        "model": "gpt-3.5-turbo",  # or "gpt-4o"
        "api_type": "openai",
        "base_url": "https://api.openai.com/v1",  # optional, defaults to OpenAI
    }
]

llm_config = {
    "config_list":  openai_config_list  ,
    "temperature": 0.7,
}

# config.py
AUTH_SERVICE_URL = "http://localhost:8001"
LOGIN_HOST = "http://localhost:4201/"

AUTH_PROFILE_ENDPOINT = f"{AUTH_SERVICE_URL}/profiles/me"
AUTH_LOGIN_URL = f"{LOGIN_HOST}/login" # Or whatever your login endpoint is
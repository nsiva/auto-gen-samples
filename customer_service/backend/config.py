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


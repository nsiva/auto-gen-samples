from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from autogen import AssistantAgent, UserProxyAgent
import os
import requests
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL_NAME = "phi3:mini"  # The name of your local model in Ollama

# CORS for frontend on localhost:4200
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

config_list = [
    {
        "model": OLLAMA_MODEL_NAME,
        "base_url": OLLAMA_BASE_URL,
        "api_type": "ollama",
    }
]

# Models
class ZipRequest(BaseModel):
    zip_code: str

# Helper: Get weather from OpenWeatherMap
def get_weather(zip_code: str) -> str:
    key = os.getenv("OPENWEATHER_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?zip={zip_code},us&appid={key}&units=imperial"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error fetching weather data: {response.status_code} - {response.text}")
        raise HTTPException(status_code=400, detail="Invalid zip code or weather API error")
    description = response.json()["weather"][0]["description"]
    return description

# Route
@app.post("/should_turn_on_sprinkler", response_model=dict)
def should_turn_on_sprinkler(zip: ZipRequest):
    weather = get_weather(zip.zip_code)

    assistant = AssistantAgent(
        name="SprinklerAdvisor",
        system_message="You're a helpful assistant. If the weather includes rain, drizzle, or storm, suggest not turning on sprinkler.",
        llm_config={
            "config_list": config_list,
            "temperature": 0.7, # Adjust as needed
        }
    )

    user = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
    )

    # AutoGen chat
    response = user.initiate_chat(
        assistant,
        message=f"The weather is: {weather}. I am going out, should I turn on sprinkler?"
    )

    return {
        "weather": weather,
        "advice": response.summary if response.summary else "Agent finished without a summary."
    }

@app.post("/should_take_umbrella", response_model=dict)
def should_take_umbrella(zip: ZipRequest):
    weather = get_weather(zip.zip_code)

    assistant = AssistantAgent(
        name="UmbrellaAdvisor",
        #system_message="You're a weater advisor to answer straight way to answer whether to take umbrella or not. If the weather includes rain, drizzle, or storm, simply answer 'to take umbrella' or 'not to take umbrella'.",
        system_message="You are a weather advisor. Based on the weather conditions, respond strictly with 'Take umbrella' if the weather includes rain, drizzle, or storm, or 'No need to take umbrella' otherwise.",
        llm_config={
            "config_list": config_list,
            "temperature": 0.7, # Adjust as needed
        }
    )

    user = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
    )

    # AutoGen chat
    response = user.initiate_chat(
        assistant,
        #message=f"The weather is: {weather}. I am going out, answer with 'to take umbrella' or 'not to take umbrella'.",
        message=f"The weather is: {weather}. I am going out, answer with 'Take umbrella' or 'No need to take umbrella'.",
        max_turns=1
    )

    return {
        "weather": weather,
        "advice": response.summary if response.summary else "Agent finished without a summary."
    }

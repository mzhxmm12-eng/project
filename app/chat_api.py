import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import DefaultHttpxClient, OpenAI
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",
        "null",
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_origin_regex=".*",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """You are an expert Urban Informatics assistant for the Hong Kong Urban Renewal Index Calculator.
You specialize in:

1. Urban Organism Theory - cities as living organisms:
   - Skeleton: building stock density, age, structural framework
   - Metabolism: POI accessibility (supermarkets, green space, services)
   - Circulatory System: road network density, transit coverage

2. Hong Kong Urban Renewal:
   - URA (Urban Renewal Authority) methodology and key projects
   - "Dual aging" challenge: aging population + aging buildings (50yr+)
   - High-density compact city characteristics unique to HK
   - Priority districts: Sham Shui Po, Kowloon City, Wan Chai, Yau Ma Tei

3. Interpreting the renewal priority scores:
   - Score 0.67-1.0 -> High priority: weak urban health, urgent renewal
   - Score 0.33-0.67 -> Medium: moderate intervention recommended
   - Score 0.0-0.33 -> Low: healthy urban fabric

4. Academic methodology: min-max normalisation, weighted composite index,
   AHP weight assignment, spatial analysis with GeoPandas.

Respond in the same language the user uses (Chinese or English).
Be concise (under 250 words) unless asked for more detail.
For Chinese queries, respond in fluent Mandarin Chinese.
"""

history = []


class ChatRequest(BaseModel):
    message: str


@app.post("/api/chat")
async def chat(req: ChatRequest):
    global history
    try:
        client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
            base_url="https://api.deepseek.com",
            http_client=DefaultHttpxClient(trust_env=False),
        )
        history.append({"role": "user", "content": req.message})

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
            max_tokens=800,
        )
        reply = response.choices[0].message.content
        history.append({"role": "assistant", "content": reply})
        return {"reply": reply}
    except Exception as e:
        return {"reply": f"Error: {str(e)}"}


@app.delete("/api/chat")
async def clear_chat():
    global history
    history = []
    return {"status": "cleared"}

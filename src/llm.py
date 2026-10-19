import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
MODEL = 'gemini-2.5-flash'  # model names change; check AI Studio for the current one

def ask_llm(prompt: str) -> str:
    resp = client.models.generate_content(model=MODEL, contents=prompt)
    return resp.text

def ask_with_image(png_bytes: bytes, instruction: str) -> str:
    part = types.Part.from_bytes(data=png_bytes, mime_type='image/png')
    resp = client.models.generate_content(model=MODEL, contents=[part, instruction])
    return resp.text
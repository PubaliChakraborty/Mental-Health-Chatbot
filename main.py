from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import base64
import requests
import io
from PIL import Image
from dotenv import load_dotenv
import os
import logging
from typing import Optional
import pathlib
from langdetect import detect
from deep_translator import GoogleTranslator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(title="Mental Health Image Analysis Assistant")

# Initialize templates
templates_dir = pathlib.Path(__file__).parent / "templates"
templates_dir.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(templates_dir))

# API configuration
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CRISIS_HOTLINE = os.getenv("CRISIS_HOTLINE", "988")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the .env file")

# Mental health prompts in English and Hindi
MENTAL_HEALTH_PROMPTS = {
    "en": [
        "Analyze facial expressions and suggest mood states",
        "Signs of depression or anxiety?",
        "Suggest mental health resources",
        "Analyze environment for risk factors",
        "I'm feeling stressed, what can I do?",
        "How to improve mental wellbeing?"
    ],
    "hi": [
        "चेहरे के भावों का विश्लेषण कर मनोदशा बताएं",
        "अवसाद या चिंता के लक्षण?",
        "मानसिक स्वास्थ्य संसाधन सुझाएं",
        "जोखिम कारकों के लिए वातावरण विश्लेषण",
        "मैं तनाव महसूस कर रहा हूँ, क्या करूँ?",
        "मानसिक स्वास्थ्य कैसे सुधारें?"
    ]
}

def translate_text(text, dest_lang='en'):
    try:
        if dest_lang == 'hi':
            translated = GoogleTranslator(source='auto', target='hi').translate(text)
            term_map = {
                "depression": "अवसाद", "anxiety": "चिंता", "stress": "तनाव",
                "mental health": "मानसिक स्वास्थ्य", "analysis": "विश्लेषण",
                "suggestion": "सुझाव", "crisis": "संकट", "emergency": "आपातकाल"
            }
            for eng, hin in term_map.items():
                translated = translated.replace(eng, hin)
            return translated
        return text
    except Exception as e:
        logger.error(f"Translation failed: {str(e)}")
        return text

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "prompts": MENTAL_HEALTH_PROMPTS["en"]
    })

@app.get("/prompts", response_class=JSONResponse)
async def get_prompts(request: Request):
    lang = request.query_params.get("lang", "en")
    return {"prompts": MENTAL_HEALTH_PROMPTS.get(lang, MENTAL_HEALTH_PROMPTS["en"])}

@app.post("/upload_and_query")
async def upload_and_query(
    image: Optional[UploadFile] = File(None), 
    query: str = Form(...)
):
    try:
        # Detect and process language
        input_lang = detect(query)
        processed_query = translate_text(query, 'en') if input_lang != 'en' else query
        
        # Prepare AI message
        messages = [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"""As a mental health professional{" analyzing this image" if image else ""}, provide:
                1. Professional analysis
                2. Possible mental health considerations  
                3. Supportive suggestions
                4. Crisis resources if needed
                
                Query: {processed_query}
                
                Important: Never diagnose, only suggest possibilities."""
            }]
        }]

        # Add image if provided
        if image:
            try:
                image_content = await image.read()
                img = Image.open(io.BytesIO(image_content))
                img.verify()
                encoded_image = base64.b64encode(image_content).decode("utf-8")
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
                })
            except Exception as e:
                logger.error(f"Invalid image: {str(e)}")
                raise HTTPException(400, "Invalid image format")

        # Call AI API
        response = requests.post(
            GROQ_API_URL,
            json={
                "model": "meta-llama/llama-4-maverick-17b-128e-instruct",
                "messages": messages,
                "max_tokens": 1500,
                "temperature": 0.7
            },
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            timeout=45
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]

        # Translate response if needed
        if input_lang == 'hi':
            content = translate_text(content, 'hi')
            crisis_alert = "🚨 आपात स्थिति में कृपया 988 या आपातकालीन सेवाओं को कॉल करें"
        else:
            crisis_alert = f"🚨 In crisis, call {CRISIS_HOTLINE} or emergency services"

        # Add crisis alert if needed
        if any(word in content.lower() for word in ["crisis", "emergency", "suicid", "self-harm"]):
            content += f"\n\n{crisis_alert}"

        return JSONResponse(content={
            "analysis": content,
            "crisis_resources": {
                "hotline": CRISIS_HOTLINE,
                "text": "Text HOME to 741741",
                "international": "https://www.befrienders.org"
            },
            "input_lang": input_lang,
            "has_image": bool(image)
        })

    except requests.RequestException as e:
        logger.error(f"API error: {str(e)}")
        raise HTTPException(500, "AI service unavailable")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(500, "Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
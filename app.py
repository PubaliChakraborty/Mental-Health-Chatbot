import base64
from groq import Groq
import io
from PIL import Image
from dotenv import load_dotenv
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CRISIS_HOTLINE = os.getenv("CRISIS_HOTLINE", "988")  # Default to US hotline

# Ensure API key is set
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the .env file")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

def analyze_mental_health(image_path, query):
    try:
        # Read and validate image
        with open(image_path, "rb") as image_file:
            image_content = image_file.read()

        try:
            img = Image.open(io.BytesIO(image_content))
            img.verify()
        except Exception as e:
            logger.error(f"Invalid image format: {str(e)}")
            return {"analysis": None, "error": "Invalid image format"}

        # Convert to base64
        encoded_image = base64.b64encode(image_content).decode("utf-8")

        # Mental health-focused prompt enhancement
        enhanced_query = f"""
        As a mental health professional analyzing this image, consider:
        1. Mood indicators (facial expressions/posture)
        2. Environmental context
        3. Potential stress/anxiety cues

        User Query: {query}

        Provide:
        - Observations (non-diagnostic)
        - Supportive suggestions
        - Resources if concerning cues appear

        Important: Never diagnose. Use phrases like "may suggest" or "could indicate".
        """

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": enhanced_query},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                ]
            }
        ]

        def get_model_response(model):
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=1500,
                    temperature=0.7
                )
                content = completion.choices[0].message.content

                # Append crisis information if risk factors are detected
                if any(term in content.lower() for term in ["crisis", "suicid", "self-harm", "emergency"]):
                    content += f"\n\nCRISIS ALERT: Contact {CRISIS_HOTLINE} or local emergency services if immediate help is needed."

                return content
            except Exception as e:
                logger.error(f"API request failed: {str(e)}")
                return {"error": f"API request failed: {str(e)}"}

        analysis_result = get_model_response("meta-llama/llama-4-maverick-17b-128e-instruct")

        if isinstance(analysis_result, dict) and "error" in analysis_result:
            return {"analysis": None, "error": analysis_result["error"]}

        return {
            "analysis": analysis_result,
            "resources": {
                "hotline": CRISIS_HOTLINE,
                "text_line": "Text HOME to 741741 (US/CAN)",
                "international": "https://www.iasp.info/resources/Crisis_Centres/"
            }
        }

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {"analysis": None, "error": "Unexpected error occurred"}

if __name__ == "__main__":
    try:
        # Example usage
        result = analyze_mental_health(
            image_path="test_face.jpg",  # Replace with the path to your test image
            query="What mood might this person be experiencing?"
        )

        if result.get("analysis"):
            print("=== Mental Health Analysis ===")
            print(result["analysis"])
        else:
            print("Analysis could not be completed.")
            print(f"Error: {result.get('error', 'Unknown error')}")

        print("\n=== Resources ===")
        resources = result.get("resources", {})
        print(f"Crisis Hotline: {resources.get('hotline', 'N/A')}")
        print(f"Text Line: {resources.get('text_line', 'N/A')}")
        print(f"International: {resources.get('international', 'N/A')}")

    except Exception as main_exception:
        logger.error(f"An error occurred in the main script: {str(main_exception)}")
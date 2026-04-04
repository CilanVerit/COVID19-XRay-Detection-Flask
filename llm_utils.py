from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

def generate_diagnosis(prediction, probabilities):
    prompt = f"""
    You are a medical AI assistant.

    Model prediction:
    Class: {prediction}
    Probabilities: {probabilities}

    Explain the result for a non-medical user in 2–3 sentences.

    Use simple language.
    You may use **bold formatting** for important terms.
    Do not make definitive medical diagnoses.
    """

    try:
        response = client.chat.completions.create(
            model="deepseek/deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful medical AI assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=120,
            extra_headers={
                "HTTP-Referer": "http://localhost:5000",
                "X-Title": "Chest X-ray AI"
            }
        )

        return response.choices[0].message.content

    except Exception as e:
        print("LLM ERROR:", e)
        return f"Prediction: {prediction}. AI explanation unavailable."
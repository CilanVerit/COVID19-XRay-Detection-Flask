from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

def generate_diagnosis(prediction, probabilities, confidence):
    # Format probabilities nicely
    probs_text = "\n".join([
        f"- {cls}: {prob*100:.1f}%"
        for cls, prob in probabilities.items()
    ])

    # Add confidence note
    if confidence < 0.7:
        confidence_note = "The model is **not very confident**, so this result is uncertain."
    elif confidence < 0.9:
        confidence_note = "The model is **moderately confident** in this result."
    else:
        confidence_note = "The model is **highly confident**, but this is still not a medical diagnosis."

    prompt = f"""
    You are a careful medical AI assistant.

    A chest X-ray model produced the following result:

    Prediction: {prediction}
    Confidence: {confidence*100:.1f}%

    Class probabilities:
    {probs_text}

    Explain this in 2–3 simple sentences for a non-medical user.

    Rules:
    - Use simple, clear language
    - Mention uncertainty appropriately based on confidence
    - Do NOT make definitive medical claims
    - Encourage professional medical consultation if needed
    - You may use **bold** for key terms
    - Avoid phrases like "this confirms" or "you have" or similar definitive diagnosis

    IMPORTANT:
    - Use the EXACT confidence value provided ({confidence*100:.2f}%)
    - Do NOT round to 100%
    - Do NOT exaggerate certainty
    """

    try:
        response = client.chat.completions.create(
            model="deepseek/deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a cautious and clear medical AI assistant."},
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
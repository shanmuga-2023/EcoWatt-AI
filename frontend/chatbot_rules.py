def eco_reply(question: str, data: dict) -> str:
    q = question.lower()
    if "carbon" in q:
        return f"Your current CO₂ savings are about {round(data.get('co2_saved',0),2)} kg. Every kWh of solar helps!"
    if "reduce" in q or "save" in q:
        return "Try scheduling heavy appliances during daylight to use solar energy and cut peak grid load."
    if "renewable" in q:
        return f"Renewable share is {data.get('renewable_ratio',0)}%. Aim for above 70% for optimal sustainability."
    if "hello" in q or "hi" in q:
        return "Hello 🌿! I’m EcoBot — your AI energy assistant."
    return "I'm still learning! Try asking about carbon savings, renewable usage, or energy tips."

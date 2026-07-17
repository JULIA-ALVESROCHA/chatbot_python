# tests/run_chatbot_test.py

def run_chatbot(prompt: str) -> str:
    """
    Wrapper de teste.
    NÃO importa pipeline real.
    NÃO usa asyncio.
    """

    # mock mínimo enquanto roda regressão
    MOCK_KNOWLEDGE = {
        "what is your name": "I am the BGO assistant.",
        "who is the ceo": "I don’t have information about that.",
    }

    prompt_l = prompt.lower()

    for k, v in MOCK_KNOWLEDGE.items():
        if k in prompt_l:
            return v

    # resposta default segura (anti-hallucination)
    return "I’m not sure about that based on the provided information."

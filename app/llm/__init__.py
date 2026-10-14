

def fresh(llm) -> dict:
    """Argumenty dla complete(), gdy odpowiedz ma byc za kazdym razem nowa (bez pamieci podrecznej LLM) -
    np. nowy post na fanpage. Atrapy LLM w testach nie maja parametru cache - wtedy pusto."""
    import inspect
    try:
        return {"cache": False} if "cache" in inspect.signature(llm.complete).parameters else {}
    except (TypeError, ValueError, AttributeError):
        return {}

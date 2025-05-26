from typing import Any, Dict
from ollamachat import OllamaChat


def __prepare_llm__(
    model: str, 
    temperature: float, 
    structured_response: Dict[str, Any])->OllamaChat:
    if "ollama:" in model:
        if structured_response is None:
            return OllamaChat(
                language_model=model[7:],
                temperature=temperature
            )
        else:
            return OllamaChat(
                language_model=model[7:],
                temperature=temperature,
                structured_response=structured_response,
            )                
    else:
        raise TypeError("__prepare_llm__: chat model name invalid")

from pydantic import BaseModel
from typing import Any, List, Dict
import ollama


'''
this function provides an ollama endpoint to chat with
'''
class OllamaChat(BaseModel):
    language_model:str=None
    temperature:float=0.0
    structured_response: Dict[str, Any] = None

    def invoke(self, history: List[Dict[str, str]])->str:
        if not any(self.language_model in model["model"] for model in ollama.list()["models"]):
            raise TypeError(f"ollama: {self.language_model} model not found") 
        if self.structured_response is None:  
            response = ollama.chat(
                model=self.language_model,
                messages=history,
                options = {
                    'temperature': self.temperature
                },
                stream=False,
            )
        else:
            response = ollama.chat(
                model=self.language_model,
                messages=history,
                options = {
                    'temperature': self.temperature
                },
                stream=False,
                format=self.structured_response,
            )
        return response['message']['content']
    
    def stream(self, history: List[Dict[str, str]]):
        if not any(self.language_model in model["model"] for model in ollama.list()["models"]):
            raise TypeError(f"ollama: {self.language_model} model not found")       
        return ollama.chat(
                model=self.language_model,
                messages=history,
                options = {
                    'temperature': self.temperature
                },
                stream=True,
            )
        
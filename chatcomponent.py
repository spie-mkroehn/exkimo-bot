from pydantic import BaseModel
from typing import Any, Dict, List
from ollamachat import OllamaChat
from componentresultobject import ComponentResultObject
import commons


'''
This component performs an llm operation given the complete chat history
(In this way it is possible to not only do conversation with user but also analyse chats)
'''
class ChatComponent(BaseModel):
    llm: OllamaChat = None
    language_model: str = "gemma3:4b"
    temperature: float = 0.0
    structured_response: Dict[str, Any] = None

    #input contains whole chat including initial system prompt
    def invoke(self, input:List[ComponentResultObject])->List[ComponentResultObject]:
        if self.llm is None:
            self.llm = commons.__prepare_llm__(
                    model=self.language_model,
                    temperature=self.temperature,
                    structured_response=self.structured_response
                )
        history = self.__prepare_messages__(input)
        answer = ComponentResultObject()
        answer["source"] = "assistant"
        answer["content"]["original_text"] = self.llm.invoke(history)
        input.append(answer) 
        return input
    
    #input contains whole chat including initial system prompt
    def stream(self, input:List[ComponentResultObject]):
        if self.llm is None:
            self.llm = commons.__prepare_llm__(
                    model=self.language_model,
                    temperature=self.temperature,
                    structured_response=self.structured_response
                )
        history = self.__prepare_messages__(input)
        return self.llm.stream(history)
    
    def __prepare_messages__(self, msgs:List[ComponentResultObject])->List[Dict[str, str]]:
        history = []
        for item in msgs:
            history.append(
                {
                    "role": item["source"],
                    "content": item["content"]["original_text"]
                }
            )
        return history

from pydantic import BaseModel, Field
from typing import ClassVar, Dict, Any


class ComponentResultObject(BaseModel):
    cro_version: ClassVar[str] = "APOFIS v1"
    dictionary: Dict[str, Any] = Field(default_factory=lambda: {
        "unique_id": None,
        "source": None,
        "content": {
            "original_text": None,
            "chapter": None,
            "page_number": None,
            "page_count": None,
            "raw_image_data": None,
            "raw_audio_data": None,
            "sampling_rate": None,
        },           
        "preprocessing": {
            "result_text": None,
            "summary": None,
            "keywords": None,
            "category": None,
            "embedding": {
                "standard": None,
                "enhanced": None,
            },
            "questions": None,
            "score": None,
        },
        "retrieval": {
            "rank": None,
            "distance": None,
        },
        "target": None,
        "version": ComponentResultObject.cro_version
    })

    def __setitem__(self, key:str, val:str)->None:
        #if key not in self.dictionary:
        #    raise KeyError()
        self.dictionary[key] = val

    def __getitem__(self, key:str)->str:
        #if key not in self.dictionary:
        #    raise KeyError()       
        return self.dictionary[key]
    
    def update(self, dict:dict[str,str])->None:
        for key in dict:
            self.__setitem__(key, dict[key])

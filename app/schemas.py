from pydantic import BaseModel
from typing import List

class AdInput(BaseModel):
    """
    Схема тела запроса (JSON)
    """
    itemId: int
    mcId: int
    mcTitle: str
    description: str

class Draft(BaseModel):
    """
    В таком формате он предлагает нам черновик (JSON)
    mcId: int - id объявления (предлагает сам)
    mcTitle: str - название предложенного объявления
    text: str - описание объявления
    """
    mcId: int
    mcTitle: str
    text: str

class ProcessingResponse(BaseModel):
    """
    Схема ответа от модели (status 200)
    
    detectedMcIds: List[int] - какие дополнительные объявления он заметил (список id)
    shouldSplit: bool - нужно ли разделять объявление на разные категории (true/false)
    drafts: List[Draft] = [] - список черновиков (из класса Draft)
    """
    detectedMcIds: List[int]
    shouldSplit: bool
    drafts: List[Draft] = []

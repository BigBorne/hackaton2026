from fastapi import FastAPI, Request
from gigachat import GigaChat
import json

from app.prompt import response_prompt
from app.schemas import AdInput, ProcessingResponse

giga = GigaChat(credentials="MDE5ZDVkYTQtZDk4Mi03NDQ5LTkzYjMtMzEwNmFmZDg4MTBmOjhjNmJlNTc2LTI0YTMtNGY0MC1iMWNiLTk2MzhhYzIzOTg2Zg==", verify_ssl_certs=False)
app = FastAPI()

# 1. Эндпоинт /analyze - post запрос на swagger ui (http://127.0.0.1:8000/docs)
@app.post('/analyze')
async def process_add(item: AdInput) -> ProcessingResponse:
    """
        Асинхронная функция:
        response:
            - задаем промпт для модели из response_prompt
            - выводим данные из ответа по шаблону из schemas.AdInput (В json формате)
    """
    response = giga.chat({
        "messages": [
            {"role": "system", "content": response_prompt},
            {"role": "user", "content": item.description}
        ],
        "model": "GigaChat-Pro",  # конкретная модель, например GigaChat-Pro
        "format": "json"  # Просим GigaChat вернуть именно JSON
    })

    # 2. Извлекаем контент
    content = response.choices[0].message.content
    result_dict = json.loads(content)

    # 3. Возвращаем результат
    return ProcessingResponse(**result_dict)

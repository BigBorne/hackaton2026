from fastapi import FastAPI, Request
from gigachat import GigaChat
import json

from app.prompt import response_prompt
from app.schemas import AdInput, ProcessingResponse

giga = GigaChat(credentials="MDE5ZDQ0YTctY2ZlYi03NjhmLTg5YzgtNDcxZGIyNDk1NmE0OjY4OWM0ZmZjLTViMjMtNDcyNS05NzNiLWY0YjlhMjZiMGFkNA==", verify_ssl_certs=False)
app = FastAPI()



@app.post('/analyze')
async def process_add(item: AdInput) -> ProcessingResponse:
    response = giga.chat({
        "messages": [
            {"role": "system", "content": response_prompt},
            {"role": "user", "content": item.description}
        ],
        "model": "GigaChat-Pro",  # Или конкретная модель, например GigaChat-Pro
        "format": "json"  # Просим GigaChat вернуть именно JSON
    })

    # 2. Извлекаем контент
    content = response.choices[0].message.content
    result_dict = json.loads(content)

    # 3. Возвращаем результат
    return ProcessingResponse(**result_dict)

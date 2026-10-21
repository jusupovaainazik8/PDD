import asyncio
import base64
from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import uvicorn
import os

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL = os.getenv('GEMINI_MODEL')


class PddOutputSchema(BaseModel):
    class_id: int
    class_name: str
    category: str
    description: str


pdd_parser = PydanticOutputParser(pydantic_object=PddOutputSchema)

pdd_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Ты эксперт по правилам дорожного движения. "
            "Определи дорожный знак на фотографии. "
            "class_id - номер класса знака в наборе данных GTSRB (число от 0 до 42), например: 14 для знака Stop. "
            "class_name - название знака на английском языке, например: Stop. "
            "category - категория знака на русском языке: Запрещающий знак, Предупреждающий знак, "
            "Предписывающий знак, Знак приоритета или Знак отмены. "
            "description - краткое описание знака на русском языке в одном предложении, например: Остановка обязательна. "
            "Если на фотографии нет дорожного знака, верни class_id равный -1, "
            "а в class_name, category и description поставь прочерк. "
            "Не придумывай того, чего нет на фотографии. "
            "{format_instructions}",
        ),
        (
            "human",
            [
                {"type": "text", "text": "Определи дорожный знак на фотографии."},
                {"type": "image_url", "image_url": {"url": "data:{mime};base64,{image}"}},
            ],
        ),
    ]
).partial(format_instructions=pdd_parser.get_format_instructions())

pdd_model = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=GEMINI_API_KEY,
    temperature=0,
)

pdd_chain = pdd_prompt | pdd_model | pdd_parser

pdd_router = APIRouter(prefix='/pdd', tags=['ПДД'])


@pdd_router.post(path='/predict/', response_model=PddOutputSchema)
async def check_image_pdd(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=422, detail='Маалымат туура эмес форматта берилип жатат')

    content = await file.read()

    if not content:
        raise HTTPException(status_code=422, detail='Маалымат туура эмес форматта берилип жатат')

    image = base64.b64encode(content).decode()

    try:
        result = await asyncio.wait_for(
            pdd_chain.ainvoke({"mime": file.content_type, "image": image}), timeout=120
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=500, detail='Модель тапшырманы аткарганы жетишкен жок')

    except OutputParserException:
        raise HTTPException(status_code=500, detail='Моделдин жообу схемага туура келбейт')

    if result.class_id == -1:
        raise HTTPException(status_code=422, detail='Сүрөттөн жол белгиси табылган жок')

    return result


pdd_app = FastAPI(title='PDD Sign Recognition')

pdd_app.include_router(pdd_router)


if __name__ == '__main__':
    uvicorn.run(pdd_app, host='127.0.0.1', port=8000)
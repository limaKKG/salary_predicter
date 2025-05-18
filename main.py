import uvicorn
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import joblib
from pydantic import BaseModel, Field
from typing import Literal, List, Dict, Any
import httpx 
import os 
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Salary Predictor API & Career Chat", version="1.1", description="API для предсказания зарплаты и чат с LLM о карьере")

templates = Jinja2Templates(directory="templates")

try:
    model = joblib.load('salary_model.joblib')
except FileNotFoundError:
    print("Ошибка: Файл 'salary_model.joblib' не найден. Поместите его в корень проекта.")
    model = None

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL = "qwen/qwen3-30b-a3b:free" 

CAREER_SYSTEM_PROMPT = """
Ты — опытный карьерный консультант. Твоя задача — помогать пользователям исследовать карьерные возможности,
давать советы по развитию навыков, составлению резюме, прохождению собеседований и другим вопросам,
связанным с карьерой. Будь дружелюбным, поддерживающим и предоставляй конструктивную информацию. Не пиши слишком длинные сообщения.
Отвечай на русском языке.
"""

EDUCATION_VALUES = ['High School', 'Bachelor', 'PhD', 'Master']
LOCATION_VALUES = ['Suburban', 'Rural', 'Urban']
JOB_TITLE_VALUES = ['Director', 'Analyst', 'Manager', 'Engineer']
GENDER_VALUES = ['Male', 'Female']

class SalaryInput(BaseModel):
    Education: Literal['High School', 'Bachelor', 'PhD', 'Master']
    Experience: int = Field(..., gt=0, description="Опыт работы в годах, должен быть больше 0")
    Location: Literal['Suburban', 'Rural', 'Urban']
    Job_Title: Literal['Director', 'Analyst', 'Manager', 'Engineer']
    Age: int = Field(..., gt=17, lt=100, description="Возраст, должен быть в разумных пределах (18-99)")
    Gender: Literal['Male', 'Female']

class PredictionResponse(BaseModel):
    predicted_salary: float

class ChatMessageInput(BaseModel):
    message: str
    history: List[Dict[str, str]] = [] 

class ChatMessageResponse(BaseModel):
    reply: str

client = httpx.AsyncClient()

@app.on_event("shutdown")
async def app_shutdown():
    await client.aclose()


@app.get("/", response_class=HTMLResponse, summary="Главная страница с формой ввода и чатом")
async def get_main_page(request: Request):
    """
    Отображает HTML-форму для ввода данных предсказания зарплаты и интерфейс чата.
    """
    return templates.TemplateResponse("index.html", {
        "request": request,
        "education_values": EDUCATION_VALUES,
        "location_values": LOCATION_VALUES,
        "job_title_values": JOB_TITLE_VALUES,
        "gender_values": GENDER_VALUES,
        "prediction": None,
        "input_data": {}
    })

@app.post("/", response_class=HTMLResponse, summary="Предсказать зарплату на основе введенных данных")
async def predict_salary_form(
    request: Request,
    Education: str = Form(...),
    Experience: int = Form(...),
    Location: str = Form(...),
    Job_Title: str = Form(...),
    Age: int = Form(...),
    Gender: str = Form(...)
):
    """
    Принимает данные из формы, выполняет предсказание и отображает результат на той же странице.
    """
    if not model:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "education_values": EDUCATION_VALUES,
            "location_values": LOCATION_VALUES,
            "job_title_values": JOB_TITLE_VALUES,
            "gender_values": GENDER_VALUES,
            "error_message": "Модель предсказания зарплаты не загружена. Проверьте консоль сервера.",
            "prediction": None,
            "input_data": {},
            "chat_active": True
        })

    input_data_dict = {
        "Education": Education,
        "Experience": Experience,
        "Location": Location,
        "Job_Title": Job_Title,
        "Age": Age,
        "Gender": Gender
    }

    try:
        salary_input_validated = SalaryInput(**input_data_dict)
    except ValueError as e:
         return templates.TemplateResponse("index.html", {
            "request": request,
            "education_values": EDUCATION_VALUES,
            "location_values": LOCATION_VALUES,
            "job_title_values": JOB_TITLE_VALUES,
            "gender_values": GENDER_VALUES,
            "error_message": f"Ошибка в данных формы: {e}",
            "prediction": None,
            "input_data": input_data_dict,
            "chat_active": True
        })

    df_input = pd.DataFrame([salary_input_validated.model_dump()])
    
    try:
        prediction_val = model.predict(df_input)[0]
        formatted_prediction = f"{prediction_val:,.2f}"
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "education_values": EDUCATION_VALUES,
            "location_values": LOCATION_VALUES,
            "job_title_values": JOB_TITLE_VALUES,
            "gender_values": GENDER_VALUES,
            "error_message": f"Ошибка при предсказании зарплаты: {e}",
            "prediction": None,
            "input_data": salary_input_validated.model_dump(),
            "chat_active": True
        })

    return templates.TemplateResponse("index.html", {
        "request": request,
        "education_values": EDUCATION_VALUES,
        "location_values": LOCATION_VALUES,
        "job_title_values": JOB_TITLE_VALUES,
        "gender_values": GENDER_VALUES,
        "prediction": formatted_prediction,
        "input_data": salary_input_validated.model_dump(),
        "chat_active": True
    })

@app.post("/predict_api", response_model=PredictionResponse, summary="API эндпоинт для предсказания зарплаты (JSON)") 
async def predict_salary_api(data: SalaryInput):
    """
    Принимает данные в формате JSON, выполняет предсказание и возвращает результат.
    - **Education**: Уровень образования ('High School', 'Bachelor', 'PhD', 'Master')
    - **Experience**: Опыт работы в годах (например, 5)
    - **Location**: Местоположение ('Suburban', 'Rural', 'Urban')
    - **Job_Title**: Должность ('Director', 'Analyst', 'Manager', 'Engineer')
    - **Age**: Возраст (например, 30)
    - **Gender**: Пол ('Male', 'Female')
    """
    if not model:
        raise HTTPException(status_code=503, detail="Модель предсказания зарплаты не загружена.")

    df_input = pd.DataFrame([data.model_dump()])
    prediction = model.predict(df_input)[0]
    return PredictionResponse(predicted_salary=prediction)

@app.post("/chat_llm", response_model=ChatMessageResponse, summary="Эндпоинт для чата с LLM о карьере")
async def chat_with_llm(chat_input: ChatMessageInput):
    if not OPENROUTER_API_KEY:
        # Возвращаем JSONResponse, чтобы ошибка отобразилась в чате на клиенте
        return JSONResponse(
            status_code=500, # Ошибка конфигурации сервера
            content={"reply": "Ошибка: API ключ для OpenRouter не настроен на сервере. Пожалуйста, установите переменную окружения OPENROUTER_API_KEY."}
        )

    messages = [{"role": "system", "content": CAREER_SYSTEM_PROMPT}]
    messages.extend(chat_input.history)
    messages.append({"role": "user", "content": chat_input.message})

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": LLM_MODEL,
        "messages": messages
    }

    try:
        response = await client.post(OPENROUTER_API_URL, json=payload, headers=headers, timeout=30.0)
        response.raise_for_status() 
        
        api_response_data = response.json()
        
        if api_response_data.get("choices") and len(api_response_data["choices"]) > 0:
            llm_reply = api_response_data["choices"][0].get("message", {}).get("content", "")
            if not llm_reply:
                 llm_reply = "Извините, я не смог получить ответ. Попробуйте еще раз."
        else:
            error_info = api_response_data.get("error", {}).get("message", "Неизвестная ошибка от API.")
            print(f"OpenRouter API error details: {api_response_data}") 
            llm_reply = f"Ошибка при обращении к LLM: {error_info}. Пожалуйста, проверьте детали в логах сервера."

    except httpx.RequestError as e:
        print(f"Ошибка запроса к OpenRouter: {e}")
        return ChatMessageResponse(reply=f"Не удалось связаться с сервисом LLM: {e}. Попробуйте позже.")
    except httpx.HTTPStatusError as e:
        print(f"Ошибка HTTP от OpenRouter: {e.response.status_code}, {e.response.text}")
        # Здесь мы уже получаем 401, если ключ неверный, но причина может быть и другая (например, 429 - rate limit)
        # Сообщение уже достаточно информативно для пользователя, детали будут в логах сервера.
        return ChatMessageResponse(reply=f"Сервис LLM вернул ошибку: {e.response.status_code}. Детали в логах сервера.")
    except Exception as e:
        print(f"Непредвиденная ошибка при общении с LLM: {e}")
        return ChatMessageResponse(reply=f"Произошла внутренняя ошибка сервера при обработке вашего запроса к LLM.")

    return ChatMessageResponse(reply=llm_reply)

if __name__ == "__main__":
    if not OPENROUTER_API_KEY:
        print("\n" + "*"*50)
        print("ВНИМАНИЕ: API КЛЮЧ OPENROUTER НЕ УСТАНОВЛЕН!")
        print("Пожалуйста, установите переменную окружения OPENROUTER_API_KEY.")
        print("Чат с LLM не будет работать без ключа.")
        print("*"*50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000) 
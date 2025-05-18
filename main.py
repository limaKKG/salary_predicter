import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pandas as pd
import joblib
from pydantic import BaseModel, Field
from typing import Literal

app = FastAPI(title="Salary Predictor API", version="1.0", description="API для предсказания зарплаты")
templates = Jinja2Templates(directory="templates")
try:
    model = joblib.load('salary_model.joblib')
except FileNotFoundError:
    print("Ошибка: Файл 'salary_model.joblib' не найден. Поместите его в корень проекта.")
    model = None 

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


@app.get("/", response_class=HTMLResponse, summary="Главная страница с формой ввода")
async def get_form(request: Request):
    """
    Отображает HTML-форму для ввода данных и получения предсказания зарплаты.
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
            "error_message": "Модель не загружена. Проверьте консоль сервера.",
            "prediction": None,
            "input_data": {}
        })

    input_data = SalaryInput(
        Education=Education,
        Experience=Experience,
        Location=Location,
        Job_Title=Job_Title,
        Age=Age,
        Gender=Gender
    )
    
    df_input = pd.DataFrame([input_data.model_dump()])
    
    try:
        prediction = model.predict(df_input)[0]
        formatted_prediction = f"{prediction:,.2f}"
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "education_values": EDUCATION_VALUES,
            "location_values": LOCATION_VALUES,
            "job_title_values": JOB_TITLE_VALUES,
            "gender_values": GENDER_VALUES,
            "error_message": f"Ошибка при предсказании: {e}",
            "prediction": None,
            "input_data": input_data.model_dump()
        })

    return templates.TemplateResponse("index.html", {
        "request": request,
        "education_values": EDUCATION_VALUES,
        "location_values": LOCATION_VALUES,
        "job_title_values": JOB_TITLE_VALUES,
        "gender_values": GENDER_VALUES,
        "prediction": formatted_prediction,
        "input_data": input_data.model_dump()
    })

@app.post("/predict", response_model=PredictionResponse, summary="API эндпоинт для предсказания зарплаты")
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
        return PredictionResponse(predicted_salary=-1.0)

    df_input = pd.DataFrame([data.model_dump()])
    prediction = model.predict(df_input)[0]
    return PredictionResponse(predicted_salary=prediction)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 
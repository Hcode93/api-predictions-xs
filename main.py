from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
model = joblib.load('model/modelo_prestamos.joblib')
model_v2 = joblib.load('model/modelo_prestamos_v2.joblib')


class PrestamoInput(BaseModel):
    edad: int
    ingresos_anuales: float
    score_crediticio: int
    deuda_total: float


@app.post("/evaluar-prestamo")
def predict(data: PrestamoInput):
    # Convertimos el input a la lista que espera el modelo
    input_vars = [[data.edad, data.ingresos_anuales, data.score_crediticio, data.deuda_total]]

    prediction = model.predict(input_vars)
    probabilidad = model.predict_proba(input_vars)[0][1]  # Probabilidad de ser "1"

    resultado = "Aprobado" if prediction[0] == 1 else "Rechazado"

    return {
        "status": resultado,
        "score_aprobacion": round(probabilidad * 100, 2),
        "mensaje": f"El crédito fue {resultado.lower()} con un {round(probabilidad * 100)}% de confianza."
    }


@app.post("/v2/evaluar-prestamo")
def predict_v2(data: PrestamoInput):
    # Feature Engineering (crear las 5 nuevas variables al vuelo)
    dti = data.deuda_total / data.ingresos_anuales if data.ingresos_anuales > 0 else 0
    capacidad_ahorro = data.ingresos_anuales - data.deuda_total
    score_normalizado_edad = data.score_crediticio / data.edad if data.edad > 0 else 0
    apalancamiento_critico = 1 if data.deuda_total > (data.ingresos_anuales * 0.5) else 0
    multiplicador_estabilidad = (data.score_crediticio * data.edad) / 100.0

    # Armamos la lista con las 9 variables, en el mismo orden que usamos en el entranamiento
    input_vars_v2 = [[
        data.edad, 
        data.ingresos_anuales, 
        data.score_crediticio, 
        data.deuda_total,
        dti,
        capacidad_ahorro,
        score_normalizado_edad,
        apalancamiento_critico,
        multiplicador_estabilidad
    ]]

    prediction = model_v2.predict(input_vars_v2)
    probabilidad = model_v2.predict_proba(input_vars_v2)[0][1]  # Probabilidad de ser "1"

    resultado = "Aprobado" if prediction[0] == 1 else "Rechazado"

    return {
        "status": resultado,
        "score_aprobacion": round(probabilidad * 100, 2),
        "mensaje": f"El crédito fue {resultado.lower()} con un {round(probabilidad * 100)}% de confianza. (Evaluado con modelo v2)"
    }
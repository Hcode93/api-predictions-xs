from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
model = joblib.load('modelo_prestamos.joblib')


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
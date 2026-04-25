import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def engineer_features(df):
    """
    Applies the 5 new Feature Engineering variables:
    1. Ratio Deuda-Ingresos (DTI)
    2. Capacidad de Ahorro / Solvencia Estimada
    3. Score Crediticio Normalizado por Edad
    4. Nivel de Apalancamiento Crítico (Flag Binario)
    5. Multiplicador de Estabilidad (Interaction Feature)
    """
    # Evitar divisiones por cero en el improbable caso de que haya ceros
    df_engineered = df.copy()
    
    # 1. DTI: deuda_total / ingresos_anuales
    df_engineered['dti'] = df_engineered['deuda_total'] / df_engineered['ingresos_anuales']
    
    # 2. Capacidad de Ahorro: ingresos_anuales - deuda_total
    df_engineered['capacidad_ahorro'] = df_engineered['ingresos_anuales'] - df_engineered['deuda_total']
    
    # 3. Score Normalizado por Edad: score_crediticio / edad
    df_engineered['score_normalizado_edad'] = df_engineered['score_crediticio'] / df_engineered['edad']
    
    # 4. Apalancamiento Crítico: 1 si deuda_total > (ingresos_anuales * 0.5) sino 0
    df_engineered['apalancamiento_critico'] = (df_engineered['deuda_total'] > (df_engineered['ingresos_anuales'] * 0.5)).astype(int)
    
    # 5. Multiplicador de Estabilidad: (score_crediticio * edad) / 100
    df_engineered['multiplicador_estabilidad'] = (df_engineered['score_crediticio'] * df_engineered['edad']) / 100.0

    return df_engineered

if __name__ == "__main__":
    print("Iniciando entrenamiento del modelo V2 con Feature Engineering...")
    
    # 1. Carga de datos
    print("Cargando dataset_prestamos_500.csv...")
    df = pd.read_csv('model/dataset_prestamos_500.csv')
    
    # 2. Preprocesamiento (Feature Engineering)
    print("Aplicando Feature Engineering...")
    df = engineer_features(df)
    
    # Mostrar las nuevas variables creadas
    features_cols = df.drop('aprobado', axis=1).columns.tolist()
    print(f"Total features ({len(features_cols)}): {features_cols}")
    
    # 3. Dividir en entrenamiento y prueba
    X = df.drop('aprobado', axis=1)
    y = df['aprobado']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Entrenar Modelo
    print("\nEntrenando RandomForestClassifier (v2)...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 5. Evaluar Modelo
    y_pred = model.predict(X_test)
    print("\n-------------------- EVALUACION --------------------")
    print("Reporte de Clasificación (v2):")
    print(classification_report(y_test, y_pred))
    
    print("\nMatriz de Confusión (v2):")
    print(confusion_matrix(y_test, y_pred))
    print("----------------------------------------------------")
    
    # Obtener importancia de las variables
    importancias = model.feature_importances_
    df_importancia = pd.DataFrame({'Variable': features_cols, 'Importancia': importancias})
    df_importancia = df_importancia.sort_values(by='Importancia', ascending=False)
    print("\nRanking de Importancia de Variables:")
    print(df_importancia.to_string(index=False))
    
    # 6. Guardar Modelo
    model_path = 'model/modelo_prestamos_v2.joblib'
    joblib.dump(model, model_path)
    print(f"\n✅ Modelo guardado con éxito en: {model_path}")

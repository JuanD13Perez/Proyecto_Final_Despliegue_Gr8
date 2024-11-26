from typing import Any, List, Optional

from pydantic import BaseModel
from model.processing.validation import DataInputSchema

# Esquema de los resultados de predicción
class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[List[float]]

# Esquema para inputs múltiples
class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "age": 70,
                        "anaemia": 1,
                        "creatinine_phosphokinase": 125,
                        "diabetes": 1,
                        "ejection_fraction": 25,
                        "high_blood_pressure": 1,
                        "platelets": 237000.0,
                        "serum_creatinine":1.0,
                        "serum_sodium":140,
                        "sex":1,
                        "smoking":0,
                        "time":15
                    }
                ]
            }
        }

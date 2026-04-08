from pydantic import BaseModel
from typing import Optional

class MedicalAction(BaseModel):
    # This is what the AGENT sends to the environment
    prediction: str

class MedicalObservation(BaseModel):
    # This is what the ENVIRONMENT sends back to the agent
    observation: str
    reward: float = 0.0  
    done: bool = False
    info: Optional[dict] = None
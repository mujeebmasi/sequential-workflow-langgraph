import os
from typing import TypedDict
from pydantic import BaseModel, field_validator
from dataclasses import dataclass, field

#using TypedDict 
class State(TypedDict):
    topic : str
    summary : str = ""  #defult value = ""
    score : int

#using pydantic
class State(BaseModel):
    topic : str
    summary : str = ""
    score : int
    
    @field_validator
    def score_positive(cls,v):
        if v <0:
            ValueError("score must be positve")
            
#using python dataclasses
@dataclass
class State:
    topic : str = ""
    summary : str = ""
    score : int 
    message : list = field(default_factory=list)

#using langgraph 
from langgraph.graph import MessageState
class State(MessageState):
    user_name : str
    language = str
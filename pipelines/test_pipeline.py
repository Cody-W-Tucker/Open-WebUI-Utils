"""
title: Journal RAG
author: Cody W Tucker
date: 2024-12-30
version: 1.0
license: MIT
description: A pipeline for testing.
requirements: pydantic==2.7.1
"""

import pydantic
print(f"Loaded Pydantic version: {pydantic.__version__}")

from pydantic import BaseModel
from typing import List  # Added this import

class Pipeline:
    class Valves(BaseModel):
        OPENAI_API_KEY: str
        model_config = {"extra": "allow"}

    def __init__(self):
        self.name = "Journal RAG"
        self.valves = self.Valves(**{"OPENAI_API_KEY": "test"})
    
    async def on_startup(self):
        print("Startup complete")
    
    async def on_shutdown(self):
        pass
    
    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> str:
        return "Hello"
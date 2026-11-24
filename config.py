from enum import Enum

import json
with open("./models.json") as f:
    models = json.load(f)
class Provider(Enum):
    OLLAMA='ollama'
    GROQ='groq'
    OPENAI='openai'

PROVIDER = Provider.GROQ
MODEL = models[PROVIDER][0]

def get_model(p:Provider=PROVIDER,model_name:str=MODEL):
    if p==Provider.OLLAMA:
        try:
            from langchain_ollama import ChatOllama
            model = ChatOllama(model=MODEL,temperature=0)
        except:
            raise ModuleNotFoundError("langchain_ollama")
    elif p==Provider.GROQ:
        try:
            from langchain_groq import ChatGroq
            model = ChatGroq(model=MODEL,temperature=0)
        except:
            raise ModuleNotFoundError("langchain_groq")
    elif p==Provider.OPENAI:
        try:
            from langchain_openai import ChatOpenAI
            model = ChatOpenAI(model=MODEL,temperature=0)
        except:
            raise ModuleNotFoundError("langchain_openai")
    else:
        raise Exception("Provider Not set in config.py")
    return model
    
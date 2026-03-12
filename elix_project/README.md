# Elix Chatbot

## Install dependencies

```powershell
..\\venv\\Scripts\\python.exe -m pip install -r requirements.txt
```

## Train the NLU model

```powershell
..\\venv\\Scripts\\python.exe manage.py train_nlu
```

## Evaluate the NLU model

```powershell
..\\venv\\Scripts\\python.exe manage.py evaluate_nlu
```

## Roll out ML NLU

Set environment variables before running Django:

```powershell
$env:NLU_ENGINE = "ml"
$env:NLU_MODEL_PATH = "D:\\elix-chatbot\\elix_project\\chatbot\\nlu_models\\intent_en_v1.joblib"
$env:NLU_CONFIDENCE_THRESHOLD = "0.55"
$env:NLU_ENABLE_RULES_FALLBACK = "true"
```

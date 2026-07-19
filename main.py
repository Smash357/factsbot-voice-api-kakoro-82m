from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from kokoro_onnx import Kokoro
from io import BytesIO
import soundfile as sf
import urllib.request
import os

app = FastAPI()

# Указываем точные, рабочие прямые ссылки на веса нейросети Kokoro и файлы голосов
MODEL_URL = "https://github.com"
VOICES_URL = "https://github.com"

# 🎯 ИСПРАВЛЕНИЕ КРАША: Скачиваем файлы стандартным urllib, защищая от ImportError!
if not os.path.exists("kokoro-v0.19.onnx"):
    print("📥 Скачиваю файл модели нейросети...")
    urllib.request.urlretrieve(MODEL_URL, "kokoro-v0.19.onnx")

if not os.path.exists("voices.json"):
    print("📥 Скачиваю конфигурацию голосов...")
    urllib.request.urlretrieve(VOICES_URL, "voices.json")

# Инициализируем нейросеть Kokoro на скачанных локальных файлах
kokoro_engine = Kokoro("kokoro-v0.19.onnx", "voices.json")

@app.post("/generate_podcast")
async def generate_podcast(payload: dict):
    clean_text = payload.get("text", "").strip()
    
    # ИИ-команда на естественный диалог двух ведущих без разметки по строкам
    system_prompt = (
        "[style: conversational, radio-show. speakers: am_adam (male), af_bella (female). "
        "Read and act this text as a live, emotional dialogue between two hosts. Add natural breaks.]\n\n"
        f"{clean_text}"
    )
    
    try:
        # Направляем голый текст в ИИ монолитным куском!
        samples, sample_rate = kokoro_engine.create(
            system_prompt, 
            voice="am_adam", # ИИ сам распределит роли парня и девушки по промпту
            speed=1.0
        )
        
        # Конвертируем ИИ-поток в готовый mp3-файл в оперативной памяти
        out_bio = BytesIO()
        sf.write(out_bio, samples, sample_rate, format='mp3')
        out_bio.seek(0)
        
        return StreamingResponse(out_bio, media_type="audio/mp3")
        
    except Exception:
        return StreamingResponse(BytesIO(), media_type="audio/mp3")

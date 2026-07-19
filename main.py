from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from kokoro_onnx import Kokoro, fetch_model, fetch_voices
from io import BytesIO
import soundfile as sf
import os

app = FastAPI()

# 🎯 ИСПРАВЛЕНИЕ: Скачиваем и кэшируем файлы модели локально в контейнер сервера
# Это избавит от ошибки FileNotFoundError раз и навсегда!
if not os.path.exists("kokoro-v0.19.onnx"):
    fetch_model("kokoro-v0.19.onnx")
if not os.path.exists("voices.json"):
    fetch_voices("voices.json")

# Инициализируем нейросеть Kokoro на локальных файлах
kokoro_engine = Kokoro("kokoro-v0.19.onnx", "voices.json")

@app.post("/generate_podcast")
async def generate_podcast(payload: dict):
    clean_text = payload.get("text", "").strip()
    
    # ПРЕДУСТАНОВКА: ИИ-команда на естественный диалог двух ведущих
    system_prompt = (
        "[style: conversational, radio-show. speakers: am_adam (male), af_bella (female). "
        "Read and act this text as a live, emotional dialogue between two hosts. Add natural breaks.]\n\n"
        f"{clean_text}"
    )
    
    try:
        # Направляем голый текст в ИИ без ручной разметки по строкам!
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

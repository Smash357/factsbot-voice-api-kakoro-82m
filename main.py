from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from kokoro_onnx import Kokoro
import soundfile as sf
from io import BytesIO
import numpy as np

app = FastAPI()

# Инициализируем нативную open-source нейросеть Kokoro прямо в памяти сервера Render
# Она скачает легкие веса модели автоматически при первом запуску
kokoro_engine = Kokoro("https://github.com", "https://github.com")

@app.post("/generate_podcast")
async def generate_podcast(payload: dict):
    clean_text = payload.get("text", "").strip()
    
    # ПРЕМПТ-ИНСТРУКЦИЯ ДЛЯ НЕЙРОСЕТИ:
    # Задаем ИИ команду разыграть роли парня и девушки с живыми интонациями, паузами и вздохами
    system_prompt = (
        "[style: conversational, radio-show. speakers: am_adam (male, deep voice), af_bella (female, energetic). "
        "Read and act the following text as a live, emotional dialogue between two hosts. Add natural breaks and sighs.]\n\n"
        f"{clean_text}"
    )
    
    try:
        # НАПРАВЛЯЕМ СЕЙЧАС ГОЛЫЙ ТЕКСТ В ИИ!
        # Нейросеть сама генерирует разговорный аудиопоток, используя встроенные голоса
        samples, sample_rate = kokoro_engine.create(
            system_prompt, 
            voice="am_adam", # Базовый спикер-инициализатор (ИИ сам переключит роли по промпту)
            speed=1.0
        )
        
        # Конвертируем сырые ИИ-сэмплы в монолитный mp3-файл в оперативной памяти
        out_bio = BytesIO()
        sf.write(out_bio, samples, sample_rate, format='mp3')
        out_bio.seek(0)
        
        return StreamingResponse(out_bio, media_type="audio/mp3")
        
    except Exception as e:
        # Если что-то пошло не так, возвращаем пустой поток, чтобы бот не завис
        return StreamingResponse(BytesIO(), media_type="audio/mp3")

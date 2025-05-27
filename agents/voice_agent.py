from .base_agent import BaseAgent
import whisper
import pyttsx3
from typing import Dict, Any, List

class VoiceAgent(BaseAgent):
    def __init__(self):
        super().__init__("voice_agent")
        self.stt_model = whisper.load_model("base")
        self.tts_engine = pyttsx3.init()
    
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        task_type = task.get("task_type")
        
        if task_type == "speech_to_text":
            audio_file = task.get("audio_file")
            return self.speech_to_text(audio_file)
        elif task_type == "text_to_speech":
            text = task.get("text")
            return self.text_to_speech(text)
        
        return {"error": "Unknown task type"}
    
    def speech_to_text(self, audio_file: str) -> Dict[str, Any]:
        result = self.stt_model.transcribe(audio_file)
        return {
            "transcription": result["text"],
            "confidence": 0.95,  # Whisper doesn't provide confidence
            "language": result.get("language", "en")
        }
    
    def text_to_speech(self, text: str) -> Dict[str, Any]:
        # Save to file or return audio data
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()
        
        return {
            "status": "success",
            "audio_length": len(text) * 0.1,  # Rough estimate
            "format": "wav"
        }
    
    def get_capabilities(self) -> List[str]:
        return ["speech_to_text", "text_to_speech", "audio_processing"]
import asyncio
import io
import logging
import tempfile
import wave
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import os

# Required installations:
# pip install openai google-cloud-texttospeech speechrecognition pyaudio pydub
import openai
from google.cloud import texttospeech
import speech_recognition as sr
from pydub import AudioSegment
from pydub.playback import play
import pyaudio

@dataclass
class VoiceConfig:
    # OpenAI Configuration
    openai_api_key: Optional[str] = None
    
    # Google Cloud Configuration (for TTS only)
    google_credentials_path: Optional[str] = None  # Path to service account JSON
    project_id: Optional[str] = None
    
    # STT Configuration (Whisper)
    stt_language: str = "en"  # ISO 639-1 language code for Whisper
    stt_timeout: float = 10.0
    whisper_model: str = "whisper-1"  # OpenAI Whisper model
    
    # TTS Configuration  
    tts_language_code: str = "en-US"
    tts_voice_name: str = "en-US-Neural2-F"  # Female neural voice
    tts_voice_gender: texttospeech.SsmlVoiceGender = texttospeech.SsmlVoiceGender.FEMALE
    tts_audio_encoding: texttospeech.AudioEncoding = texttospeech.AudioEncoding.MP3
    tts_speaking_rate: float = 1.0
    tts_pitch: float = 0.0
    
    # Audio Configuration
    sample_rate: int = 16000
    chunk_size: int = 1024
    audio_format: int = pyaudio.paInt16
    channels: int = 1

class WhisperVoiceAgent:
    def __init__(self, config: VoiceConfig = None, openai_api_key: Optional[str] = None):
        """
        Initialize Voice Agent with OpenAI Whisper and Google Cloud Text-to-Speech
        
        Args:
            config: Voice configuration settings
            openai_api_key: OpenAI API key for Whisper
        """
        self.config = config or VoiceConfig()
        self.logger = logging.getLogger(__name__)
        
        # Set up OpenAI API key
        self._setup_openai_credentials(openai_api_key)
        
        # Initialize OpenAI client
        try:
            self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
            self.logger.info("OpenAI client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
        
        # Set up Google Cloud credentials for TTS
        self._setup_google_credentials()
        
        # Initialize Google Cloud TTS client
        try:
            self.tts_client = texttospeech.TextToSpeechClient()
            self.logger.info("Google Cloud TTS client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Google Cloud TTS client: {e}")
            raise
        
        # Initialize speech recognition for local processing
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Audio setup
        self.audio = pyaudio.PyAudio()
        
        # Calibrate microphone for ambient noise
        self._calibrate_microphone()
        
        self.logger.info("Whisper Voice Agent initialized successfully")

    def _setup_openai_credentials(self, api_key: Optional[str]):
        """
        Set up OpenAI authentication
        """
        if api_key:
            self.openai_api_key = api_key
        elif self.config.openai_api_key:
            self.openai_api_key = self.config.openai_api_key
        elif os.environ.get('OPENAI_API_KEY'):
            self.openai_api_key = os.environ.get('OPENAI_API_KEY')
        else:
            self.logger.warning("No OpenAI API key provided. Set OPENAI_API_KEY environment variable or pass api_key parameter")
            self.openai_api_key = None

    def _setup_google_credentials(self):
        """
        Set up Google Cloud authentication for TTS
        """
        if self.config.google_credentials_path:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.config.google_credentials_path
        elif not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
            self.logger.warning("No Google Cloud credentials provided. Make sure to set GOOGLE_APPLICATION_CREDENTIALS environment variable")

    def _calibrate_microphone(self):
        """
        Calibrate microphone for ambient noise reduction
        """
        try:
            with self.microphone as source:
                self.logger.info("Calibrating microphone for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                self.logger.info("Microphone calibration complete")
        except Exception as e:
            self.logger.warning(f"Microphone calibration failed: {e}")

    async def listen_and_convert(self, 
                               timeout: Optional[float] = None,
                               phrase_timeout: float = 1.0,
                               use_whisper: bool = True) -> Dict[str, Any]:
        """
        Listen to microphone input and convert speech to text using OpenAI Whisper
        
        Args:
            timeout: Maximum time to wait for speech (None = indefinite)
            phrase_timeout: Seconds of silence before considering phrase complete
            use_whisper: Use OpenAI Whisper (True) or local recognition (False)
            
        Returns:
            Dict containing transcribed text and metadata
        """
        timeout = timeout or self.config.stt_timeout
        
        try:
            self.logger.info("Listening for voice input...")
            
            with self.microphone as source:
                # Listen for audio input
                audio_data = self.recognizer.listen(
                    source, 
                    timeout=timeout,
                    phrase_time_limit=phrase_timeout
                )
            
            self.logger.info("Audio captured, converting to text...")
            
            # Convert to text using Whisper or local recognition
            if use_whisper and self.openai_api_key:
                text, confidence = await self._whisper_stt(audio_data)
            else:
                text, confidence = await self._google_local_stt(audio_data)
            
            result = {
                "success": True,
                "text": text,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat(),
                "provider": "openai_whisper" if (use_whisper and self.openai_api_key) else "google_local",
                "language": self.config.stt_language
            }
            
            self.logger.info(f"STT successful: '{text[:50]}...' (confidence: {confidence})")
            return result
            
        except sr.WaitTimeoutError:
            return {
                "success": False,
                "error": "No speech detected within timeout period",
                "timestamp": datetime.now().isoformat()
            }
        except sr.UnknownValueError:
            return {
                "success": False,
                "error": "Could not understand the audio",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"STT error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def _whisper_stt(self, audio_data: sr.AudioData) -> tuple[str, str]:
        """
        Convert speech to text using OpenAI Whisper API
        """
        try:
            # Convert AudioData to WAV format
            wav_data = audio_data.get_wav_data()
            
            # Create a temporary file for the audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_audio.write(wav_data)
                temp_audio_path = temp_audio.name
            
            try:
                # Open the audio file and send to Whisper
                with open(temp_audio_path, "rb") as audio_file:
                    transcript = self.openai_client.audio.transcriptions.create(
                        model=self.config.whisper_model,
                        file=audio_file,
                        language=self.config.stt_language,
                        response_format="verbose_json"
                    )
                
                # Extract text and confidence (Whisper doesn't provide confidence scores)
                text = transcript.text.strip()
                
                # Estimate confidence based on text length and quality
                confidence = self._estimate_whisper_confidence(text)
                
                return text, confidence
                
            finally:
                # Clean up temporary file
                os.unlink(temp_audio_path)
                
        except Exception as e:
            self.logger.error(f"Whisper STT failed: {e}")
            # Fallback to local recognition
            return await self._google_local_stt(audio_data)

    def _estimate_whisper_confidence(self, text: str) -> str:
        """
        Estimate confidence level for Whisper transcription
        """
        if not text or len(text.strip()) < 3:
            return "low"
        elif len(text.split()) < 3:
            return "medium"
        elif any(word in text.lower() for word in ["um", "uh", "hmm", "..."]):
            return "medium"
        else:
            return "high"

    async def _google_local_stt(self, audio_data: sr.AudioData) -> tuple[str, str]:
        """
        Convert speech to text using local Google Speech Recognition (fallback)
        """
        try:
            text = self.recognizer.recognize_google(
                audio_data, 
                language=self.config.stt_language if self.config.stt_language.startswith('en') else 'en-US'
            )
            return text, "medium"  # Local recognition doesn't provide confidence scores
        except Exception as e:
            raise Exception(f"Google local STT failed: {str(e)}")

    async def convert_and_speak(self, 
                              text: str,
                              save_audio: bool = False,
                              audio_path: Optional[str] = None,
                              voice_settings: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Convert text to speech using Google Cloud Text-to-Speech and play it
        
        Args:
            text: Text to convert to speech
            save_audio: Whether to save audio file
            audio_path: Path to save audio file (if save_audio=True)
            voice_settings: Override default voice settings
            
        Returns:
            Dict containing success status and metadata
        """
        try:
            self.logger.info(f"Converting text to speech: '{text[:50]}...'")
            
            # Generate speech using Google Cloud TTS
            audio_content = await self._google_tts(text, voice_settings)
            
            # Save audio if requested
            saved_path = None
            if save_audio:
                saved_path = audio_path or f"tts_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
                with open(saved_path, "wb") as f:
                    f.write(audio_content)
                self.logger.info(f"Audio saved to: {saved_path}")
            
            # Play audio
            await self._play_audio(audio_content)
            
            result = {
                "success": True,
                "text": text,
                "audio_length_seconds": self._estimate_audio_length(text),
                "timestamp": datetime.now().isoformat(),
                "provider": "google_cloud_tts",
                "voice": voice_settings.get("voice_name", self.config.tts_voice_name) if voice_settings else self.config.tts_voice_name,
                "saved_path": saved_path
            }
            
            self.logger.info("TTS completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"TTS error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "text": text,
                "timestamp": datetime.now().isoformat()
            }

    async def _google_tts(self, text: str, voice_settings: Optional[Dict] = None) -> bytes:
        """
        Convert text to speech using Google Cloud Text-to-Speech
        """
        # Use custom voice settings or defaults
        settings = voice_settings or {}
        
        # Set the text input to be synthesized
        synthesis_input = texttospeech.SynthesisInput(text=text)
        
        # Build the voice request
        voice = texttospeech.VoiceSelectionParams(
            language_code=settings.get("language_code", self.config.tts_language_code),
            name=settings.get("voice_name", self.config.tts_voice_name),
            ssml_gender=settings.get("gender", self.config.tts_voice_gender),
        )
        
        # Select the type of audio file
        audio_config = texttospeech.AudioConfig(
            audio_encoding=settings.get("encoding", self.config.tts_audio_encoding),
            speaking_rate=settings.get("speaking_rate", self.config.tts_speaking_rate),
            pitch=settings.get("pitch", self.config.tts_pitch),
        )
        
        # Perform the text-to-speech request
        response = self.tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        
        return response.audio_content

    async def _play_audio(self, audio_content: bytes):
        """
        Play audio content
        """
        try:
            # Convert bytes to AudioSegment
            audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_content))
            
            # Play audio
            play(audio_segment)
        except Exception as e:
            self.logger.error(f"Audio playback error: {str(e)}")
            raise

    def _estimate_audio_length(self, text: str) -> float:
        """
        Estimate audio length based on text length and speech speed
        """
        # Average speaking rate: ~150 words per minute
        words = len(text.split())
        base_duration = (words / 150) * 60  # seconds
        return base_duration / self.config.tts_speaking_rate

    def get_available_voices(self) -> Dict[str, Any]:
        """
        Get list of available Google TTS voices
        """
        try:
            voices = self.tts_client.list_voices()
            
            voice_list = []
            for voice in voices.voices:
                voice_info = {
                    "name": voice.name,
                    "language_codes": list(voice.language_codes),
                    "gender": voice.ssml_gender.name,
                    "natural_sample_rate": voice.natural_sample_rate_hertz
                }
                voice_list.append(voice_info)
            
            return {
                "success": True,
                "voices": voice_list,
                "total_count": len(voice_list)
            }
        except Exception as e:
            self.logger.error(f"Error fetching voices: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def transcribe_audio_file(self, file_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe an audio file using Whisper
        
        Args:
            file_path: Path to audio file
            language: Language code (optional, auto-detect if None)
            
        Returns:
            Dict containing transcription results
        """
        try:
            self.logger.info(f"Transcribing audio file: {file_path}")
            
            if not self.openai_api_key:
                raise Exception("OpenAI API key not configured")
            
            with open(file_path, "rb") as audio_file:
                transcript = self.openai_client.audio.transcriptions.create(
                    model=self.config.whisper_model,
                    file=audio_file,
                    language=language or self.config.stt_language,
                    response_format="verbose_json"
                )
            
            result = {
                "success": True,
                "text": transcript.text,
                "language": transcript.language if hasattr(transcript, 'language') else language,
                "duration": transcript.duration if hasattr(transcript, 'duration') else None,
                "segments": transcript.segments if hasattr(transcript, 'segments') else None,
                "timestamp": datetime.now().isoformat(),
                "provider": "openai_whisper",
                "file_path": file_path
            }
            
            self.logger.info(f"File transcription successful: '{transcript.text[:50]}...'")
            return result
            
        except Exception as e:
            self.logger.error(f"File transcription error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path,
                "timestamp": datetime.now().isoformat()
            }

    async def interactive_session(self):
        """
        Run an interactive voice session for testing
        """
        print("🎤 Whisper Voice Agent Interactive Session")
        print("Say 'exit' or 'quit' to end the session")
        print("=" * 50)
        
        # Check API key status
        if not self.openai_api_key:
            print("⚠️  Warning: No OpenAI API key configured. Using local Google STT as fallback.")
        else:
            print("✅ OpenAI Whisper API configured")
        
        # Show available voices
        print("\n🎵 Available voice options:")
        voices = self.get_available_voices()
        if voices["success"]:
            en_voices = [v for v in voices["voices"] if "en-US" in v["language_codes"]][:5]
            for voice in en_voices:
                print(f"  - {voice['name']} ({voice['gender']})")
        
        while True:
            try:
                print("\n🎤 Listening... (speak now)")
                
                # Listen for input
                stt_result = await self.listen_and_convert(timeout=10.0)
                
                if not stt_result["success"]:
                    print(f"❌ STT Error: {stt_result['error']}")
                    continue
                
                user_text = stt_result["text"]
                print(f"📝 You said: '{user_text}' (confidence: {stt_result['confidence']}, provider: {stt_result['provider']})")
                
                # Check for exit commands
                if user_text.lower() in ['exit', 'quit', 'stop', 'end']:
                    print("👋 Ending voice session...")
                    break
                
                # Echo back the text (in a real system, this would go to your Language Agent)
                response_text = f"I heard you say: {user_text}. This would normally be processed by the financial analysis system using Whisper transcription."
                
                print(f"🤖 Response: {response_text}")
                
                # Convert response to speech
                tts_result = await self.convert_and_speak(response_text)
                
                if not tts_result["success"]:
                    print(f"❌ TTS Error: {tts_result['error']}")
                
            except KeyboardInterrupt:
                print("\n👋 Session ended by user")
                break
            except Exception as e:
                print(f"❌ Session error: {str(e)}")

    def cleanup(self):
        """
        Clean up audio resources
        """
        try:
            self.audio.terminate()
            self.logger.info("Whisper Voice Agent cleanup completed")
        except Exception as e:
            self.logger.error(f"Cleanup error: {str(e)}")

    def __del__(self):
        """
        Destructor to ensure cleanup
        """
        self.cleanup()

# Integration class for connecting to your Language Agent
class VoiceLanguageInterface:
    def __init__(self, voice_agent: WhisperVoiceAgent, language_agent=None):
        """
        Interface to connect Whisper Voice Agent with Language Agent
        """
        self.voice_agent = voice_agent
        self.language_agent = language_agent
        self.logger = logging.getLogger(__name__)

    async def voice_query_pipeline(self, timeout: float = 10.0) -> Dict[str, Any]:
        """
        Complete pipeline: Voice Input → Language Processing → Voice Output
        """
        pipeline_result = {
            "stt_result": None,
            "language_result": None,
            "tts_result": None,
            "success": False,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Step 1: Convert speech to text using Whisper
            print("🎤 Listening for your financial query...")
            stt_result = await self.voice_agent.listen_and_convert(timeout=timeout)
            pipeline_result["stt_result"] = stt_result
            
            if not stt_result["success"]:
                return pipeline_result
            
            user_query = stt_result["text"]
            print(f"📝 Query: {user_query} (via {stt_result['provider']})")
            
            # Step 2: Process with Language Agent
            if self.language_agent:
                # Integration with your actual Language Agent would go here
                language_result = await self.language_agent.process_query(user_query)
                response_text = language_result.get("response", "No response generated")
            else:
                # Simulated response for testing
                response_text = f"Based on current market data analysis, here's what I found regarding '{user_query}': Market indicators show positive momentum with key metrics suggesting continued growth in the financial sectors you're interested in. This analysis was processed using Whisper speech recognition."
            
            pipeline_result["language_result"] = {
                "success": True,
                "response": response_text,
                "query": user_query
            }
            
            # Step 3: Convert response to speech
            print("🤖 Converting response to speech...")
            tts_result = await self.voice_agent.convert_and_speak(response_text)
            pipeline_result["tts_result"] = tts_result
            
            pipeline_result["success"] = tts_result["success"]
            
            return pipeline_result
            
        except Exception as e:
            self.logger.error(f"Pipeline error: {str(e)}")
            pipeline_result["error"] = str(e)
            return pipeline_result

# Example usage and testing
async def test_whisper_voice_agent():
    """
    Test the Whisper Voice Agent functionality
    """
    # Initialize with custom config
    config = VoiceConfig(
        tts_voice_name="en-US-Neural2-F",  # Female neural voice
        tts_speaking_rate=1.1,
        stt_language="en",  # Whisper language code
        whisper_model="whisper-1"
    )
    
    # Initialize agent (make sure to set OPENAI_API_KEY and GOOGLE_APPLICATION_CREDENTIALS environment variables)
    voice_agent = WhisperVoiceAgent(config)
    
    # Test listing available voices
    print("Available Google TTS voices:")
    voices_result = voice_agent.get_available_voices()
    if voices_result["success"]:
        print(f"Found {voices_result['total_count']} voices")
        # Show first few English voices
        en_voices = [v for v in voices_result["voices"] if "en-US" in v["language_codes"]][:3]
        for voice in en_voices:
            print(f"  - {voice['name']} ({voice['gender']})")
    
    # Test TTS first (doesn't require microphone)
    print("\nTesting Google Text-to-Speech...")
    tts_result = await voice_agent.convert_and_speak(
        "Hello! I'm your financial AI assistant powered by OpenAI Whisper and Google Cloud. I can help you analyze market data, portfolio performance, and investment opportunities using advanced speech recognition."
    )
    print(f"TTS Result: {json.dumps(tts_result, indent=2, default=str)}")
    
    # Test STT (requires microphone)
    print("\nTesting Whisper Speech-to-Text (speak something about finance)...")
    stt_result = await voice_agent.listen_and_convert(timeout=8.0)
    print(f"STT Result: {json.dumps(stt_result, indent=2, default=str)}")
    
    # Test complete pipeline
    print("\nTesting complete voice pipeline...")
    interface = VoiceLanguageInterface(voice_agent)
    pipeline_result = await interface.voice_query_pipeline(timeout=8.0)
    print(f"Pipeline Result: {json.dumps(pipeline_result, indent=2, default=str)}")
    
    voice_agent.cleanup()

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    asyncio.run(test_whisper_voice_agent())
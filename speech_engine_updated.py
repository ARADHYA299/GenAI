# core/speech_engine.py - Advanced Speech Recognition and Text-to-Speech
import asyncio
import speech_recognition as sr
import pyttsx3
import pyaudio
import wave
import numpy as np
from typing import Optional, Callable, Dict, List
import threading
import queue
import time
from dataclasses import dataclass
from utils.logger import setup_logger
from config.settings import SETTINGS

@dataclass
class AudioConfig:
    """Audio configuration settings"""
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    format: int = pyaudio.paInt16
    wake_word_threshold: float = 0.7
    noise_threshold: float = 0.3

class SpeechEngine:
    """
    Advanced Speech Recognition and Text-to-Speech Engine
    
    Key Features:
    1. Continuous listening with wake word detection
    2. Noise cancellation and voice activity detection
    3. Multi-threading for non-blocking audio processing
    4. Configurable voice profiles and languages
    5. Audio feedback and status indicators
    
    Interview Topics:
    - Threading vs Asyncio for audio processing
    - Producer-Consumer pattern with queues
    - Real-time audio processing concepts
    - Error handling in streaming applications
    """
    
    def __init__(self):
        self.logger = setup_logger("speech_engine")
        self.config = AudioConfig()
        
        # Speech Recognition
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.is_listening = False
        self.wake_word_detected = False
        
        # Text-to-Speech
        self.tts_engine = None
        self.voice_config = SETTINGS.get("voice", {})
        
        # Audio Processing
        self.audio_queue = queue.Queue()
        self.command_queue = queue.Queue()
        self.audio_thread = None
        self.processing_thread = None
        
        # Wake Word Detection (simplified - in production use specialized libraries)
        self.wake_words = ["jarvis", "hey jarvis", "ok jarvis"]
        
        # Event callbacks
        self.on_wake_word: Optional[Callable] = None
        self.on_command: Optional[Callable] = None
        self.on_listening_start: Optional[Callable] = None
        self.on_listening_stop: Optional[Callable] = None
        
    async def initialize(self):
        """Initialize speech recognition and TTS engines"""
        try:
            self.logger.info("🎤 Initializing Speech Engine...")
            
            # Initialize microphone with error handling
            await self._initialize_microphone()
            
            # Initialize TTS engine
            await self._initialize_tts()
            
            # Calibrate for ambient noise
            await self._calibrate_microphone()
            
            # Start background threads
            self._start_audio_threads()
            
            self.logger.info("✅ Speech Engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Speech Engine initialization failed: {e}")
            raise
    
    async def _initialize_microphone(self):
        """Initialize microphone with proper error handling"""
        try:
            # Test microphone availability
            self.microphone = sr.Microphone()
            
            # Configure recognizer settings
            self.recognizer.energy_threshold = 300
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 0.8
            self.recognizer.operation_timeout = None
            self.recognizer.phrase_threshold = 0.3
            self.recognizer.non_speaking_duration = 0.5
            
            self.logger.info("🎙️ Microphone initialized")
            
        except Exception as e:
            self.logger.error(f"Microphone initialization error: {e}")
            raise
    
    async def _initialize_tts(self):
        """Initialize Text-to-Speech engine"""
        try:
            self.tts_engine = pyttsx3.init()
            
            # Configure voice settings
            voices = self.tts_engine.getProperty('voices')
            
            # Set voice based on configuration
            preferred_voice = self.voice_config.get("voice_id")
            if preferred_voice:
                self.tts_engine.setProperty('voice', preferred_voice)
            elif voices:
                # Default to first available voice
                self.tts_engine.setProperty('voice', voices[0].id)
            
            # Set speech rate and volume
            self.tts_engine.setProperty('rate', self.voice_config.get("rate", 180))
            self.tts_engine.setProperty('volume', self.voice_config.get("volume", 0.9))
            
            self.logger.info("🔊 TTS Engine initialized")
            
        except Exception as e:
            self.logger.error(f"TTS initialization error: {e}")
            raise
    
    async def _calibrate_microphone(self):
        """Calibrate microphone for ambient noise"""
        try:
            self.logger.info("🔧 Calibrating microphone for ambient noise...")
            
            with self.microphone as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
                
            self.logger.info(f"🎯 Microphone calibrated - Energy threshold: {self.recognizer.energy_threshold}")
            
        except Exception as e:
            self.logger.warning(f"Microphone calibration failed: {e}")
    
    def _start_audio_threads(self):
        """Start background threads for audio processing"""
        # Audio capture thread
        self.audio_thread = threading.Thread(
            target=self._audio_capture_loop,
            daemon=True,
            name="AudioCapture"
        )
        self.audio_thread.start()
        
        # Audio processing thread
        self.processing_thread = threading.Thread(
            target=self._audio_processing_loop,
            daemon=True,
            name="AudioProcessing"
        )
        self.processing_thread.start()
        
        self.logger.info("🔄 Audio processing threads started")
    
    def _audio_capture_loop(self):
        """
        Continuous audio capture loop - Producer in Producer-Consumer pattern
        
        Why threading here instead of asyncio?
        - Audio capture is I/O bound but requires precise timing
        - pyaudio callbacks work better with threads
        - Separates audio capture from processing logic
        """
        while self.is_listening:
            try:
                with self.microphone as source:
                    # Listen for audio with timeout
                    audio = self.recognizer.listen(
                        source,
                        timeout=1,
                        phrase_time_limit=5
                    )
                    
                    # Add to processing queue
                    self.audio_queue.put(audio)
                    
            except sr.WaitTimeoutError:
                # Normal timeout, continue listening
                continue
            except Exception as e:
                self.logger.error(f"Audio capture error: {e}")
                time.sleep(0.1)  # Brief pause before retry
    
    def _audio_processing_loop(self):
        """
        Audio processing loop - Consumer in Producer-Consumer pattern
        
        Processing Pipeline:
        1. Get audio from queue
        2. Check for wake word
        3. If wake word detected, process command
        4. Handle recognition errors gracefully
        """
        while True:
            try:
                # Get audio from queue (blocking)
                audio = self.audio_queue.get(timeout=1)
                
                # Process audio
                self._process_audio(audio)
                
                # Mark task as done
                self.audio_queue.task_done()
                
            except queue.Empty:
                # No audio to process, continue
                continue
            except Exception as e:
                self.logger.error(f"Audio processing error: {e}")
    
    def _process_audio(self, audio):
        """Process captured audio for wake words and commands"""
        try:
            # Recognize speech
            text = self.recognizer.recognize_google(
                audio,
                language=self.voice_config.get("language", "en-US")
            ).lower()
            
            self.logger.debug(f"Recognized: {text}")
            
            # Check for wake word
            if self._contains_wake_word(text):
                self.logger.info(f"🎯 Wake word detected: {text}")
                self.wake_word_detected = True
                
                # Trigger wake word callback
                if self.on_wake_word:
                    asyncio.create_task(self.on_wake_word({"text": text}))
                
                # Process as command if it contains more than just wake word
                command = self._extract_command_from_wake_phrase(text)
                if command:
                    self._handle_command(command)
                    
            elif self.wake_word_detected:
                # Process as command after wake word
                self._handle_command(text)
                self.wake_word_detected = False
                
        except sr.UnknownValueError:
            # Could not understand audio - normal occurrence
            pass
        except sr.RequestError as e:
            self.logger.error(f"Speech recognition service error: {e}")
        except Exception as e:
            self.logger.error(f"Audio processing error: {e}")
    
    def _contains_wake_word(self, text: str) -> bool:
        """Check if text contains wake word"""
        return any(wake_word in text for wake_word in self.wake_words)
    
    def _extract_command_from_wake_phrase(self, text: str) -> Optional[str]:
        """Extract command from wake phrase (e.g., 'Hey Jarvis, what time is it')"""
        for wake_word in self.wake_words:
            if wake_word in text:
                # Extract everything after the wake word
                command = text.split(wake_word, 1)[-1].strip()
                if len(command) > 2:  # Ignore very short commands
                    return command
        return None
    
    def _handle_command(self, command: str):
        """Handle recognized command"""
        self.logger.info(f"🎤 Command received: {command}")
        
        # Add to command queue for processing
        self.command_queue.put(command)
        
        # Trigger command callback
        if self.on_command:
            asyncio.create_task(self.on_command({"command": command}))
    
    async def listen_for_commands(self):
        """Start listening for voice commands"""
        if not self.is_listening:
            self.is_listening = True
            self.logger.info("👂 Started listening for commands")
            
            if self.on_listening_start:
                await self.on_listening_start({})
    
    async def stop_listening(self):
        """Stop listening for voice commands"""
        if self.is_listening:
            self.is_listening = False
            self.wake_word_detected = False
            self.logger.info("🔇 Stopped listening")
            
            if self.on_listening_stop:
                await self.on_listening_stop({})
    
    async def speak(self, text: str, interrupt: bool = False):
        """
        Convert text to speech with async support
        
        Args:
            text: Text to speak
            interrupt: Whether to interrupt current speech
        """
        try:
            if interrupt and self.tts_engine._inLoop:
                self.tts_engine.stop()
            
            self.logger.info(f"🔊 Speaking: {text[:50]}...")
            
            # Run TTS in thread to avoid blocking
            await asyncio.get_event_loop().run_in_executor(
                None, self._speak_sync, text
            )
            
        except Exception as e:
            self.logger.error(f"TTS error: {e}")
    
    def _speak_sync(self, text: str):
        """Synchronous TTS execution"""
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()
    
    async def play_acknowledgment_sound(self):
        """Play acknowledgment sound when wake word is detected"""
        try:
            # Play a simple beep or sound file
            # Implementation would play actual sound file
            self.logger.info("🔔 Playing acknowledgment sound")
        except Exception as e:
            self.logger.error(f"Acknowledgment sound error: {e}")
    
    async def set_voice_config(self, config: Dict):
        """Update voice configuration dynamically"""
        try:
            self.voice_config.update(config)
            
            if "rate" in config:
                self.tts_engine.setProperty('rate', config["rate"])
            if "volume" in config:
                self.tts_engine.setProperty('volume', config["volume"])
            if "voice_id" in config:
                self.tts_engine.setProperty('voice', config["voice_id"])
                
            self.logger.info("Voice configuration updated")
            
        except Exception as e:
            self.logger.error(f"Voice config update error: {e}")
    
    async def get_available_voices(self) -> List[Dict]:
        """Get list of available TTS voices"""
        try:
            voices = self.tts_engine.getProperty('voices')
            return [
                {
                    "id": voice.id,
                    "name": voice.name,
                    "language": getattr(voice, 'languages', ['unknown']),
                    "gender": getattr(voice, 'gender', 'unknown')
                }
                for voice in voices
            ]
        except Exception as e:
            self.logger.error(f"Error getting voices: {e}")
            return []
    
    def get_pending_commands(self) -> List[str]:
        """Get pending voice commands from queue"""
        commands = []
        while not self.command_queue.empty():
            try:
                commands.append(self.command_queue.get_nowait())
            except queue.Empty:
                break
        return commands
    
    async def health_check(self) -> bool:
        """Check speech engine health"""
        try:
            # Check if microphone is available
            if not self.microphone:
                return False
            
            # Check if TTS engine is working
            if not self.tts_engine:
                return False
            
            # Check if threads are alive
            if self.audio_thread and not self.audio_thread.is_alive():
                return False
            
            if self.processing_thread and not self.processing_thread.is_alive():
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Health check error: {e}")
            return False
    
    async def get_audio_stats(self) -> Dict:
        """Get audio processing statistics"""
        return {
            "is_listening": self.is_listening,
            "wake_word_detected": self.wake_word_detected,
            "audio_queue_size": self.audio_queue.qsize(),
            "command_queue_size": self.command_queue.qsize(),
            "energy_threshold": self.recognizer.energy_threshold,
            "microphone_available": self.microphone is not None,
            "tts_available": self.tts_engine is not None
        }
    
    async def shutdown(self):
        """Graceful shutdown of speech engine"""
        self.logger.info("🔄 Shutting down Speech Engine...")
        
        # Stop listening
        await self.stop_listening()
        
        # Wait for threads to finish current tasks
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=2)
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2)
        
        # Cleanup TTS engine
        if self.tts_engine:
            try:
                self.tts_engine.stop()
            except:
                pass
        
        self.logger.info("✅ Speech Engine shutdown complete")
# core/jarvis_core.py - Main JARVIS Engine
import asyncio
import json
import os
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime

from .speech_engine import SpeechEngine
from .nlp_processor import NLPProcessor
from .automation_engine import AutomationEngine
from .memory_manager import MemoryManager
from .web_interface import WebInterface
from .plugin_manager import PluginManager
from utils.logger import setup_logger
from config.settings import SETTINGS

@dataclass
class JarvisResponse:
    """Structured response format - Important for API consistency"""
    text: str
    action: Optional[str] = None
    data: Optional[Dict] = None
    confidence: float = 0.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class JarvisCore:
    """
    Central JARVIS Engine - Orchestrates all components
    
    Key Architecture Patterns:
    1. Observer Pattern: Event-driven communication
    2. Strategy Pattern: Pluggable command handlers
    3. Singleton Pattern: Single instance coordination
    4. Factory Pattern: Dynamic component creation
    """
    
    def __init__(self):
        self.logger = setup_logger("jarvis_core")
        
        # Core components - Dependency Injection pattern
        self.speech_engine: Optional[SpeechEngine] = None
        self.nlp_processor: Optional[NLPProcessor] = None
        self.automation_engine: Optional[AutomationEngine] = None
        self.memory_manager: Optional[MemoryManager] = None
        self.web_interface: Optional[WebInterface] = None
        self.plugin_manager: Optional[PluginManager] = None
        
        # Event system - Observer pattern implementation
        self.event_handlers: Dict[str, List[Callable]] = {}
        
        # State management
        self.is_active = False
        self.current_context = {}
        self.conversation_history = []
        
    async def initialize(self):
        """Initialize all JARVIS components with proper error handling"""
        try:
            self.logger.info("🔧 Initializing JARVIS components...")
            
            # Initialize components in dependency order
            self.memory_manager = MemoryManager()
            await self.memory_manager.initialize()
            
            self.nlp_processor = NLPProcessor()
            await self.nlp_processor.initialize()
            
            self.speech_engine = SpeechEngine()
            await self.speech_engine.initialize()
            
            self.automation_engine = AutomationEngine()
            await self.automation_engine.initialize()
            
            self.plugin_manager = PluginManager()
            await self.plugin_manager.load_plugins()
            
            self.web_interface = WebInterface(self)
            await self.web_interface.initialize()
            
            # Setup event handlers
            self._setup_event_handlers()
            
            self.is_active = True
            self.logger.info("✅ All JARVIS components initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Component initialization failed: {e}")
            raise
    
    def _setup_event_handlers(self):
        """Setup event-driven communication between components"""
        # Voice command events
        self.register_event_handler("voice_command", self._handle_voice_command)
        self.register_event_handler("wake_word_detected", self._handle_wake_word)
        
        # System events
        self.register_event_handler("system_alert", self._handle_system_alert)
        self.register_event_handler("automation_trigger", self._handle_automation)
        
        # Web interface events
        self.register_event_handler("web_command", self._handle_web_command)
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """Register event handler - Observer pattern"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    async def emit_event(self, event_type: str, data: Dict = None):
        """Emit event to all registered handlers"""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    await handler(data or {})
                except Exception as e:
                    self.logger.error(f"Event handler error: {e}")
    
    async def process_command(self, command: str, source: str = "voice") -> JarvisResponse:
        """
        Main command processing pipeline
        
        Processing Steps:
        1. NLP analysis and intent extraction
        2. Context awareness and memory retrieval
        3. Command routing and execution
        4. Response generation and delivery
        """
        try:
            self.logger.info(f"🎯 Processing command: '{command}' from {source}")
            
            # Step 1: NLP Analysis
            nlp_result = await self.nlp_processor.analyze(command)
            intent = nlp_result.get("intent")
            entities = nlp_result.get("entities", {})
            confidence = nlp_result.get("confidence", 0.0)
            
            # Step 2: Context Enhancement
            context = await self._build_context(command, intent, entities)
            
            # Step 3: Command Routing (Strategy Pattern)
            response = await self._route_command(intent, entities, context, confidence)
            
            # Step 4: Memory Storage
            await self._store_interaction(command, response, source)
            
            # Step 5: Response Delivery
            await self._deliver_response(response, source)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Command processing error: {e}")
            error_response = JarvisResponse(
                text="I encountered an error processing that command. Please try again.",
                confidence=0.0
            )
            return error_response
    
    async def _build_context(self, command: str, intent: str, entities: Dict) -> Dict:
        """Build execution context with memory and environment data"""
        context = {
            "current_time": datetime.now(),
            "user_location": await self._get_user_location(),
            "recent_history": await self.memory_manager.get_recent_interactions(5),
            "system_status": await self._get_system_status(),
            "user_preferences": await self.memory_manager.get_user_preferences(),
            "active_applications": await self.automation_engine.get_active_apps(),
        }
        return context
    
    async def _route_command(self, intent: str, entities: Dict, context: Dict, confidence: float) -> JarvisResponse:
        """Route command to appropriate handler based on intent"""
        
        # Command routing map - Strategy Pattern
        command_handlers = {
            "automation": self._handle_automation_command,
            "information": self._handle_information_command,
            "system_control": self._handle_system_command,
            "entertainment": self._handle_entertainment_command,
            "productivity": self._handle_productivity_command,
            "personal": self._handle_personal_command,
        }
        
        # Route to appropriate handler
        handler = command_handlers.get(intent, self._handle_unknown_command)
        return await handler(entities, context, confidence)
    
    async def _handle_automation_command(self, entities: Dict, context: Dict, confidence: float) -> JarvisResponse:
        """Handle automation commands (smart home, workflows, etc.)"""
        action = entities.get("action")
        target = entities.get("target")
        
        result = await self.automation_engine.execute_automation(action, target, context)
        
        return JarvisResponse(
            text=result.get("message", "Automation executed successfully"),
            action="automation",
            data=result,
            confidence=confidence
        )
    
    async def _handle_information_command(self, entities: Dict, context: Dict, confidence: float) -> JarvisResponse:
        """Handle information requests (weather, news, facts, etc.)"""
        query_type = entities.get("query_type")
        subject = entities.get("subject")
        
        # Use appropriate information source
        if query_type == "weather":
            info = await self._get_weather_info(subject or context.get("user_location"))
        elif query_type == "news":
            info = await self._get_news_info(subject)
        elif query_type == "time":
            info = await self._get_time_info(subject)
        else:
            info = await self._get_general_info(subject)
        
        return JarvisResponse(
            text=info.get("text", "Here's what I found"),
            action="information",
            data=info,
            confidence=confidence
        )
    
    async def _handle_system_command(self, entities: Dict, context: Dict, confidence: float) -> JarvisResponse:
        """Handle system control commands"""
        action = entities.get("action")
        target = entities.get("target")
        
        result = await self.automation_engine.execute_system_command(action, target)
        
        return JarvisResponse(
            text=result.get("message", "System command executed"),
            action="system_control",
            data=result,
            confidence=confidence
        )
    
    async def _handle_unknown_command(self, entities: Dict, context: Dict, confidence: float) -> JarvisResponse:
        """Handle unrecognized commands with fallback strategies"""
        # Try plugin handlers first
        plugin_response = await self.plugin_manager.try_plugins(entities, context)
        if plugin_response:
            return plugin_response
        
        # Fallback to general AI response
        return JarvisResponse(
            text="I'm not sure how to help with that. Could you rephrase your request?",
            confidence=0.1
        )
    
    async def start_voice_listening(self):
        """Start continuous voice listening in background"""
        while self.is_active:
            try:
                await self.speech_engine.listen_for_commands()
                await asyncio.sleep(0.1)  # Prevent CPU overload
            except Exception as e:
                self.logger.error(f"Voice listening error: {e}")
                await asyncio.sleep(1)  # Brief pause before retry
    
    async def start_web_interface(self):
        """Start web interface server"""
        await self.web_interface.start_server()
    
    async def start_background_monitoring(self):
        """Start background system monitoring"""
        while self.is_active:
            try:
                # Monitor system resources
                await self._monitor_system_health()
                
                # Check for automated tasks
                await self.automation_engine.check_scheduled_tasks()
                
                # Clean up old data
                await self.memory_manager.cleanup_old_data()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Background monitoring error: {e}")
    
    async def health_check(self):
        """System health check - Important for production monitoring"""
        health_status = {
            "jarvis_core": self.is_active,
            "speech_engine": await self.speech_engine.health_check(),
            "nlp_processor": await self.nlp_processor.health_check(),
            "automation_engine": await self.automation_engine.health_check(),
            "memory_manager": await self.memory_manager.health_check(),
            "web_interface": await self.web_interface.health_check(),
        }
        
        all_healthy = all(health_status.values())
        if not all_healthy:
            self.logger.warning(f"Health check issues detected: {health_status}")
        
        return health_status
    
    async def shutdown(self):
        """Graceful shutdown of all components"""
        self.logger.info("🔄 Shutting down JARVIS components...")
        self.is_active = False
        
        # Shutdown in reverse dependency order
        if self.web_interface:
            await self.web_interface.shutdown()
        if self.automation_engine:
            await self.automation_engine.shutdown()
        if self.speech_engine:
            await self.speech_engine.shutdown()
        if self.nlp_processor:
            await self.nlp_processor.shutdown()
        if self.memory_manager:
            await self.memory_manager.shutdown()
        
        self.logger.info("✅ JARVIS shutdown complete")
    
    # Event Handlers
    async def _handle_voice_command(self, data: Dict):
        """Handle voice command events"""
        command = data.get("command", "")
        if command:
            await self.process_command(command, "voice")
    
    async def _handle_wake_word(self, data: Dict):
        """Handle wake word detection"""
        self.logger.info("👂 Wake word detected")
        await self.speech_engine.play_acknowledgment_sound()
    
    async def _handle_web_command(self, data: Dict):
        """Handle web interface commands"""
        command = data.get("command", "")
        if command:
            await self.process_command(command, "web")
    
    # Helper methods
    async def _get_user_location(self) -> str:
        """Get user location for context"""
        # Implementation would use IP geolocation or GPS
        return "Unknown"
    
    async def _get_system_status(self) -> Dict:
        """Get current system status"""
        return {
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "disk_usage": 0.0,
            "network_status": "connected"
        }
    
    async def _store_interaction(self, command: str, response: JarvisResponse, source: str):
        """Store interaction in memory for learning"""
        interaction = {
            "timestamp": datetime.now(),
            "command": command,
            "response": response.text,
            "source": source,
            "confidence": response.confidence
        }
        await self.memory_manager.store_interaction(interaction)
    
    async def _deliver_response(self, response: JarvisResponse, source: str):
        """Deliver response through appropriate channel"""
        if source == "voice":
            await self.speech_engine.speak(response.text)
        elif source == "web":
            await self.web_interface.send_response(response)
    
    async def _monitor_system_health(self):
        """Monitor system health metrics"""
        # Implementation would check CPU, memory, disk, network
        pass
    
    async def _get_weather_info(self, location: str) -> Dict:
        """Get weather information"""
        # Implementation would use weather API
        return {"text": f"Weather information for {location} is not available right now."}
    
    async def _get_news_info(self, topic: str) -> Dict:
        """Get news information"""
        # Implementation would use news API
        return {"text": f"Latest news about {topic} is not available right now."}
    
    async def _get_time_info(self, timezone: str) -> Dict:
        """Get time information"""
        current_time = datetime.now().strftime("%I:%M %p")
        return {"text": f"The current time is {current_time}"}
    
    async def _get_general_info(self, query: str) -> Dict:
        """Get general information using AI"""
        # Implementation would use LLM API
        return {"text": f"I don't have specific information about {query} right now."}

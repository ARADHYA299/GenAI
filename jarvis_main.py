# main.py - JARVIS AI Assistant
import asyncio
import logging
import sys
import signal
from pathlib import Path
from core.jarvis_core import JarvisCore
from config.settings import SETTINGS
from utils.logger import setup_logger

class JarvisApplication:
    """
    Main JARVIS Application Controller
    
    Key Design Decisions:
    1. Async/await pattern for non-blocking operations
    2. Signal handling for graceful shutdown
    3. Centralized error handling and logging
    4. Modular architecture with dependency injection
    """
    
    def __init__(self):
        self.logger = setup_logger("jarvis_main")
        self.jarvis_core = None
        self.running = False
        
    async def initialize(self):
        """Initialize all JARVIS components"""
        try:
            self.logger.info("🚀 Initializing JARVIS...")
            
            # Initialize core with dependency injection pattern
            self.jarvis_core = JarvisCore()
            await self.jarvis_core.initialize()
            
            # Setup signal handlers for graceful shutdown
            self._setup_signal_handlers()
            
            self.logger.info("✅ JARVIS initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize JARVIS: {e}")
            return False
    
    def _setup_signal_handlers(self):
        """Setup graceful shutdown on SIGINT/SIGTERM"""
        def signal_handler(signum, frame):
            self.logger.info(f"📡 Received signal {signum}, shutting down...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def run(self):
        """Main application loop"""
        if not await self.initialize():
            sys.exit(1)
        
        self.running = True
        self.logger.info("🎯 JARVIS is now active and listening...")
        
        try:
            # Start background tasks concurrently
            tasks = [
                self.jarvis_core.start_voice_listening(),
                self.jarvis_core.start_web_interface(),
                self.jarvis_core.start_background_monitoring(),
                self._health_check_loop()
            ]
            
            # Run all tasks concurrently
            await asyncio.gather(*tasks)
            
        except Exception as e:
            self.logger.error(f"💥 Critical error in main loop: {e}")
            await self.shutdown()
    
    async def _health_check_loop(self):
        """Monitor system health - Important for production systems"""
        while self.running:
            try:
                await self.jarvis_core.health_check()
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                self.logger.warning(f"⚠️ Health check failed: {e}")
    
    async def shutdown(self):
        """Graceful shutdown - Critical for preventing data loss"""
        self.logger.info("🔄 Shutting down JARVIS...")
        self.running = False
        
        if self.jarvis_core:
            await self.jarvis_core.shutdown()
        
        self.logger.info("👋 JARVIS shutdown complete")
        sys.exit(0)

def main():
    """Entry point - Why async main is important for modern Python"""
    app = JarvisApplication()
    
    try:
        # Use asyncio.run() for proper event loop management
        asyncio.run(app.run())
    except KeyboardInterrupt:
        print("\n👋 JARVIS terminated by user")
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

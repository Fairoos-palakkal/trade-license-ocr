"""
Simple script to run the FastAPI application.
This makes starting the server easier for beginners.
"""

import uvicorn

if __name__ == "__main__":
    """
    Run the FastAPI application with uvicorn.
    
    Parameters explained:
    - "app.main:app" = Module path to FastAPI instance
      * "app.main" = File location (app/main.py)
      * ":app" = Variable name (app = FastAPI())
    
    - host="0.0.0.0" = Listen on all network interfaces
      * "0.0.0.0" means accessible from localhost AND your local network
      * "127.0.0.1" would mean localhost only
    
    - port=8000 = Server runs on port 8000
    
    - reload=True = Auto-restart when code changes (DEVELOPMENT ONLY)
      * DO NOT use in production
      * Great for development - save file = instant reload
    
    - log_level="info" = Show informational messages
      * Options: "critical", "error", "warning", "info", "debug"
    """
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

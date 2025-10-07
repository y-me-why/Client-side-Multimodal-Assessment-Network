import os
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from routes import register_routes, ConnectionManager
import json
import asyncio
import logging

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

app = FastAPI(title="AI Interview Prep AI Services")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize connection manager for WebSocket connections
manager = ConnectionManager()

# WebSocket endpoint for live analysis
@app.websocket("/ws/live-analysis/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    try:
        await manager.connect(websocket, session_id)
        logger.info(f"WebSocket connected successfully for session {session_id}")
        
        while True:
            try:
                # Receive data from client (video/audio frames)
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Process the analysis request
                response = await manager.process_live_analysis(message, session_id)
                
                # Send results back to client
                await manager.send_analysis_result(response, session_id)
                
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON received from session {session_id}: {str(e)}")
                error_response = {
                    "type": "error", 
                    "session_id": session_id,
                    "error": "Invalid JSON format"
                }
                await websocket.send_text(json.dumps(error_response))
            except Exception as e:
                logger.error(f"Error processing message for session {session_id}: {str(e)}")
                error_response = {
                    "type": "error", 
                    "session_id": session_id,
                    "error": f"Processing error: {str(e)}"
                }
                await websocket.send_text(json.dumps(error_response))
                
    except WebSocketDisconnect:
        logger.info(f"Client disconnected from session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {str(e)}")
    finally:
        manager.disconnect(websocket, session_id)

register_routes(app)

# Add a simple WebSocket test endpoint
@app.websocket("/ws/test")
async def websocket_test_endpoint(websocket: WebSocket):
    """Simple WebSocket test endpoint to verify connectivity."""
    await websocket.accept()
    logger.info("Test WebSocket connection established")
    try:
        while True:
            data = await websocket.receive_text()
            logger.info(f"Test WebSocket received: {data}")
            await websocket.send_text(f"Echo: {data}")
    except WebSocketDisconnect:
        logger.info("Test WebSocket disconnected")
    except Exception as e:
        logger.error(f"Test WebSocket error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Enable more detailed logging for debugging
    logger.info("Starting AI Services with WebSocket support...")
    logger.info(f"WebSocket libraries installed: websockets, wsproto")
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("AI_SVC_PORT", 8001)), log_level="info")

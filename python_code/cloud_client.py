import asyncio
import websockets
import json
import base64

class ElevenLabsClient:
    def __init__(self, api_key, sample_rate=16000):
        self.api_key = api_key
        self.sample_rate = sample_rate
        self.ws_url = f"wss://api.elevenlabs.io/v1/speech-to-text/stream?api_key={self.api_key}"
        self.websocket = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.close()

    async def connect(self):
        print(f"Connecting to {self.ws_url}...")
        self.websocket = await websockets.connect(self.ws_url)
        print("Connected.")
        
        # Send Initial Configuration
        await self.websocket.send(json.dumps({
            "model_id": "scribe_v2", 
            "language_code": "eng",
            "sample_rate": self.sample_rate
        }))
        return self.websocket

    async def close(self):
        if self.websocket:
            await self.websocket.close()
            print("Connection closed.")

    async def send_audio(self, audio_data):
        """
        Sends a chunk of audio data to the server.
        audio_data: bytes
        """
        if not self.websocket:
            raise Exception("Not connected")
        
        payload = {
            "audio_event": {
                "audio_base_64": base64.b64encode(audio_data).decode("utf-8"),
            }
        }
        await self.websocket.send(json.dumps(payload))

    async def receive(self):
        """
        Async generator that yields responses from the server.
        """
        if not self.websocket:
            raise Exception("Not connected")
        
        async for message in self.websocket:
            yield json.loads(message)

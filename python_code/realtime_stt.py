import asyncio
import pyaudio
import os
from cloud_client import ElevenLabsClient

# Configuration
API_KEY = os.getenv("ELEVENLABS_API_KEY", "YOUR_ELEVENLABS_API_KEY")
RATE = 16000
CHUNK = 2048

async def main():
    if API_KEY == "YOUR_ELEVENLABS_API_KEY":
        print("⚠️  Please set your ElevenLabs API Key in the script or environment variable 'ELEVENLABS_API_KEY'.")

    client = ElevenLabsClient(API_KEY, RATE)
    
    try:
        async with client:
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
            print("🎤 Listening... (Ctrl+C to stop)")

            async def sender():
                while True:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    await client.send_audio(data)
                    await asyncio.sleep(0.001)

            async def receiver():
                async for response in client.receive():
                    if response.get('type') == 'partial_transcription':
                        text = response['partial_transcription_event']['partial_transcript']
                        if text:
                            # Overwrite the current line for partial updates
                            print(f"\r> {text}", end="", flush=True)
                    elif response.get('type') == 'transcription':
                         # Final transcription
                         text = response['transcription_event']['transcript']
                         print(f"\r✅ {text}") 

            await asyncio.gather(sender(), receiver())
            
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        if 'p' in locals():
            p.terminate()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Stopped.")

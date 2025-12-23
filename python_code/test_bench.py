import asyncio
import pyaudio
import time
import os
import statistics
from cloud_client import ElevenLabsClient

# Configuration
# You can set this env var or replace the string below
API_KEY = os.getenv("ELEVENLABS_API_KEY", "YOUR_ELEVENLABS_API_KEY")
RATE = 16000
CHUNK = 2048

async def run_test_bench():
    if API_KEY == "YOUR_ELEVENLABS_API_KEY":
        print("⚠️  Please set your ElevenLabs API Key in the script or environment variable 'ELEVENLABS_API_KEY'.")
        # We continue to allow the user to see the setup, but it will fail on connection if key is invalid.
    
    print(f"Initializing Client with Rate={RATE}, Chunk={CHUNK}...")
    client = ElevenLabsClient(API_KEY, RATE)
    
    # Shared state for metrics
    state = {
        "last_send_time": 0,
        "latencies": []
    }

    try:
        async with client:
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
            
            print("🎤 Streaming... Speak now! (Ctrl+C to stop)")
            print(f"{'TRANSCRIPT':<40} | {'LATENCY':<10} | {'AVG(10)':<10} | {'JITTER':<10}")
            print("-" * 80)

            async def sender():
                """Captures mic audio and sends to Cloud"""
                try:
                    while True:
                        data = stream.read(CHUNK, exception_on_overflow=False)
                        state["last_send_time"] = time.time()  # Mark time sent
                        await client.send_audio(data)
                        await asyncio.sleep(0.001) # Yield to event loop
                except Exception as e:
                    print(f"Sender Error: {e}")

            async def receiver():
                """Listens for Transcripts from Cloud and calculates metrics"""
                try:
                    async for response in client.receive():
                        recv_time = time.time()
                        
                        # Check for transcription events
                        if response.get('type') == 'partial_transcription':
                            text = response['partial_transcription_event']['partial_transcript']
                            if text:
                                # ROUGH Latency calc: Time Received - Time of last chunk sent
                                latency_ms = (recv_time - state["last_send_time"]) * 1000
                                state["latencies"].append(latency_ms)
                                
                                # Calculate stats
                                recent_latencies = state["latencies"][-10:]
                                avg_latency = statistics.mean(recent_latencies) if recent_latencies else 0
                                jitter = statistics.stdev(recent_latencies) if len(recent_latencies) > 1 else 0
                                
                                print(f"{text:<40} | {latency_ms:4.0f}ms    | {avg_latency:4.0f}ms    | {jitter:4.0f}ms")
                                
                except Exception as e:
                    print(f"Receiver Error: {e}")

            # Run both tasks concurrently
            await asyncio.gather(sender(), receiver())

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'p' in locals():
            p.terminate()

if __name__ == "__main__":
    try:
        asyncio.run(run_test_bench())
    except KeyboardInterrupt:
        print("\n🛑 Stopped.")

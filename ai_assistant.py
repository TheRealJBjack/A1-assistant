#!/usr/bin/env python3
"""
AI assistant with audio and optional video/screen input.

This script implements a multimodal AI assistant that can interact with the user
through real-time audio. It can also optionally capture and process input from
the camera or screen.

Key features:
- Real-time audio conversation.
- Optional camera input for visual context.
- Optional screen capture for sharing screen content.
- Integration with Google Search for information retrieval.
- Function calling to launch predefined applications (e.g., web browsers, calculator).

Dependencies:
To run this script, you need to install the following Python packages:
  pip install google-genai opencv-python pyaudio pillow mss
You also need to have a GEMINI_API_KEY environment variable set.
"""
import os
import asyncio
import base64
import io
import subprocess
import traceback

import cv2
import pyaudio
import PIL.Image
import mss

import argparse

from google import genai
from google.genai import types

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL = "models/gemini-2.5-flash-preview-native-audio-dialog"

DEFAULT_MODE = "camera"

# Predefined list of allowed programs for launching.
# Keys are user-friendly names/aliases, values are executable paths.
# IMPORTANT: For security, only add trusted applications.
ALLOWED_PROGRAMS = {
    "firefox": "firefox.exe",  # Example for Windows
    "chrome": "chrome.exe",    # Example for Windows
    "calculator": "calc.exe",  # Example for Windows
    # Add more aliases and cross-platform names as needed
    # e.g., "firefox": "firefox" on Linux
}

client = genai.Client(
    http_options={"api_version": "v1beta"},
    api_key=os.environ.get("GEMINI_API_KEY"),
)

tools = [
    types.Tool(google_search=types.GoogleSearch()),
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="launch_application",
                description="Launches a specified application on the user's computer. Only a predefined list of applications are supported.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "program_name": types.Schema(type=types.Type.STRING, description="The executable name of the program to launch (e.g., 'firefox.exe', 'chrome.exe').")
                    },
                    required=["program_name"]
                )
            )
        ]
    ),
]

CONFIG = types.LiveConnectConfig(
    response_modalities=[
        "AUDIO",
    ],
    media_resolution="MEDIA_RESOLUTION_MEDIUM",
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Zephyr")
        )
    ),
    realtime_input_config=types.RealtimeInputConfig(turn_coverage="TURN_INCLUDES_ALL_INPUT"),
    context_window_compression=types.ContextWindowCompressionConfig(
        trigger_tokens=25600,
        sliding_window=types.SlidingWindow(target_tokens=12800),
    ),
    tools=tools,
    system_instruction=types.Content(
        parts=[types.Part.from_text(text="You are a helpful AI assistant. You can engage in spoken dialogue and have access to Google Search and other tools to assist the user.")],
        role="user"
    ),
)

pya = pyaudio.PyAudio()


class AudioLoop:
    def __init__(self, video_mode=DEFAULT_MODE):
        self.video_mode = video_mode

        self.audio_in_queue = None
        self.out_queue = None

        self.session = None

        self.send_text_task = None
        self.receive_audio_task = None
        self.play_audio_task = None

    async def send_text(self):
        while True:
            text = await asyncio.to_thread(
                input,
                "message > ",
            )
            if text.lower() == "q":
                break
            await self.session.send(input=text or ".", end_of_turn=True)

    def _get_frame(self, cap):
        # Read the frameq
        ret, frame = cap.read()
        # Check if the frame was read successfully
        if not ret:
            return None
        # Fix: Convert BGR to RGB color space
        # OpenCV captures in BGR but PIL expects RGB format
        # This prevents the blue tint in the video feed
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)  # Now using RGB frame
        img.thumbnail([1024, 1024])

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        mime_type = "image/jpeg"
        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_frames(self):
        # This takes about a second, and will block the whole program
        # causing the audio pipeline to overflow if you don't to_thread it.
        cap = await asyncio.to_thread(
            cv2.VideoCapture, 0
        )  # 0 represents the default camera

        while True:
            frame = await asyncio.to_thread(self._get_frame, cap)
            if frame is None:
                break

            await asyncio.sleep(1.0)

            await self.out_queue.put(frame)

        # Release the VideoCapture object
        cap.release()

    def _get_screen(self):
        sct = mss.mss()
        monitor = sct.monitors[0]

        i = sct.grab(monitor)

        mime_type = "image/jpeg"
        image_bytes = mss.tools.to_png(i.rgb, i.size)
        img = PIL.Image.open(io.BytesIO(image_bytes))

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_screen(self):

        while True:
            frame = await asyncio.to_thread(self._get_screen)
            if frame is None:
                break

            await asyncio.sleep(1.0)

            await self.out_queue.put(frame)

    async def send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            await self.session.send(input=msg)

    async def listen_audio(self):
        mic_info = pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        if __debug__:
            kwargs = {"exception_on_overflow": False}
        else:
            kwargs = {}
        while True:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def handle_function_call(self, function_call):
        """Processes a function call request from the AI.

        Currently supports 'launch_application'. Sends a FunctionResponse
        back to the AI indicating the outcome.
        """
        if function_call.name == "launch_application":
            program_name_input = function_call.args["program_name"].lower()
            
            # Normalize input, e.g. "firefox browser" -> "firefox"
            # This is a simple example; more robust normalization might be needed.
            executable_name = None
            for app_alias, app_exe in ALLOWED_PROGRAMS.items():
                if app_alias in program_name_input:
                    executable_name = app_exe
                    break
            
            if executable_name:
                try:
                    print(f"\n[Attempting to launch: {executable_name}]")
                    subprocess.Popen([executable_name])
                    response_content = f"Successfully launched {executable_name}."
                    print(f"[Launched {executable_name}]")
                except FileNotFoundError:
                    response_content = f"Error: The application '{executable_name}' was not found."
                    print(f"[Error launching {executable_name}: Not found]")
                except Exception as e:
                    response_content = f"An error occurred while trying to launch {executable_name}: {e}"
                    print(f"[Error launching {executable_name}: {e}]")
            else:
                response_content = f"Error: Program '{program_name_input}' is not on the allowed list or is not recognized."
                print(f"[Program '{program_name_input}' not allowed or recognized]")

            # Send response back to the model
            await self.session.send(
                input=[types.Part(
                    function_response=types.FunctionResponse(
                        name="launch_application",
                        response={"content": response_content}
                    )
                )],
                end_of_turn=False # Or True, depending on desired interaction flow
            )
        else:
            # Handle other function calls if any, or send a generic "not implemented"
            await self.session.send(
                input=[types.Part(
                    function_response=types.FunctionResponse(
                        name=function_call.name,
                        response={"content": f"Function {function_call.name} is not implemented."}
                    )
                )],
                end_of_turn=False
            )

    async def receive_audio(self):
        "Background task to reads from the websocket and write pcm chunks to the output queue"
        while True:
            turn = self.session.receive()
            async for response in turn:
                if data := response.data:
                    self.audio_in_queue.put_nowait(data)
                    continue
                if text := response.text:
                    print(text, end="")
                
                # Check for function call from the AI
                if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if part.function_call: # AI is requesting a function call
                            print(f"\n[Function Call: {part.function_call.name} with args: {part.function_call.args}]") 
                            await self.handle_function_call(part.function_call)
                            # The function handler will send a response back to the model.
                            # We then break from processing further parts in this response and wait for the next turn.
                            break 
                    else: # else for the for loop, if no function_call part was found and `break` was not hit
                        continue # continue to the next response in `turn`
                    break # break from iterating `async for response in turn` because a function call was handled.

            # If you interrupt the model, it sends a turn_complete.
            # For interruptions to work, we need to stop playback.
            # So empty out the audio queue because it may have loaded
            # much more audio than has played yet.
            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()

    async def play_audio(self):
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            bytestream = await self.audio_in_queue.get()
            await asyncio.to_thread(stream.write, bytestream)

    async def run(self):
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session

                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)

                send_text_task = tg.create_task(self.send_text())
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                if self.video_mode == "camera":
                    tg.create_task(self.get_frames())
                elif self.video_mode == "screen":
                    tg.create_task(self.get_screen())

                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())

                await send_text_task
                raise asyncio.CancelledError("User requested exit")

        except asyncio.CancelledError:
            pass
        except ExceptionGroup as EG:
            self.audio_stream.close()
            traceback.print_exception(EG)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        help="pixels to stream from",
        choices=["camera", "screen", "none"],
    )
    args = parser.parse_args()
    main = AudioLoop(video_mode=args.mode)
    asyncio.run(main.run())

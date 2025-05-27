# AI Assistant with Multimodal Capabilities

This project implements a Python-based AI assistant that can engage in real-time audio conversations and optionally process visual input from the camera or screen. It uses the Google Gemini API for its conversational AI capabilities.

## Features

-   Real-time audio interaction (voice input and output).
-   Function calling to launch predefined local applications (e.g., web browsers, calculator).
-   Optional camera input for sending visual context to the AI.
-   Optional screen capture for sharing screen content with the AI.
-   Integration with Google Search for information retrieval.
-   System instructions to guide the AI's persona and behavior.

## Setup Instructions

Follow these steps to set up and run the AI assistant:

1.  **Set API Key**:
    You need a Gemini API key. Set it as an environment variable named `GEMINI_API_KEY`.
    ```bash
    export GEMINI_API_KEY="YOUR_API_KEY_HERE" 
    ```
    (On Windows, use `set GEMINI_API_KEY=YOUR_API_KEY_HERE` in Command Prompt or set it through System Properties.)

2.  **Install Dependencies**:
    A Python script is provided to install all necessary dependencies. Run it using:
    ```bash
    python install_dependencies.py
    ```
    This will install `google-genai`, `opencv-python`, `pyaudio`, `pillow`, and `mss`.

3.  **Run the Assistant**:
    Once the dependencies are installed and the API key is set, you can run the assistant:
    ```bash
    python ai_assistant.py
    ```
    You can also run it with different input modes:
    -   `python ai_assistant.py --mode screen` (to enable screen sharing)
    -   `python ai_assistant.py --mode none` (for audio-only interaction)
    -   `python ai_assistant.py --mode camera` (default, to use camera input)

## How It Works

The assistant uses:
-   **PyAudio** for capturing microphone input and playing back audio responses.
-   **OpenCV** and **Pillow** for camera input processing.
-   **MSS** for screen capture.
-   **Google GenAI SDK** for connecting to and interacting with the Gemini model, handling audio streaming, function calls, and text-to-speech.

## Program Launching

The assistant can launch a predefined set of applications using function calling. The current allowlist includes:
-   Firefox
-   Chrome
-   Calculator
This list is defined in `ai_assistant.py` in the `ALLOWED_PROGRAMS` variable and can be modified carefully.

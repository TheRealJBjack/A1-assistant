# This script installs the necessary dependencies for the AI Assistant.
# To run it, execute: python install_dependencies.py
import subprocess
import sys

# List of dependencies to install
dependencies = [
    "google-genai",
    "opencv-python",
    "pyaudio",
    "pillow",
    "mss"
]

def install_packages(packages):
    """
    Installs the given list of Python packages using pip.
    """
    print(f"Attempting to install: {', '.join(packages)}")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *packages])
        print("\nSuccessfully installed all required dependencies.")
    except subprocess.CalledProcessError as e:
        print(f"\nError during installation: {e}")
        print("Please try installing the packages manually.")
    except FileNotFoundError:
        print("\nError: 'pip' command not found. Make sure pip is installed and in your PATH.")
        print("You might need to install pip first, or use 'python -m pip install ...'")

if __name__ == "__main__":
    print("----------------------------------------------------")
    print("Starting dependency installation for AI Assistant...")
    print("----------------------------------------------------")
    install_packages(dependencies)
    print("\n----------------------------------------------------")
    print("Setup script finished.")
    print("Next steps:")
    print("1. Ensure you have set your GEMINI_API_KEY environment variable.")
    print("2. Run the assistant using: python ai_assistant.py")
    print("----------------------------------------------------")

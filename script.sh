#!/bin/bash

# FLUX.1 [schnell] FastAPI Text-to-Image Generation API Setup Script

echo "Starting setup for FLUX.1 [schnell] FastAPI Text-to-Image Generation API..."

# 1. Prerequisites Check (Basic)
echo "Checking for Python 3.8+..."
if ! command -v python3 &> /dev/null || ! python3 -c 'import sys; exit(sys.version_info.major < 3 or sys.version_info.minor < 8)' &> /dev/null; then
    echo "Python 3.8 or later is required. Please install it first."
    exit 1
fi
echo "Python 3.8+ detected."

# 2. Clone the Repository (Assuming current directory is where the repo should be cloned)
# The user needs to provide their repo URL
read -p "Enter your repository URL (e.g., https://github.com/your_username/your_repo.git): " REPO_URL
if [ -z "$REPO_URL" ]; then
    echo "Repository URL cannot be empty. Exiting."
    exit 1
fi

REPO_FOLDER=$(basename "$REPO_URL" .git)

echo "Cloning the repository from $REPO_URL..."
if git clone "$REPO_URL"; then
    echo "Repository cloned successfully."
    cd "$REPO_FOLDER" || { echo "Failed to change directory to $REPO_FOLDER. Exiting."; exit 1; }
else
    echo "Failed to clone the repository. Please check the URL and your internet connection. Exiting."
    exit 1
fi

# 3. Create and Activate Python Virtual Environment
echo "Creating and activating Python virtual environment..."
python3 -m venv venv
if [ -f venv/bin/activate ]; then
    source venv/bin/activate
    echo "Virtual environment activated (Linux/macOS)."
elif [ -f venv/Scripts/activate ]; then
    source venv/Scripts/activate
    echo "Virtual environment activated (Windows)."
else
    echo "Could not find virtual environment activation script. Please activate it manually: source venv/bin/activate or venv\\Scripts\\activate."
    # We will still proceed, but the user might need to activate manually for subsequent commands
fi

# 4. Install Dependencies
echo "Installing Python dependencies..."
pip install fastapi uvicorn torch diffusers huggingface_hub accelerate sentencepiece protobuf transformers
if [ $? -eq 0 ]; then
    echo "All dependencies installed successfully."
else
    echo "Failed to install some dependencies. Please check the error messages above. Exiting."
    exit 1
fi

# 5. Set Your Hugging Face Token
echo "Setting up Hugging Face Token..."
read -p "Do you want to set your Hugging Face token as an environment variable (recommended)? (y/n): " SET_ENV_VAR
if [[ "$SET_ENV_VAR" =~ ^[Yy]$ ]]; then
    read -sp "Enter your Hugging Face token: " HF_TOKEN_INPUT
    echo
    if [ -z "$HF_TOKEN_INPUT" ]; then
        echo "Hugging Face token cannot be empty. Skipping environment variable setup."
    else
        export HF_TOKEN="$HF_TOKEN_INPUT"
        echo "HF_TOKEN environment variable set for this session."
    fi
else
    echo "Skipping environment variable setup for HF_TOKEN."
    echo "Remember to set HF_TOKEN in your app.py file if you haven't set it as an environment variable."
fi

# 6. Run the API Server
echo "Attempting to run the API server..."
echo "The server will start at http://0.0.0.0:8000. Press Ctrl+C to stop it."

# Assuming app.py is in the root of the cloned repository
if [ -f "app.py" ]; then
    python app.py
else
    echo "Error: app.py not found in the current directory ($PWD). Please ensure you are in the correct repository folder."
    echo "You might need to navigate into the cloned repository directory: cd $REPO_FOLDER"
    echo "Then run 'python app.py' manually."
fi

echo "Setup script finished."
echo "You can now use the API by sending POST requests to http://0.0.0.0:8000/generate-image"
echo "Example JSON payload:"
echo '{'
echo '  "prompt": "A cinematic portrait of a woman with sharp green eyes, photorealistic, 8K",'
echo '  "guidance_scale": 7.5,'
echo '  "num_inference_steps": 20,'
echo '  "max_sequence_length": 77,'
echo '  "seed": 42'
echo '}'

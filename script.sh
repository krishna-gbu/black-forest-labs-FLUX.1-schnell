#!/bin/bash

# FLUX.1 [schnell] FastAPI Text-to-Image Generation API Setup Script
# This script assumes you have ALREADY CLONED the repository
# and are currently INSIDE the 'black-forest-labs-FLUX.1-schnell' directory.

echo "Starting setup for FLUX.1 [schnell] FastAPI Text-to-Image Generation API..."
echo "Assuming you are currently in the 'black-forest-labs-FLUX.1-schnell' repository directory."

# 1. Prerequisites Check (Basic)
echo "Checking for Python 3.8+..."
if ! command -v python3 &> /dev/null || ! python3 -c 'import sys; exit(sys.version_info.major < 3 or sys.version_info.minor < 8)' &> /dev/null; then
    echo "Python 3.8 or later is required. Please install it first."
    exit 1
fi
echo "Python 3.8+ detected."

# 2. Create and Activate Python Virtual Environment
echo "Creating and activating Python virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment 'venv' already exists. Activating."
else
    python3 -m venv venv
fi

if [ -f venv/bin/activate ]; then
    source venv/bin/activate
    echo "Virtual environment activated (Linux/macOS)."
elif [ -f venv/Scripts/activate ]; then
    source venv/Scripts/activate
    echo "Virtual environment activated (Windows)."
else
    echo "Could not find virtual environment activation script. Please activate it manually: source venv/bin/activate or venv\\Scripts\\activate."
    echo "WARNING: Subsequent commands might fail if venv is not active."
fi

# 3. Install Dependencies
echo "Installing Python dependencies..."
pip install fastapi uvicorn torch diffusers huggingface_hub accelerate sentencepiece protobuf transformers
if [ $? -eq 0 ]; then
    echo "All dependencies installed successfully."
else
    echo "Failed to install some dependencies. Please check the error messages above. Exiting."
    exit 1
fi

# 4. Set Your Hugging Face Token
echo "Setting up Hugging Face Token..."
read -sp "Enter your Hugging Face token (it will not be displayed): " HF_TOKEN_INPUT
echo
if [ -z "$HF_TOKEN_INPUT" ]; then
    echo "Hugging Face token cannot be empty. Exiting."
    exit 1
fi

export HF_TOKEN="$HF_TOKEN_INPUT"
echo "HF_TOKEN environment variable set for this session."
echo "IMPORTANT: Ensure app.py uses os.getenv('HF_TOKEN') for authentication."

# 5. Check and (Optionally) Modify app.py to use HF_TOKEN from environment
echo "Checking app.py for token usage..."
if [ ! -f "app.py" ]; then
    echo "Error: app.py not found in the current directory ($PWD). Please ensure you are in the correct repository folder."
    exit 1
fi

# This heuristic attempts to modify app.py to use the environment variable
# if it finds a hardcoded token or a specific 'flux' token.
if grep -q "login(token=['\"]\(flux\|your_token_here\)['\"]\)" app.py; then
    echo "WARNING: app.py seems to hardcode a token or look for 'flux'."
    echo "It's highly recommended to modify app.py to use 'login(token=os.getenv(\"HF_TOKEN\"))'."
    read -p "Attempt to replace 'login(token=HF_TOKEN)' in app.py with a generic version using os.getenv? (y/n): " REPLACE_APP_PY_TOKEN
    if [[ "$REPLACE_APP_PY_TOKEN" =~ ^[Yy]$ ]]; then
        # Use a temporary file for sed to avoid issues and ensure atomicity
        sed -i.bak -E "s/login\(token=['\"]\([^'\"]*\)['\"]\)/import os\nlogin(token=os.getenv(\"HF_TOKEN\"))/" app.py
        if [ $? -eq 0 ]; then
            echo "Attempted to modify app.py. Original backed up as app.py.bak."
        else
            echo "Failed to modify app.py automatically. Please modify it manually."
        fi
    fi
else
    echo "app.py seems to handle tokens generically or expects an environment variable. No automatic modification needed."
fi


# 6. Run the API Server
echo "Attempting to run the API server..."
echo "The server will start at http://0.0.0.0:8000. Press Ctrl+C to stop it."

python app.py

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

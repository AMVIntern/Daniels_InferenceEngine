#!/bin/bash

# launch.sh - Opens two terminals for FastAPI + ngrok

gnome-terminal -- bash -c "uvicorn fill_estimator_api:app --host 0.0.0.0 --port 8000; exec bash" &
gnome-terminal -- bash -c "ngrok http 8000; exec bash" &
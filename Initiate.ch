#!/bin/bash
echo "Starting TaxiPred application..."

# Kill any existing Python processes
echo "Cleaning up existing Python processes..."
pkill -f python || true

# Clear Python cache directories
echo "Clearing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} +

# Store the original location
ORIGINAL_DIR=$(pwd)

# Start FastAPI backend in the background
echo "Starting FastAPI backend..."
(
  cd src/taxipred/backend || exit
  uvicorn api:app --reload --host 127.0.0.1 --port 8000
) &
FASTAPI_PID=$!

# Give backend time to start
sleep 3

# Check if FastAPI process is still running
if ps -p $FASTAPI_PID > /dev/null; then
  echo "FastAPI backend started successfully on http://127.0.0.1:8000"
  echo "API docs available at http://127.0.0.1:8000/docs"
else
  echo "Failed to start FastAPI backend"
  exit 1
fi

# Start Streamlit dashboard
echo "Starting Streamlit dashboard..."
streamlit run src/taxipred/frontend/dashboard.py --server.port 8501

# When Streamlit closes, clean up
echo "Stopping background processes..."
kill $FASTAPI_PID 2>/dev/null || true
pkill -f python || true

# Return to original location
cd "$ORIGINAL_DIR" || exit
echo "Cleanup completed."

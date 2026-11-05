#!/data/data/com.termux/files/usr/bin/bash
export PYTHONPATH=src
termux-wake-lock
echo "Starting TechBot..."
python -m techbot.main --run-index 1
termux-wake-unlock
echo "Done! You can close the terminal."

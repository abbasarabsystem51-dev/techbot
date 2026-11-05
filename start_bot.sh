#!/bin/bash
cd /mnt/e/tech_news_110/techbot/techbot
source ~/.venvs/techbot/bin/activate

# Kill existing session to prevent duplicates
tmux kill-session -t bot 2>/dev/null

# Start fresh session in background
tmux new-session -d -s bot "PYTHONPATH=src python -m techbot.main --serve"

echo "✅ Bot is now running in the background!"

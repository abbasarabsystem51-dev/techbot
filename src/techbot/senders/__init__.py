"""
src/techbot/senders package — Publishing destinations for Telegram and Eitaa.
"""
from src.techbot.senders.base import BaseSender, DeliveryResult
from src.techbot.senders.eitaa import EitaaSender
from src.techbot.senders.telegram import TelegramSender

__all__ = [
    "BaseSender",
    "DeliveryResult",
    "TelegramSender",
    "EitaaSender",
]

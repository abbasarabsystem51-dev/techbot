"""
src/techbot/senders/base.py — Abstract Base Class for destination publishing senders.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class DeliveryResult:
    destination: str
    success: bool
    message_id: Optional[str] = None
    status: str = "sent"
    error: Optional[str] = None


class BaseSender(ABC):
    """
    Abstract publishing destination sender.
    """

    def __init__(self, destination_name: str, enabled: bool = True) -> None:
        self.destination_name = destination_name
        self.enabled = enabled

    @abstractmethod
    async def send_post(
        self,
        text: str,
        image_bytes: Optional[bytes] = None,
        image_url: Optional[str] = None,
        source_url: Optional[str] = None,
        quiz: Optional[dict[str, Any]] = None,
    ) -> DeliveryResult:
        """
        Send a post to the destination platform.
        """
        raise NotImplementedError

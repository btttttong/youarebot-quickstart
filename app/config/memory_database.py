"""
In-memory database replacement for PostgreSQL
Stores messages in memory - data will be lost when service restarts
"""

from typing import List, Dict
from datetime import datetime
import threading

# Thread-safe in-memory storage
_messages_store = []
_store_lock = threading.Lock()


def get_db_connection():
    """
    Dummy function for compatibility - returns None since we don't use connections
    """
    return None


def init_db() -> None:
    """
    Initialize in-memory database (no-op since we use list)
    """
    with _store_lock:
        # Clear any existing data on init
        _messages_store.clear()
    print("✅ In-memory database initialized")


def insert_message(
        id,  # UUID
        text: str,
        dialog_id,  # UUID
        participant_index: int
) -> None:
    """
    Stores one message in memory
    """
    with _store_lock:
        message = {
            "id": str(id),
            "text": text,
            "dialog_id": str(dialog_id),
            "participant_index": participant_index,
            "created_at": datetime.now()
        }
        _messages_store.append(message)


def select_messages_by_dialog(dialog_id) -> List[Dict[str, str]]:
    """
    Returns list of messages for current dialog_id ordered by time
    """
    with _store_lock:
        dialog_messages = [
            msg for msg in _messages_store 
            if msg["dialog_id"] == str(dialog_id)
        ]
        
        # Sort by created_at timestamp
        dialog_messages.sort(key=lambda x: x["created_at"])
        
        return [
            {"text": msg["text"], "participant_index": msg["participant_index"]}
            for msg in dialog_messages
        ]


def get_memory_stats():
    """
    Get statistics about in-memory storage (for debugging)
    """
    with _store_lock:
        total_messages = len(_messages_store)
        unique_dialogs = len(set(msg["dialog_id"] for msg in _messages_store))
        
        return {
            "total_messages": total_messages,
            "unique_dialogs": unique_dialogs,
            "storage_type": "in-memory"
        }
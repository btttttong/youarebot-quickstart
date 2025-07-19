"""
Database wrapper that switches between PostgreSQL and in-memory storage
"""

from app.config.config import USE_MEMORY_DB

if USE_MEMORY_DB:
    from app.config.memory_database import (
        get_db_connection,
        init_db,
        insert_message,
        select_messages_by_dialog
    )
else:
    from app.config.database import (
        get_db_connection,
        init_db,
        insert_message,
        select_messages_by_dialog
    )

# Export the functions so they can be imported as before
__all__ = [
    'get_db_connection',
    'init_db', 
    'insert_message',
    'select_messages_by_dialog'
]
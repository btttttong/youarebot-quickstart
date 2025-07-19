#!/usr/bin/env python3
"""Test the memory database functionality."""

import os
import uuid

# Set environment variable to use memory database
os.environ["USE_MEMORY_DB"] = "true"

from app.config.database_wrapper import init_db, insert_message, select_messages_by_dialog

def test_memory_database():
    """Test in-memory database functionality."""
    print("🧪 Testing in-memory database...")
    
    # Initialize database
    init_db()
    print("✅ Database initialized")
    
    # Create test data
    dialog_id = uuid.uuid4()
    message_id_1 = uuid.uuid4()
    message_id_2 = uuid.uuid4()
    
    # Insert messages
    insert_message(message_id_1, "Hello, are you a bot?", dialog_id, 0)
    insert_message(message_id_2, "No, I'm human!", dialog_id, 1)
    print("✅ Messages inserted")
    
    # Retrieve messages
    messages = select_messages_by_dialog(dialog_id)
    print(f"✅ Retrieved {len(messages)} messages")
    
    # Verify data
    expected_messages = [
        {"text": "Hello, are you a bot?", "participant_index": 0},
        {"text": "No, I'm human!", "participant_index": 1}
    ]
    
    if messages == expected_messages:
        print("✅ Memory database test passed!")
        return True
    else:
        print(f"❌ Test failed. Expected: {expected_messages}, Got: {messages}")
        return False

if __name__ == "__main__":
    success = test_memory_database()
    if success:
        print("\n🎉 Memory database is working correctly!")
    else:
        print("\n💥 Memory database test failed!")
        exit(1)
import os
import aiosqlite
from typing import List, Dict, Optional
from datetime import datetime


class SessionStore:
    """SQLite-based session and message storage."""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the session store.
        
        Args:
            db_path: Path to SQLite database file (defaults to DATABASE_URL env var)
        """
        if db_path is None:
            db_url = os.getenv("DATABASE_URL", "sqlite:///./chat.db")
            # Remove sqlite:/// prefix if present
            db_path = db_url.replace("sqlite:///", "").replace("sqlite://", "")
        
        self.db_path = db_path
        self.db = None
    
    async def initialize(self):
        """Initialize database and create tables if they don't exist."""
        self.db = await aiosqlite.connect(self.db_path)
        
        # Create sessions table
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create messages table
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
        """)
        
        # Create index on session_id for faster queries
        await self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_session_id
            ON messages(session_id)
        """)
        
        await self.db.commit()
        print(f"✅ Database initialized at {self.db_path}")
    
    async def close(self):
        """Close database connection."""
        if self.db:
            await self.db.close()
    
    async def create_session(self, session_id: str):
        """
        Create a new session.
        
        Args:
            session_id: Unique session identifier
        """
        await self.db.execute(
            "INSERT OR IGNORE INTO sessions (id) VALUES (?)",
            (session_id,)
        )
        await self.db.commit()
    
    async def save_message(self, session_id: str, role: str, content: str):
        """
        Save a message to the database.
        
        Args:
            session_id: Session identifier
            role: Message role (user/assistant/system)
            content: Message content
        """
        # Ensure session exists
        await self.create_session(session_id)
        
        await self.db.execute(
            """
            INSERT INTO messages (session_id, role, content)
            VALUES (?, ?, ?)
            """,
            (session_id, role, content)
        )
        await self.db.commit()
    
    async def get_session_history(self, session_id: str) -> List[Dict[str, str]]:
        """
        Retrieve chat history for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of messages with role and content
        """
        cursor = await self.db.execute(
            """
            SELECT role, content, timestamp
            FROM messages
            WHERE session_id = ?
            ORDER BY timestamp ASC
            """,
            (session_id,)
        )
        
        rows = await cursor.fetchall()
        
        return [
            {
                "role": row[0],
                "content": row[1],
                "timestamp": row[2]
            }
            for row in rows
        ]
    
    async def delete_session(self, session_id: str):
        """
        Delete a session and all its messages.
        
        Args:
            session_id: Session identifier
        """
        await self.db.execute(
            "DELETE FROM messages WHERE session_id = ?",
            (session_id,)
        )
        await self.db.execute(
            "DELETE FROM sessions WHERE id = ?",
            (session_id,)
        )
        await self.db.commit()
    
    async def get_all_sessions(self) -> List[Dict[str, str]]:
        """
        Get all sessions.
        
        Returns:
            List of sessions with id and created_at
        """
        cursor = await self.db.execute(
            "SELECT id, created_at FROM sessions ORDER BY created_at DESC"
        )
        
        rows = await cursor.fetchall()
        
        return [
            {
                "id": row[0],
                "created_at": row[1]
            }
            for row in rows
        ]


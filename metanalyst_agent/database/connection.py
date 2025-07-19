"""
Database connection manager for metanalyst-agent.

Provides unified access to PostgreSQL for both LangGraph checkpointers/stores
and application-specific database operations.
"""

import os
import asyncio
from typing import Optional, Dict, Any
from contextlib import contextmanager, asynccontextmanager
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Centralized database manager for metanalyst-agent.
    
    Provides:
    - Connection pooling
    - LangGraph checkpointer and store instances
    - Direct database access for application queries
    - Health checks and monitoring
    """
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self._checkpointer: Optional[PostgresSaver] = None
        self._store: Optional[PostgresStore] = None
        self._connection_pool = None
        
    @property
    def checkpointer(self) -> PostgresSaver:
        """Get LangGraph PostgreSQL checkpointer (short-term memory)."""
        if self._checkpointer is None:
            self._checkpointer = PostgresSaver.from_conn_string(self.database_url)
            logger.info("PostgreSQL checkpointer initialized")
        return self._checkpointer
    
    @property 
    def store(self) -> PostgresStore:
        """Get LangGraph PostgreSQL store (long-term memory)."""
        if self._store is None:
            self._store = PostgresStore.from_conn_string(self.database_url)
            logger.info("PostgreSQL store initialized")
        return self._store
    
    @contextmanager
    def get_connection(self):
        """Get a direct PostgreSQL connection for application queries."""
        conn = None
        try:
            conn = psycopg2.connect(
                self.database_url,
                cursor_factory=RealDictCursor
            )
            conn.autocommit = False
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    @contextmanager
    def get_cursor(self):
        """Get a cursor with automatic connection management."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                yield cursor, conn
            finally:
                cursor.close()
    
    def execute_query(self, query: str, params: tuple = None) -> list:
        """Execute a SELECT query and return results."""
        with self.get_cursor() as (cursor, conn):
            cursor.execute(query, params)
            return cursor.fetchall()
    
    def execute_command(self, command: str, params: tuple = None) -> int:
        """Execute an INSERT/UPDATE/DELETE command and return affected rows."""
        with self.get_cursor() as (cursor, conn):
            cursor.execute(command, params)
            affected_rows = cursor.rowcount
            conn.commit()
            return affected_rows
    
    def execute_script(self, script: str) -> None:
        """Execute a SQL script (multiple statements)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(script)
                conn.commit()
                logger.info("SQL script executed successfully")
            except Exception as e:
                conn.rollback()
                logger.error(f"Error executing SQL script: {e}")
                raise
            finally:
                cursor.close()
    
    def health_check(self) -> Dict[str, Any]:
        """Perform database health check."""
        try:
            with self.get_cursor() as (cursor, conn):
                # Test basic connectivity
                cursor.execute("SELECT 1 as test")
                result = cursor.fetchone()
                
                # Get database stats
                cursor.execute("""
                    SELECT 
                        pg_database_size(current_database()) as db_size_bytes,
                        (SELECT count(*) FROM checkpoints) as checkpoint_count,
                        (SELECT count(*) FROM store) as store_count,
                        (SELECT count(*) FROM meta_analyses) as meta_analyses_count
                """)
                stats = cursor.fetchone()
                
                return {
                    "status": "healthy",
                    "connection": "ok" if result["test"] == 1 else "failed",
                    "database_size_mb": round(stats["db_size_bytes"] / (1024 * 1024), 2),
                    "checkpoint_count": stats["checkpoint_count"],
                    "store_count": stats["store_count"],
                    "meta_analyses_count": stats["meta_analyses_count"]
                }
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def get_meta_analysis_stats(self) -> list:
        """Get statistics for all meta-analyses."""
        query = "SELECT * FROM meta_analysis_stats ORDER BY created_at DESC"
        return self.execute_query(query)
    
    def get_agent_performance(self) -> list:
        """Get agent performance statistics."""
        query = "SELECT * FROM agent_performance"
        return self.execute_query(query)
    
    def cleanup_old_data(self, days_to_keep: int = 30) -> str:
        """Clean up old data using stored procedure."""
        with self.get_cursor() as (cursor, conn):
            cursor.execute("SELECT cleanup_old_data(%s)", (days_to_keep,))
            result = cursor.fetchone()
            conn.commit()
            return result[0]
    
    def get_database_stats(self) -> list:
        """Get database table statistics."""
        query = "SELECT * FROM database_stats()"
        return self.execute_query(query)
    
    def close(self):
        """Close all database connections."""
        if self._checkpointer:
            # PostgresSaver doesn't have explicit close method
            self._checkpointer = None
            
        if self._store:
            # PostgresStore doesn't have explicit close method  
            self._store = None
            
        logger.info("Database connections closed")


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """
    Get the global database manager instance.
    
    Initializes on first call using DATABASE_URL environment variable.
    """
    global _db_manager
    
    if _db_manager is None:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError(
                "DATABASE_URL environment variable not set. "
                "Please configure your PostgreSQL connection string."
            )
        
        _db_manager = DatabaseManager(database_url)
        logger.info("Database manager initialized")
    
    return _db_manager


def initialize_database(database_url: str = None) -> DatabaseManager:
    """
    Initialize database manager with specific URL.
    
    Useful for testing or when you need multiple database connections.
    """
    if database_url is None:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL must be provided or set as environment variable")
    
    return DatabaseManager(database_url)


# Async versions for future use
class AsyncDatabaseManager:
    """Async version of DatabaseManager for high-performance scenarios."""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self._pool = None
    
    async def initialize_pool(self):
        """Initialize async connection pool."""
        try:
            import asyncpg
            self._pool = await asyncpg.create_pool(self.database_url)
            logger.info("Async database pool initialized")
        except ImportError:
            logger.error("asyncpg not installed. Install with: pip install asyncpg")
            raise
    
    @asynccontextmanager
    async def get_connection(self):
        """Get async database connection."""
        if not self._pool:
            await self.initialize_pool()
        
        async with self._pool.acquire() as conn:
            yield conn
    
    async def execute_query(self, query: str, *args) -> list:
        """Execute async query."""
        async with self.get_connection() as conn:
            return await conn.fetch(query, *args)
    
    async def execute_command(self, command: str, *args) -> str:
        """Execute async command."""
        async with self.get_connection() as conn:
            return await conn.execute(command, *args)
    
    async def close(self):
        """Close async pool."""
        if self._pool:
            await self._pool.close()
            logger.info("Async database pool closed")
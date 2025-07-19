"""
Database setup utilities for Metanalyst-Agent

Handles database initialization and health checks for the multi-agent system.
"""

import asyncio
import logging
from typing import Dict, Any
from datetime import datetime, timedelta
import psycopg2
from psycopg2 import sql
import redis
import pymongo
from urllib.parse import urlparse


logger = logging.getLogger(__name__)


async def ensure_databases_ready(database_configs: Dict[str, str]) -> bool:
    """
    Ensure all required databases are ready and accessible
    
    Args:
        database_configs: Dictionary of database connection strings
        
    Returns:
        True if all databases are ready, False otherwise
    """
    
    results = {}
    
    # Check PostgreSQL
    if "postgresql" in database_configs:
        results["postgresql"] = await check_postgresql(database_configs["postgresql"])
    
    # Check Redis
    if "redis" in database_configs:
        results["redis"] = await check_redis(database_configs["redis"])
    
    # Check MongoDB
    if "mongodb" in database_configs:
        results["mongodb"] = await check_mongodb(database_configs["mongodb"])
    
    all_ready = all(results.values())
    
    if all_ready:
        logger.info("All databases are ready")
    else:
        failed_dbs = [db for db, status in results.items() if not status]
        logger.error(f"Failed to connect to databases: {failed_dbs}")
    
    return all_ready


async def check_postgresql(connection_string: str) -> bool:
    """
    Check PostgreSQL database connectivity and setup
    
    Args:
        connection_string: PostgreSQL connection string
        
    Returns:
        True if database is ready, False otherwise
    """
    
    try:
        # Parse connection string
        parsed = urlparse(connection_string)
        
        # Connect to PostgreSQL
        conn = psycopg2.connect(connection_string)
        cursor = conn.cursor()
        
        # Test basic connectivity
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        logger.info(f"PostgreSQL connected: {version[0][:50]}...")
        
        # Create database if it doesn't exist
        database_name = parsed.path.lstrip('/')
        if database_name:
            cursor.execute(
                sql.SQL("SELECT 1 FROM pg_database WHERE datname = %s"),
                [database_name]
            )
            
            if not cursor.fetchone():
                logger.info(f"Creating database: {database_name}")
                conn.autocommit = True
                cursor.execute(sql.SQL("CREATE DATABASE {}").format(
                    sql.Identifier(database_name)
                ))
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        logger.error(f"PostgreSQL connection failed: {str(e)}")
        return False


async def check_redis(connection_string: str) -> bool:
    """
    Check Redis connectivity
    
    Args:
        connection_string: Redis connection string
        
    Returns:
        True if Redis is ready, False otherwise
    """
    
    try:
        # Connect to Redis
        r = redis.from_url(connection_string)
        
        # Test basic connectivity
        r.ping()
        info = r.info()
        
        logger.info(f"Redis connected: version {info.get('redis_version', 'unknown')}")
        
        r.close()
        return True
        
    except Exception as e:
        logger.error(f"Redis connection failed: {str(e)}")
        return False


async def check_mongodb(connection_string: str) -> bool:
    """
    Check MongoDB connectivity
    
    Args:
        connection_string: MongoDB connection string
        
    Returns:
        True if MongoDB is ready, False otherwise
    """
    
    try:
        # Connect to MongoDB
        client = pymongo.MongoClient(connection_string)
        
        # Test basic connectivity
        server_info = client.server_info()
        logger.info(f"MongoDB connected: version {server_info.get('version', 'unknown')}")
        
        # Parse database name from connection string
        parsed = urlparse(connection_string)
        database_name = parsed.path.lstrip('/')
        
        if database_name:
            # Test database access
            db = client[database_name]
            collections = db.list_collection_names()
            logger.info(f"MongoDB database '{database_name}' accessible")
        
        client.close()
        return True
        
    except Exception as e:
        logger.error(f"MongoDB connection failed: {str(e)}")
        return False


async def initialize_postgresql_schema(connection_string: str) -> bool:
    """
    Initialize PostgreSQL schema for LangGraph checkpoints and store
    
    Args:
        connection_string: PostgreSQL connection string
        
    Returns:
        True if schema initialization succeeded, False otherwise
    """
    
    try:
        conn = psycopg2.connect(connection_string)
        cursor = conn.cursor()
        
        # Create checkpoints table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL DEFAULT '',
                checkpoint_id TEXT NOT NULL,
                parent_checkpoint_id TEXT,
                type TEXT,
                checkpoint JSONB NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{}',
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
            );
        """)
        
        # Create index for efficient lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS checkpoints_thread_id_idx 
            ON checkpoints (thread_id, checkpoint_ns);
        """)
        
        # Create store table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS store (
                namespace TEXT[] NOT NULL,
                key TEXT NOT NULL,
                value JSONB NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                PRIMARY KEY (namespace, key)
            );
        """)
        
        # Create index for namespace queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS store_namespace_idx 
            ON store USING GIN (namespace);
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info("PostgreSQL schema initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"PostgreSQL schema initialization failed: {str(e)}")
        return False


async def setup_development_databases() -> Dict[str, str]:
    """
    Setup development databases using Docker containers
    
    Returns:
        Dictionary of connection strings for development databases
    """
    
    # This would typically use Docker or docker-compose
    # For now, return localhost connections
    
    return {
        "postgresql": "postgresql://metanalyst:password@localhost:5432/metanalysis",
        "redis": "redis://localhost:6379/0",
        "mongodb": "mongodb://localhost:27017/metanalysis"
    }


async def cleanup_old_data(
    database_configs: Dict[str, str],
    days_old: int = 30
) -> bool:
    """
    Cleanup old data from databases
    
    Args:
        database_configs: Database connection configurations
        days_old: Number of days after which data is considered old
        
    Returns:
        True if cleanup succeeded, False otherwise
    """
    
    try:
        # Cleanup PostgreSQL
        if "postgresql" in database_configs:
            await cleanup_postgresql_data(database_configs["postgresql"], days_old)
        
        # Cleanup Redis
        if "redis" in database_configs:
            await cleanup_redis_data(database_configs["redis"], days_old)
        
        # Cleanup MongoDB
        if "mongodb" in database_configs:
            await cleanup_mongodb_data(database_configs["mongodb"], days_old)
        
        logger.info(f"Cleanup completed for data older than {days_old} days")
        return True
        
    except Exception as e:
        logger.error(f"Data cleanup failed: {str(e)}")
        return False


async def cleanup_postgresql_data(connection_string: str, days_old: int) -> None:
    """Cleanup old PostgreSQL data"""
    
    conn = psycopg2.connect(connection_string)
    cursor = conn.cursor()
    
    # Cleanup old checkpoints
    cursor.execute("""
        DELETE FROM checkpoints 
        WHERE (metadata->>'created_at')::timestamp < NOW() - INTERVAL '%s days'
    """, (days_old,))
    
    # Cleanup old store entries
    cursor.execute("""
        DELETE FROM store 
        WHERE created_at < NOW() - INTERVAL '%s days'
    """, (days_old,))
    
    conn.commit()
    cursor.close()
    conn.close()


async def cleanup_redis_data(connection_string: str, days_old: int) -> None:
    """Cleanup old Redis data"""
    
    r = redis.from_url(connection_string)
    
    # Redis TTL-based cleanup would be handled by Redis itself
    # For manual cleanup, we'd need to track timestamps in key names
    
    r.close()


async def cleanup_mongodb_data(connection_string: str, days_old: int) -> None:
    """Cleanup old MongoDB data"""
    
    client = pymongo.MongoClient(connection_string)
    
    # Parse database name
    parsed = urlparse(connection_string)
    database_name = parsed.path.lstrip('/')
    
    if database_name:
        db = client[database_name]
        
        # Cleanup old documents
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        for collection_name in db.list_collection_names():
            collection = db[collection_name]
            collection.delete_many({"created_at": {"$lt": cutoff_date}})
    
    client.close()
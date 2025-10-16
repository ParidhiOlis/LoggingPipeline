"""
Initialize ClickHouse database and tables for logging.

This script sets up the necessary database, tables, and indexes in ClickHouse
for storing and querying audit logs.
"""
import asyncio
import logging
import os
import sys
from typing import Dict, List, Optional

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import ClickHouse client
try:
    from clickhouse_driver import Client as ClickHouseClient
    from clickhouse_driver.errors import ServerException
    CLICKHOUSE_AVAILABLE = True
except ImportError:
    CLICKHOUSE_AVAILABLE = False

# Import configuration
from config.clickhouse_config import env_config, get_schema, get_materialized_views, get_indexes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ClickHouseInitializer:
    """Initialize ClickHouse database and tables for logging."""
    
    def __init__(self, config=None):
        """Initialize with configuration."""
        self.config = config or env_config
        self.client: Optional[ClickHouseClient] = None
    
    async def connect(self) -> bool:
        """Connect to ClickHouse server."""
        if not CLICKHOUSE_AVAILABLE:
            logger.error("clickhouse-driver is not installed. Install it with: pip install clickhouse-driver")
            return False
        
        try:
            # Connect to the server (without specifying a database first)
            connection_params = self.config.get_connection_params()
            database = connection_params.pop("database")
            
            self.client = ClickHouseClient(**connection_params)
            
            # Check if the database exists, create if it doesn't
            self.client.execute("CREATE DATABASE IF NOT EXISTS {database} ON CLUSTER '{cluster}'".format(
                database=database,
                cluster=os.getenv("CLICKHOUSE_CLUSTER", "")
            ) if os.getenv("CLICKHOUSE_CLUSTER") else 
               f"CREATE DATABASE IF NOT EXISTS {database}")
            
            # Switch to the database
            self.client.database = database
            
            logger.info(f"Connected to ClickHouse at {self.config.host}:{self.config.port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to ClickHouse: {str(e)}")
            return False
    
    async def initialize_schema(self) -> bool:
        """Initialize the database schema."""
        if not self.client:
            logger.error("Not connected to ClickHouse")
            return False
        
        try:
            # Create the main audit logs table
            schema_sql = get_schema()
            logger.info("Creating audit_logs table...")
            self.client.execute(schema_sql)
            
            # Create materialized views
            logger.info("Creating materialized views...")
            for name, sql in get_materialized_views().items():
                try:
                    self.client.execute(sql)
                    logger.debug(f"Created materialized view: {name}")
                except Exception as e:
                    logger.warning(f"Failed to create materialized view {name}: {str(e)}")
            
            # Add additional indexes
            logger.info("Creating additional indexes...")
            for idx in get_indexes():
                try:
                    alter_sql = f"ALTER TABLE audit_logs ADD {idx['sql']}"
                    self.client.execute(alter_sql)
                    logger.debug(f"Added index: {idx['name']}")
                except Exception as e:
                    logger.warning(f"Failed to add index {idx['name']}: {str(e)}")
            
            logger.info("Database schema initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize schema: {str(e)}")
            return False
    
    async def check_connection(self) -> bool:
        """Check if we can connect to ClickHouse and the table exists."""
        if not self.client:
            return False
        
        try:
            # Check if the table exists
            result = self.client.execute(
                "SELECT count() FROM system.tables WHERE database = %(database)s AND name = %(table)s",
                {"database": self.config.database, "table": self.config.table_name}
            )
            return result[0][0] > 0
        except Exception as e:
            logger.error(f"Error checking table existence: {str(e)}")
            return False
    
    async def close(self):
        """Close the ClickHouse connection."""
        if self.client:
            self.client.disconnect()
            self.client = None

async def main():
    """Main function to initialize ClickHouse."""
    if not CLICKHOUSE_AVAILABLE:
        logger.error("clickhouse-driver is not installed. Install it with: pip install clickhouse-driver")
        return False
    
    initializer = ClickHouseInitializer()
    
    try:
        # Connect to ClickHouse
        if not await initializer.connect():
            return False
        
        # Check if the table already exists
        if await initializer.check_connection():
            logger.info(f"Table {initializer.config.table_name} already exists in database {initializer.config.database}")
            if "--force" not in sys.argv:
                logger.info("Use --force to recreate the table")
                return True
            
            # Drop the existing table if --force is specified
            logger.warning(f"Dropping existing table {initializer.config.table_name}...")
            initializer.client.execute(f"DROP TABLE IF EXISTS {initializer.config.table_name}")
        
        # Initialize the schema
        return await initializer.initialize_schema()
        
    except Exception as e:
        logger.error(f"Error initializing ClickHouse: {str(e)}")
        return False
    finally:
        await initializer.close()

if __name__ == "__main__":
    logger.info("Initializing ClickHouse database...")
    success = asyncio.run(main())
    if success:
        logger.info("ClickHouse initialization completed successfully")
        sys.exit(0)
    else:
        logger.error("ClickHouse initialization failed")
        sys.exit(1)

"""
Simple script to test ClickHouse connection.
"""
import sys
from pathlib import Path

# Add the project root to the Python path
root_dir = str(Path(__file__).parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

import requests
from requests.auth import HTTPBasicAuth

def test_clickhouse_connection():
    """Test connection to ClickHouse using HTTP interface."""
    try:
        # Try with HTTP basic auth
        url = "http://localhost:8123/"
        query = "SHOW DATABASES"
        
        # Make the request with basic auth
        response = requests.post(
            url,
            data=query,
            auth=HTTPBasicAuth('default', 'password'),
            headers={'Content-Type': 'text/plain'}
        )
        
        if response.status_code == 200:
            print("✅ Successfully connected to ClickHouse via HTTP!")
            print("\nAvailable databases:")
            for db in response.text.strip().split('\n'):
                if db:  # Skip empty lines
                    print(f"- {db}")
            return True
        else:
            print(f"❌ Error connecting to ClickHouse: HTTP {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error connecting to ClickHouse: {str(e)}")
        return False

if __name__ == "__main__":
    test_clickhouse_connection()

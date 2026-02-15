import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file in PROJECT ROOT (not config folder)
env_path = Path(__file__).parent.parent / '.env'  # Goes up one level to project root
load_dotenv(dotenv_path=env_path)

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'user': os.getenv('DB_USER', 'bree'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME', 'pharmacy_forecasting')
}

# Validate password is loaded
if not DB_CONFIG['password']:
    raise ValueError("DB_PASSWORD not found. Check your .env file exists and contains DB_PASSWORD")

# Connection string for SQLAlchemy (used by pandas)
DATABASE_URL = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
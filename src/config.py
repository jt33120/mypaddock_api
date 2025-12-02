from pathlib import Path
import os

from dotenv import load_dotenv

# Load .env from the project root (Valuation/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

MARKETCHECK_API_KEY = os.getenv("MARKETCHECK_API_KEY")
MARKETCHECK_TOKEN = os.getenv("MARKETCHECK_TOKEN")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

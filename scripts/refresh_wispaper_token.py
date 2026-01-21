#!/usr/bin/env python3
"""
Manual script to refresh Wispaper token when refresh_token has expired.
Triggers the full OAuth flow via browser login.
"""

import sys
import os
import logging
from pathlib import Path

# Add parent directory to path for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
from dotenv import load_dotenv
env_file = project_root / '.env'
if env_file.exists():
    load_dotenv(env_file)
    print(f"✓ Loaded environment variables from {env_file}")
else:
    print(f"⚠️  Warning: .env file not found at {env_file}")

# Import the OAuth flow from wispaper_client
from paper_novelty_pipeline.services.wispaper_client import _run_oauth_flow

def main():
    """Run OAuth flow to get fresh tokens."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("\n🔄 Wispaper Token Refresh Tool")
    print("=" * 60)
    print("This will open your browser to login to Wispaper.")
    print("After successful login, your tokens will be refreshed.")
    print("=" * 60 + "\n")
    
    # Run the OAuth flow
    token = _run_oauth_flow()
    
    if token:
        print("\n✅ Token refreshed successfully!")
        print("You can now run your Phase2 scripts again.")
        return 0
    else:
        print("\n❌ Token refresh failed.")
        print("Please check your network connection and try again.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

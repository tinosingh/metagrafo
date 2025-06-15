import logging

logging.basicConfig(level=logging.DEBUG)

try:
    print("✅ Main app imports succeeded")
except Exception as e:
    print(f"❌ Import failed: {str(e)}")

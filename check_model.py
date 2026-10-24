"""
Check Model Preflight Utility

Performs a 1-message preflight connectivity check against the configured model endpoint.
Reuses ModelClient from src.harness.model_client without exposing credentials.
"""
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from dotenv import load_dotenv
    project_root = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(project_root, ".env")
    if os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path, override=True)
    else:
        load_dotenv(override=True)
except ImportError:
    pass


from src.harness.model_client import ModelClient
from src.harness.logger import Logger
from src.harness.terminal import print_red, print_green, print_blue


def main():
    model_name = os.environ.get("MODEL", "").strip()
    api_key = os.environ.get("API_KEY", "").strip()
    base_url = os.environ.get("BASE_URL", "").strip()

    if not model_name:
        print_red("FATAL: MODEL environment variable is not set.", bold=True)
        sys.exit(1)
    if not api_key:
        print_red("FATAL: API_KEY environment variable is not set.", bold=True)
        sys.exit(1)


    # Use Logger._redact to ensure safe printing
    logger = Logger()
    safe_model = logger._redact(model_name)
    safe_base = logger._redact(base_url) if base_url else "(default)"

    print("Checking model connectivity...")
    print("  Model: %s" % safe_model)
    print("  Base URL: %s" % safe_base)

    client = ModelClient(model=model_name, api_key=api_key, base_url=base_url)
    messages = [{"role": "user", "content": "ping"}]

    response = client.chat(messages, tools=[], timeout=30.0)

    if response.get("error"):
        safe_error = logger._redact(response["error"])
        print_red("Preflight check failed: %s" % safe_error, bold=True)
        sys.exit(1)

    print_green("Preflight check passed.", bold=True)

    print("  Latency: %.2fms" % response.get("latency_ms", 0.0))
    print("  Tokens: %d" % response.get("tokens", 0))
    sys.exit(0)


if __name__ == "__main__":
    main()

"""Run the synthetic sentence-gate staging service on 127.0.0.1 only."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from matchlab.local_staging import create_synthetic_staging_server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--port", default=8765, type=int)
    args = parser.parse_args()

    server = create_synthetic_staging_server(args.seed, port=args.port)
    print(f"Local synthetic gate staging: http://127.0.0.1:{server.server_port}")
    print("This unauthenticated process is local-staging-only; it is not production-ready.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping local gate staging service.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

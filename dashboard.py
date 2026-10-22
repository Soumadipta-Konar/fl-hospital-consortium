import argparse
import functools
import http.server
import os
import socketserver
import webbrowser

DASHBOARD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard")


def start_server(port=8000, open_browser=True):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=DASHBOARD_DIR)
    # Enable address reuse so restarts don't fail with port in use
    socketserver.TCPServer.allow_reuse_address = True

    with socketserver.TCPServer(("", port), handler) as httpd:
        url = f"http://localhost:{port}"
        print("=" * 65)
        print("  FL Hospital Consortium - Interactive Control Plane Dashboard")
        print(f"  Live at: {url}")
        print("  Serving from:", DASHBOARD_DIR)
        print("  Press Ctrl+C to stop.")
        print("=" * 65)

        if open_browser:
            webbrowser.open(url)

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down dashboard server.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch FL Hospital Consortium Web Dashboard")
    parser.add_argument("--port", type=int, default=8000, help="Port to run dashboard server on")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    args = parser.parse_args()

    start_server(port=args.port, open_browser=not args.no_browser)

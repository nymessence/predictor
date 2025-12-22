#!/usr/bin/env python3
"""
Simple HTTP server for the Flame Fractal Gallery
This script serves the fractal images and parameter files so they can be accessed from the gallery HTML page
"""

import http.server
import socketserver
import os
from pathlib import Path

PORT = 8000
DIRECTORY = Path(__file__).parent.resolve()  # Current directory (where gallery.html is)


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIRECTORY), **kwargs)

    def end_headers(self):
        # Add CORS headers to allow local access
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()


def main():
    print(f"Starting server at http://localhost:{PORT}")
    print(f"Serving directory: {DIRECTORY}")
    print("Open your browser and go to http://localhost:8000/gallery.html")
    print("Press Ctrl+C to stop the server")
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


if __name__ == "__main__":
    main()
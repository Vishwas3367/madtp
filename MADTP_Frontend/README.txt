MADTP frontend

The frontend is served by the project-root api_server.py, which also exposes read-only endpoints:
  GET /api/status
  GET /api/latest
  GET /api/history

Start the Python live runner and API server in separate terminals, then open http://127.0.0.1:8000. Use Settings and turn Demo values off to display rows written by live_madtp.py. The API only reads the results CSV; it does not calculate risk.

Demo mode is for UI layout only. Values shown in demo mode are placeholders, not actual worker measurements.

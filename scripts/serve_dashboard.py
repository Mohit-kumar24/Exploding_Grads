"""
KingdomForge Dashboard Server with Real Evaluation Data API

Serves the React evaluation dashboard and provides read-only REST API endpoints:
- GET /api/summary               -> Overall evaluation, baseline, and test summary
- GET /api/tasks                 -> Real task inventory from src/env/tasks.py with latest metrics
- GET /api/baselines             -> Baseline verification matrix from baseline-results/gate.json
- GET /api/runs                  -> Scanned model and baseline evaluation runs from logs/
- GET /api/runs/{run_id}         -> Single run metadata and verification details
- GET /api/runs/{run_id}/trajectory -> Granular step-by-step tool-call trajectory
- GET /api/tests                 -> Real test suite execution report

Security:
- Read-only endpoints with path-traversal sanitization
- Blocks access to .env, credentials, or internal git files
- No expensive model evaluations triggered from dashboard
"""
import http.server
import json
import os
import socketserver
import sys
import urllib.parse
import webbrowser
from typing import Any, Dict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PORT = 8000

from src.harness.dashboard_data import (
    compute_summary,
    get_baseline_gates,
    get_run_trajectory,
    get_task_inventory,
    get_test_report,
    sanitize_filename,
    scan_run_logs,
    GATE_FILE_PATH,
)



class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".json": "application/json",
        ".js": "application/javascript",
        ".html": "text/html",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        # Enable CORS and disable caching for real-time live evaluation metrics
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        clean_path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        # Security barrier: Block access to .env, git metadata, or hidden config files
        normalized_path = clean_path.lower()
        if "/.env" in normalized_path or normalized_path.startswith("/.git") or ".." in clean_path:
            self.send_error(403, "Access Denied: Protected Configuration")
            return

        # Route API requests
        if clean_path.startswith("/api/"):
            self._handle_api(clean_path, query)
            return

        # Serve static assets (dashboard/index.html, vendor scripts, etc.)
        super().do_GET()

    def _send_json(self, data: Any, status: int = 200):
        body = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_api(self, path: str, query: Dict[str, list]):
        try:
            if path == "/api/summary":
                tasks = get_task_inventory()
                gates = get_baseline_gates()
                runs = scan_run_logs(limit=200)
                test_rep = get_test_report(run_if_missing=False)
                summary = compute_summary(tasks, gates, runs, test_summary=test_rep)
                self._send_json(summary)
                return

            if path == "/api/tasks":
                tasks = get_task_inventory()
                gates = get_baseline_gates()
                runs = scan_run_logs(limit=200)

                tasks_list = []
                for tid in sorted(tasks.keys()):
                    task_data = tasks[tid]
                    task_runs = [r for r in runs if r.get("task_id") == tid and r.get("mode") == "model"]
                    latest_run = task_runs[0] if task_runs else None
                    tasks_list.append({
                        **task_data,
                        "baseline_gate": gates.get(tid),
                        "model_summary": {
                            "total_runs": len(task_runs),
                            "completed_runs": sum(1 for r in task_runs if r.get("is_completed")),
                            "successful_runs": sum(1 for r in task_runs if r.get("passed") is True),
                            "crashed_runs": sum(1 for r in task_runs if r.get("is_crash")),
                            "latest_score": latest_run.get("final_score") if latest_run else None,
                            "latest_passed": latest_run.get("passed") if latest_run else None,
                            "latest_run_id": latest_run.get("run_id") if latest_run else None,
                            "latest_timestamp": latest_run.get("timestamp") if latest_run else None,
                        } if task_runs else None,
                    })
                self._send_json({"status": "ok", "total": len(tasks_list), "tasks": tasks_list})
                return

            if path == "/api/baselines":
                tasks = get_task_inventory()
                gates = get_baseline_gates()
                gate_list = []
                for tid in sorted(tasks.keys()):
                    g = gates.get(tid, {})
                    gate_list.append({
                        "task_id": tid,
                        "task_title": tasks[tid].get("title", tid),
                        "difficulty": tasks[tid].get("difficulty", "unknown"),
                        "reference_score": g.get("reference_score"),
                        "reference_threshold": 0.95,
                        "reference_passed": g.get("reference_passed", False),
                        "noop_score": g.get("noop_score"),
                        "noop_threshold": 0.05,
                        "noop_passed": g.get("noop_passed", False),
                        "gate_passed": g.get("gate_passed", False),
                        "is_evaluated": bool(g),
                    })
                self._send_json({
                    "status": "ok",
                    "source": "baseline-results/gate.json",
                    "exists": os.path.exists(GATE_FILE_PATH),
                    "total": len(gate_list),
                    "gates": gate_list,
                })
                return

            if path == "/api/runs":
                task_id = query.get("task_id", [None])[0]
                mode = query.get("mode", [None])[0]
                try:
                    limit = int(query.get("limit", ["50"])[0])
                except (ValueError, TypeError):
                    limit = 50

                runs = scan_run_logs(task_id_filter=task_id, mode_filter=mode, limit=limit)
                self._send_json({
                    "status": "ok",
                    "total_returned": len(runs),
                    "filter_task_id": task_id,
                    "filter_mode": mode,
                    "runs": runs,
                })
                return

            if path.startswith("/api/runs/"):
                sub_path = path[len("/api/runs/"):]
                if sub_path.endswith("/trajectory"):
                    raw_id = sub_path[:-len("/trajectory")].strip("/")
                    safe_id = sanitize_filename(raw_id)
                    if not safe_id:
                        self._send_json({"status": "error", "error": "Invalid run ID parameter"}, status=400)
                        return
                    traj = get_run_trajectory(safe_id)
                    if not traj:
                        self._send_json({"status": "not_found", "error": f"Trajectory not found for {safe_id}"}, status=404)
                        return
                    self._send_json({"status": "ok", **traj})
                    return
                else:
                    safe_id = sanitize_filename(sub_path.strip("/"))
                    if not safe_id:
                        self._send_json({"status": "error", "error": "Invalid run ID parameter"}, status=400)
                        return
                    traj = get_run_trajectory(safe_id)
                    if not traj:
                        self._send_json({"status": "not_found", "error": f"Run not found: {safe_id}"}, status=404)
                        return
                    # Return metadata summary without full step payload
                    meta = {k: v for k, v in traj.items() if k != "steps"}
                    self._send_json({"status": "ok", "run": meta})
                    return

            if path == "/api/tests":
                refresh = query.get("refresh", ["false"])[0].lower() in ("true", "1")
                report = get_test_report(run_if_missing=True if refresh else False)
                self._send_json(report)
                return

            self._send_json({"status": "not_found", "error": f"Unknown API endpoint: {path}"}, status=404)
        except Exception as e:
            self._send_json({"status": "error", "error": str(e)}, status=500)


def main():
    os.chdir(PROJECT_ROOT)
    socketserver.TCPServer.allow_reuse_address = True

    httpd = None
    active_port = PORT
    for candidate_port in [8000, 8080, 8081, 8888]:
        try:
            httpd = http.server.ThreadingHTTPServer(("", candidate_port), DashboardHandler)
            active_port = candidate_port
            break
        except OSError:
            continue

    if not httpd:
        print("  Error: Could not bind to candidate ports (8000, 8080, 8081, 8888).")
        sys.exit(1)

    url = f"http://localhost:{active_port}/dashboard/index.html"
    print("=" * 60)
    print("  KingdomForge Data-Driven Dashboard Server")
    print("=" * 60)
    print(f"  Local URL: {url}")
    print("  Active Endpoints:")
    print("    - GET /api/summary")
    print("    - GET /api/tasks")
    print("    - GET /api/baselines")
    print("    - GET /api/runs")
    print("    - GET /api/runs/{run_id}/trajectory")
    print("    - GET /api/tests")
    print("  Press Ctrl+C to stop the server.")
    print("=" * 60)

    try:
        webbrowser.open(url)
    except Exception:
        pass

    with httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  Dashboard server stopped.")
            sys.exit(0)


if __name__ == "__main__":
    main()

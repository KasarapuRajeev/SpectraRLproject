#!/usr/bin/env python3
"""Start SpectraRL: python run.py [--port 8000] [--host 127.0.0.1]"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))
import uvicorn

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    uvicorn.run("spectrarl.api:app", host=a.host, port=a.port, log_level="info")

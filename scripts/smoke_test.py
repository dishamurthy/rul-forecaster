"""
Quick end-to-end check against a RUNNING server (start it first):

    uvicorn api.server:app --port 8000
    python scripts/smoke_test.py            # optional: --base http://127.0.0.1:8000

Verifies the API is up, the dataset + checkpoints load, and that the live-computed
metrics reproduce the notebook's reported numbers (RMSE 13.95 LSTM / 14.82 CNN).
"""
import argparse
import json
import sys
import urllib.request

REPORTED = {"lstm": 13.95, "cnn": 14.82}
TOL = 0.5          # allow small drift (GPU vs CPU float differences)


def get(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    base = ap.parse_args().base.rstrip("/")
    ok = True

    st = get(f"{base}/api/status")
    print(f"status      : {st['status']} | device={st['device']} | data_ready={st['data_ready']} | checkpoints={st['checkpoints']}")
    if not st["data_ready"]:
        print("FAIL: run  python scripts/download_data.py"); return 1

    b = get(f"{base}/api/benchmark")
    for arch, m in b["models"].items():
        diff = abs(m["rmse"] - REPORTED[arch])
        flag = "OK " if diff <= TOL else "DIFF"
        ok &= diff <= TOL
        print(f"{flag} {arch:5s}: RMSE={m['rmse']} (reported {REPORTED[arch]}) MAE={m['mae']} "
              f"NASA={m['nasa_score']} CI90-coverage={m['ci90_coverage']}")

    e = get(f"{base}/api/engine/1?arch=lstm")
    last = e["steps"][-1]
    print(f"engine 1    : {len(e['steps'])} steps | last pred={last['rul_mean']} "
          f"[{last['ci_lo']}, {last['ci_hi']}] true={last['true_rul']}")

    req = urllib.request.Request(
        f"{base}/api/predict",
        data=json.dumps({"sequence": [[0.5] * 15] * 30, "arch": "lstm"}).encode(),
        headers={"Content-Type": "application/json"})
    print("predict     :", json.loads(urllib.request.urlopen(req, timeout=60).read()))

    print("\nALL GOOD" if ok else "\nMetrics differ from the README by more than the tolerance — see README 'Known limitations'.")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())

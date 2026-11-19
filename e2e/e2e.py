"""End-to-end UI test. Usage: python e2e/e2e.py [base_url] [shots_dir]"""
import sys, time
from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
SHOTS = sys.argv[2] if len(sys.argv) > 2 else "/tmp/shots"
ROUTES = ["/", "/lab/train", "/lab/monitor", "/lab/inspector", "/results/benchmarks", "/results/runs", "/learn/research", "/learn/system", "/learn/reward", "/learn/algorithm", "/learn/setup", "/train", "/benchmarks"]
errors, fails = [], []

def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond: fails.append(msg)

with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--no-sandbox"])
    pg = b.new_page(viewport={"width": 1440, "height": 900})
    pg.on("console", lambda m: errors.append((pg.url, m.text)) if m.type == "error" else None)
    pg.on("pageerror", lambda e: errors.append((pg.url, str(e))))
    for r in ROUTES:
        pg.goto(BASE + r); pg.wait_for_load_state("networkidle"); time.sleep(1.2)
        pg.screenshot(path=f"{SHOTS}/{r.strip('/').replace('/','_') or 'home'}.png")
        check(pg.locator("h1").count() > 0, f"route {r} renders a heading")
    pg.goto(BASE + "/"); check(pg.locator("aside a.nav").count() == 4, "sidebar has exactly 4 items")
    pg.goto(BASE + "/train"); pg.wait_for_url("**/lab/train", timeout=5000); check(True, "legacy /train redirects to /lab/train")
    # Train workflow
    pg.goto(BASE + "/lab/train"); pg.wait_for_load_state("networkidle")
    pg.get_by_role("button", name="Quick").first.click() if pg.get_by_role("button", name="Quick").count() else print("no quick preset")
    btn = pg.get_by_role("button", name="Train", exact=True)
    check(btn.count() > 0, "train page has a Train button")
    if btn.count():
        btn.first.click(); time.sleep(8)
        pg.screenshot(path=f"{SHOTS}/train_running.png")
        for _ in range(90):
            if pg.get_by_text("completed", exact=False).count() or pg.get_by_text("done", exact=True).count(): break
            time.sleep(2)
        pg.screenshot(path=f"{SHOTS}/train_done.png")
    for r in ["/lab/monitor", "/lab/inspector", "/results/runs", "/results/benchmarks"]:
        pg.goto(BASE + r); pg.wait_for_load_state("networkidle"); time.sleep(3)
        pg.screenshot(path=f"{SHOTS}/after_{r.strip('/').replace('/','_')}.png")
    # mobile
    m = b.new_page(viewport={"width": 390, "height": 800}); m.goto(BASE + "/"); m.wait_for_load_state("networkidle"); time.sleep(1)
    m.screenshot(path=f"{SHOTS}/mobile_home.png")
    ow = m.evaluate("document.documentElement.scrollWidth - window.innerWidth")
    check(ow <= 2, f"no horizontal overflow on mobile (overflow={ow}px)")
    b.close()
check(not errors, f"no console/page errors {errors[:5]}")
print("FAILS:", fails); sys.exit(1 if fails else 0)

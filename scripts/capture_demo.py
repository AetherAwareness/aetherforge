#!/usr/bin/env python3
"""
Capture AetherForge Neural Command demo assets for README / Hugging Face.

Produces under docs/demo/ (or --out):
  - hero.png          full console screenshot
  - studio.png        expert group studio focus
  - fire.png          neural fire preview
  - demo.webm         short animated tour (ffmpeg)
  - demo.gif          lightweight loop for README (if palette ok)

Usage:
  # Ensure a recent run exists first:
  #   bash scripts/run_flagship.sh dry-run
  python scripts/capture_demo.py
  python scripts/capture_demo.py --theme matrix --port 8765
  python scripts/capture_demo.py --no-video   # screenshots only
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def find_run(runs_root: Path) -> Path | None:
    if not runs_root.exists():
        return None
    runs = [p for p in runs_root.iterdir() if p.is_dir()]
    # prefer flagship
    flag = sorted(
        [p for p in runs if "flagship" in p.name],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if flag:
        return flag[0]
    runs = sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


def start_dashboard(port: int) -> subprocess.Popen:
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "aetherforge.cli",
            "dashboard",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--runs-root",
            str(ROOT / "artifacts" / "runs"),
        ],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def wait_http(url: str, timeout: float = 20.0) -> bool:
    import urllib.request

    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.3)
    return False


def capture(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frames = out / "frames"
    if frames.exists():
        shutil.rmtree(frames)
    frames.mkdir()

    run = find_run(ROOT / "artifacts" / "runs")
    if not run:
        print("No runs found. Run: bash scripts/run_flagship.sh dry-run")
        return 1
    print(f"Using run: {run.name}")

    port = args.port
    base = f"http://127.0.0.1:{port}"
    proc = None
    owned = False
    if not wait_http(base + "/api/health", timeout=1.5):
        print(f"Starting dashboard on :{port} …")
        proc = start_dashboard(port)
        owned = True
        if not wait_http(base + "/api/health", timeout=15):
            print("Dashboard failed to start")
            if proc:
                proc.terminate()
            return 1
    else:
        print(f"Reusing dashboard already on :{port}")

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright not installed: pip install playwright && playwright install chromium")
        if owned and proc:
            proc.terminate()
        return 1

    theme = args.theme
    with sync_playwright() as p:
        # Prefer system Chrome when Playwright browsers aren't installed
        try:
            browser = p.chromium.launch(headless=True, channel="chrome")
        except Exception:
            try:
                browser = p.chromium.launch(headless=True)
            except Exception as e:
                print(
                    "Could not launch browser. Try:\n"
                    "  playwright install chromium\n"
                    f"or install Google Chrome. Underlying error: {e}"
                )
                if owned and proc:
                    proc.terminate()
                return 1
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1.25)
        page.goto(base + "/", wait_until="networkidle", timeout=60000)
        # set theme before heavy UI
        page.evaluate(
            """(t) => {
              localStorage.setItem('aetherforge-theme', t);
              document.documentElement.setAttribute('data-theme', t);
              document.querySelectorAll('[data-theme-btn]').forEach(b => {
                b.classList.toggle('active', b.getAttribute('data-theme-btn') === t);
              });
            }""",
            theme,
        )
        page.wait_for_timeout(600)

        # click latest run if list present
        page.wait_for_selector(".run-item, #runList", timeout=15000)
        items = page.query_selector_all(".run-item")
        target = None
        for el in items:
            name = (el.inner_text() or "").split("\n")[0].strip()
            if run.name in name or "flagship" in name:
                target = el
                break
        if not target and items:
            target = items[0]
        if target:
            target.click()
            page.wait_for_timeout(1500)

        # wait for detail
        page.wait_for_selector("#detail", timeout=20000)
        page.wait_for_timeout(800)

        # Hero
        page.screenshot(path=str(out / "hero.png"), full_page=False)
        print("  wrote hero.png")

        # Scroll to studio
        studio = page.query_selector("#studioPanel")
        if studio:
            studio.scroll_into_view_if_needed()
            page.wait_for_timeout(700)
            page.screenshot(path=str(out / "studio.png"), full_page=False)
            print("  wrote studio.png")

        # Advanced bay — fire
        adv = page.query_selector("#advBay")
        if adv:
            adv.scroll_into_view_if_needed()
            page.wait_for_timeout(400)
            fire_btn = page.query_selector("#btnFire")
            if fire_btn:
                fire_btn.click()
                page.wait_for_timeout(2200)
            page.screenshot(path=str(out / "fire.png"), full_page=False)
            print("  wrote fire.png")

            # constellation tab
            tab = page.query_selector('.adv-tabs button[data-tab="constellation"]')
            if tab:
                tab.click()
                page.wait_for_timeout(1600)
                page.screenshot(path=str(out / "constellation.png"), full_page=False)
                print("  wrote constellation.png")

        # Timeline replay for motion frames
        if not args.no_video:
            page.evaluate("window.scrollTo(0, 0)")
            page.wait_for_timeout(400)
            n_frames = args.frames
            for i in range(n_frames):
                try:
                    if i == 2:
                        page.evaluate("window.scrollTo(0, 0)")
                        btn = page.query_selector("#btnReplay")
                        if btn:
                            btn.click(timeout=3000)
                    if i == 6:
                        page.evaluate(
                            """() => {
                              const order = ['nexus','matrix','plasma'];
                              const t = document.documentElement.getAttribute('data-theme') || 'nexus';
                              const next = order[(order.indexOf(t)+1) % order.length];
                              localStorage.setItem('aetherforge-theme', next);
                              document.documentElement.setAttribute('data-theme', next);
                              document.querySelectorAll('[data-theme-btn]').forEach(b => {
                                b.classList.toggle('active', b.getAttribute('data-theme-btn') === next);
                              });
                            }"""
                        )
                    if i == 9:
                        tab = page.query_selector('.adv-tabs button[data-tab="constellation"]')
                        if tab:
                            tab.scroll_into_view_if_needed()
                            tab.click(timeout=3000)
                except Exception as e:
                    print(f"  frame {i} action skipped: {e}")
                page.screenshot(path=str(frames / f"f{i:03d}.png"), full_page=False)
                page.wait_for_timeout(int(1000 / max(args.fps, 1)))
            print(f"  captured {n_frames} frames")

        browser.close()

    if owned and proc:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

    # encode video
    if not args.no_video and shutil.which("ffmpeg") and list(frames.glob("f*.png")):
        webm = out / "demo.webm"
        gif = out / "demo.gif"
        # webm
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(args.fps),
            "-i", str(frames / "f%03d.png"),
            "-c:v", "libvpx-vp9", "-b:v", "1.5M", "-pix_fmt", "yuva420p",
            str(webm),
        ]
        # fallback if yuva fails
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(args.fps),
                "-i", str(frames / "f%03d.png"),
                "-c:v", "libvpx-vp9", "-b:v", "1.5M", "-pix_fmt", "yuv420p",
                str(webm),
            ]
            r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            print(f"  wrote {webm.name} ({webm.stat().st_size // 1024} KB)")
        else:
            print("  webm encode failed:", (r.stderr or "")[-400:])

        # gif palette
        pal = frames / "palette.png"
        subprocess.run(
            [
                "ffmpeg", "-y", "-framerate", str(args.fps),
                "-i", str(frames / "f%03d.png"),
                "-vf", "fps=8,scale=960:-1:flags=lanczos,palettegen=stats_mode=diff",
                str(pal),
            ],
            capture_output=True,
        )
        if pal.exists():
            r = subprocess.run(
                [
                    "ffmpeg", "-y", "-framerate", str(args.fps),
                    "-i", str(frames / "f%03d.png"), "-i", str(pal),
                    "-lavfi", "fps=8,scale=960:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer",
                    str(gif),
                ],
                capture_output=True,
                text=True,
            )
            if r.returncode == 0 and gif.exists():
                print(f"  wrote {gif.name} ({gif.stat().st_size // 1024} KB)")

    # manifest
    manifest = {
        "run": run.name,
        "theme": theme,
        "files": sorted([p.name for p in out.iterdir() if p.is_file()]),
        "generated_at": time.time(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nDemo assets → {out}")
    print("Embed in README:\n  ![AetherForge](docs/demo/hero.png)\n  ![Demo](docs/demo/demo.gif)")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Capture AetherForge demo assets")
    ap.add_argument("--out", default=str(ROOT / "docs" / "demo"))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--theme", default="nexus", choices=["nexus", "matrix", "plasma"])
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    # ensure package import path
    sys.path.insert(0, str(ROOT))
    raise SystemExit(capture(args))


if __name__ == "__main__":
    main()

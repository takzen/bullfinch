"""Render the Bullfinch logo and the 1280x640 banner (GitHub social preview, Hugging Face card).

    uv run --with playwright python assets/render_assets.py

Needs a Playwright Chromium (``playwright install chromium``). Outputs land next to this file.
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
LOGO = (HERE / "bullfinch-logo.svg").read_text(encoding="utf-8")

BANNER = f"""<!doctype html><html><head><meta charset="utf-8"><style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ width:1280px; height:640px; background:#08090c; color:#e5e5e5;
       font-family:"Segoe UI",-apple-system,Roboto,Arial,sans-serif;
       display:flex; align-items:center; gap:64px; padding:0 88px; }}
.logo svg {{ width:360px; height:360px; display:block; }}
.eyebrow {{ color:#ff8a70; font-family:Consolas,monospace; font-size:18px; font-weight:700;
           letter-spacing:.2em; text-transform:uppercase; margin-bottom:18px; }}
h1 {{ font-size:96px; line-height:1; font-weight:700; color:#fff; letter-spacing:-.02em; }}
h2 {{ font-size:36px; line-height:1.2; font-weight:600; color:#ff8a70; margin-top:14px; }}
p {{ font-size:24px; line-height:1.45; color:#9ca3af; margin-top:22px; max-width:640px; }}
.foot {{ margin-top:34px; font-size:20px; color:#6b7280; }}
.foot b {{ color:#e5e5e5; font-weight:600; }}
</style></head><body>
  <div class="logo">{LOGO}</div>
  <div>
    <div class="eyebrow">Typed decisions · Polski</div>
    <h1>Bullfinch 4B</h1>
    <h2>A small decision model for Polish</h2>
    <p>Choice, yes/no and score questions in Polish. One forward pass, a probability for
       every option, and it runs on your own machine.</p>
    <div class="foot"><b>Kanari</b> · kanari.pl · sibling of Serin 4B</div>
  </div>
</body></html>"""


def main() -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 512, "height": 512})
        page.set_content(f"<html><body style='margin:0;background:transparent'>{LOGO}</body></html>")
        page.screenshot(path=str(HERE / "bullfinch-logo.png"), omit_background=True)
        page = browser.new_page(viewport={"width": 1280, "height": 640})
        page.set_content(BANNER)
        page.screenshot(path=str(HERE / "bullfinch-banner.png"))
        browser.close()
    for name in ("bullfinch-logo.png", "bullfinch-banner.png"):
        print(name, (HERE / name).stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()

import os
import subprocess
import shutil
import pymupdf

HTML_PATH = os.path.abspath("hybridsense_paper/paper.html")
PDF_OUTPUT = os.path.abspath("hybridsense_paper/HybridSense_Research_Paper.pdf")
ROOT_PDF = os.path.abspath("HybridSense_Research_Paper.pdf")
PREVIEW_DIR = os.path.abspath("hybridsense_paper/preview_pages")

os.makedirs(PREVIEW_DIR, exist_ok=True)

chrome_paths = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
]

browser_exe = None
for p in chrome_paths:
    if os.path.exists(p):
        browser_exe = p
        break

if not browser_exe:
    raise RuntimeError("No Chrome or Edge binary found!")

print(f"Using browser: {browser_exe}")

# Run headless chrome to print to PDF
cmd = [
    browser_exe,
    "--headless=new",
    "--disable-gpu",
    "--no-pdf-header-footer",
    f"--print-to-pdf={PDF_OUTPUT}",
    f"file:///{HTML_PATH}"
]

print("Executing Chrome headless print-to-pdf...")
res = subprocess.run(cmd, capture_output=True, text=True)
print("STDOUT:", res.stdout)
print("STDERR:", res.stderr)

if not os.path.exists(PDF_OUTPUT):
    # Try legacy headless if new headless didn't create file
    cmd_legacy = [
        browser_exe,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={PDF_OUTPUT}",
        f"file:///{HTML_PATH}"
    ]
    subprocess.run(cmd_legacy, capture_output=True, text=True)

if not os.path.exists(PDF_OUTPUT):
    raise RuntimeError("PDF generation failed!")

print(f"PDF successfully generated: {PDF_OUTPUT} ({os.path.getsize(PDF_OUTPUT)} bytes)")

# Copy to root directory
shutil.copy2(PDF_OUTPUT, ROOT_PDF)
print(f"Copied to root: {ROOT_PDF}")

# Verify with PyMuPDF and generate preview pages
doc = pymupdf.open(PDF_OUTPUT)
print(f"Total Pages in Generated Paper: {len(doc)}")

for i, page in enumerate(doc):
    pix = page.get_pixmap(dpi=150)
    out_img = os.path.join(PREVIEW_DIR, f"page_{i+1}.png")
    pix.save(out_img)
    print(f"Rendered Page {i+1} -> {out_img}")

print("Compilation and verification complete!")

"""Find procurement decisions with an inline comparison table in Dynasty portals.

Finnish municipalities publish decisions through Dynasty tietopalvelu
(*.oncloudos.com). The decision text is server-rendered HTML, so a comparison
table printed into the decision body is machine-readable - unlike the same table
shipped as a PDF attachment.

    DREQUEST.PHP?page=meeting_frames        bodies, with meeting ids
    DREQUEST.PHP?page=meeting&id=M          one meeting, with item ids M-1, M-2
    DREQUEST.PHP?page=meetingitem&id=M-k    the decision text

This walks meetings, keeps items whose title looks like a procurement, fetches
them, and reports which ones contain a table with several numeric columns.
"""
import html
import re
import subprocess
import sys
import time
from pathlib import Path

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/131.0 Safari/537.36"
MEET = re.compile(r"DREQUEST\.PHP\?page=meeting&id=(\d+)")
ITEM = re.compile(r"DREQUEST\.PHP\?page=meetingitem&id=([0-9-]+)")
LINK = re.compile(r'<a[^>]*meetingitem&id=([0-9-]+)[^>]*>(.*?)</a>', re.S | re.I)
TAG = re.compile(r"<[^>]+>")
KEY = re.compile(r"hankin|tarjous|kilpailut|urakka|puitejärjest|puitejarjest",
                 re.I)
TABLE = re.compile(r"<table.*?</table>", re.S | re.I)
ROW = re.compile(r"<tr.*?</tr>", re.S | re.I)
CELL = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.S | re.I)
NUMC = re.compile(r"^-?\d+(?:[.,]\d+)?$")


def get(url: str) -> str:
    r = subprocess.run(["curl", "-sL", "--max-time", "30", "-A", UA, url],
                       capture_output=True)
    return r.stdout.decode("latin-1", "replace")


def text(s: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(TAG.sub(" ", s))).strip()


def numeric_tables(page: str):
    """Tables with at least 3 rows and 2 numeric columns - a scoring table shape."""
    out = []
    for tb in TABLE.findall(page):
        rows = []
        for r in ROW.findall(tb):
            cells = [text(c) for c in CELL.findall(r)]
            if cells:
                rows.append(cells)
        if len(rows) < 3:
            continue
        w = max(len(r) for r in rows)
        numcols = 0
        for c in range(w):
            col = [r[c] for r in rows if len(r) > c]
            hits = sum(1 for v in col if NUMC.match(v.replace(" ", "")))
            if hits >= max(2, len(col) - 2):
                numcols += 1
        if numcols >= 2:
            out.append(rows)
    return out


def main():
    host = sys.argv[1] if len(sys.argv) > 1 else "espoo.oncloudos.com"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    base = f"https://{host}/cgi/"
    frames = get(base + "DREQUEST.PHP?page=meeting_frames")
    meetings = sorted(set(MEET.findall(frames)), reverse=True)[:limit]
    print(f"  {host}: {len(meetings)} meetings", flush=True)
    hits = 0
    outdir = Path(__file__).with_name("proc_items")
    outdir.mkdir(exist_ok=True)
    for mid in meetings:
        page = get(base + f"DREQUEST.PHP?page=meeting&id={mid}")
        # The title is the anchor's TEXT, not what follows the link. The first
        # version split on the id and read the next 400 characters, which is
        # the surrounding table markup, so KEY never matched and the scan
        # reported 0 hits from 20 meetings without fetching a single item.
        for iid, title_raw in LINK.findall(page):
            title = text(title_raw)
            if not KEY.search(title):
                continue
            item = get(base + f"DREQUEST.PHP?page=meetingitem&id={iid}")
            tabs = numeric_tables(item)
            if tabs:
                hits += 1
                p = outdir / f"{host}_{iid}.html"
                p.write_text(item, encoding="utf-8", errors="replace",
                             newline="\n")
                print(f"  HIT {iid}  {len(tabs)} numeric table(s)  {title[:70]}",
                      flush=True)
            time.sleep(0.05)
    print(f"done: {hits} items with numeric tables")


if __name__ == "__main__":
    main()

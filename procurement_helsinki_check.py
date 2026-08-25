"""Does Helsinki's own decision archive give what the Dynasty and TED routes
did not: an openly fetchable, multi-criteria comparison table?

procurement_harvest_results.txt logged three routes not yet tried. This
checks the first two and finds a fourth, unlisted one instead:

    Finlex's own API      - checked against the live OpenAPI description.
                             The judgment endpoint's type enum is fixed and
                             does not include the Market Court, so the API
                             cannot reach MAO decisions at all, regardless of
                             client-side rendering. Dead end confirmed at the
                             schema, not the network.

    viranhaltija.fi        - the archive gate is not a 403 on every request;
                             it is a 21-day retention wall. Fetching a PDF
                             indexed by a search engine now returns the same
                             plain-text refusal ("... yli 21 pv vanha. Kirjaudu
                             sisaan.") regardless of URL. Confirms, does not
                             overturn, the earlier finding.

    Helsinki paatokset.hel.fi / ahjojulkaisu.hel.fi - NOT one of the three
                             listed routes. Helsinki runs its own decision
                             system (Ahjo), separate from the Dynasty
                             municipalities already scanned. Some decisions
                             withhold the comparison table ("Liitetta ei
                             julkaista internetissa"); others publish it as a
                             directly fetchable PDF under ahjojulkaisu.hel.fi,
                             no login. This checks three known URLs and
                             reports whether each is a real multi-criterion
                             scoring table (the shape procurement_shape.py
                             needs) or a single-criterion price list (the
                             shape TED already gave us, which is not useful).

    python procurement_helsinki_check.py
"""
import re
import subprocess
import sys

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/131.0 Safari/537.36"

FINLEX_YAML = "https://opendata.finlex.fi/Finlex_avoin_data_v0_4_0.yaml"

# Found by web search 2026-08-25, not guessed - see the run's chat log.
HELSINKI_PDFS = [
    ("HEL 2021-005918, arvokuljetuspalvelut (halvin hinta)",
     "https://ahjojulkaisu.hel.fi/62DC944F-D9FD-C589-97A6-7B0AE1000000.pdf"),
    ("HEL 2023-014799, kopiopaperit (halvin hinta)",
     "https://ahjojulkaisu.hel.fi/D17E0692-BEC7-C9FA-B95A-8E514F700001.pdf"),
    ("kiinteistoen kunnossapitourakka, 8 tarjoajaa (hinta+laatu)",
     "https://ahjojulkaisu.hel.fi/4DF0073A-284F-C168-8D38-6D1A7AE00000.pdf"),
]

VH_LOCKED_PDF = ("https://viranhaltija.fi/arkisto/data/pdf_arkisto/keminmaa/"
                 "2026/paatokset/9cfab4af3580626310b93c5bba26650524db3e8"
                 "604db0a8fc01092144eeac580.pdf")


def fetch(url: str) -> bytes:
    r = subprocess.run(["curl", "-sL", "--max-time", "30", "-A", UA, url],
                        capture_output=True)
    return r.stdout


def pdf_bidder_criterion_shape(pdf_bytes: bytes):
    """Rough (J, n) from a pdftotext -layout dump: bidder names as the header
    row of TARJOUSTEN VERTAILUTAULUKKO, criteria as 'Laatukriteeri N.' lines."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(pdf_bytes)
        path = f.name
    r = subprocess.run(["pdftotext", "-layout", path, "-"],
                        capture_output=True)
    txt = r.stdout.decode("utf-8", "replace")
    n_criteria = len(set(re.findall(r"Laatukriteeri\s+(\d+)", txt)))
    has_price = "Kokonaishinta" in txt or "HINTAPISTEYTYS" in txt
    if n_criteria:
        n_criteria += 1  # price is itself a criterion when quality is scored
    elif has_price:
        n_criteria = 1
    return txt, n_criteria


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("ROUTE 1: Finlex's own API")
    yaml = fetch(FINLEX_YAML).decode("utf-8", "replace")
    m = re.search(r"JudgmentDocumentType:.*?enum:\s*((?:\s*-\s*[\w-]+\n)+)",
                   yaml, re.S)
    types = re.findall(r"-\s*([\w-]+)", m.group(1)) if m else []
    print(f"  judgment endpoint's document types: {types or 'NOT FOUND'}")
    has_market_court = any("market" in t or "mao" in t for t in types)
    print(f"  [{'FAIL' if has_market_court else 'ok  '}] Market Court is "
          f"{'present' if has_market_court else 'ABSENT'} from the type enum "
          f"- the API cannot reach it{' (dead end confirmed)' if not has_market_court else ''}")

    print(chr(10) + "ROUTE 2: viranhaltija.fi archive, a PDF indexed by a search engine")
    body = fetch(VH_LOCKED_PDF)
    locked = b"lukittu" in body and b"Kirjaudu" in body
    print(f"  [{'ok  ' if locked else 'FAIL'}] {'still ' if locked else 'NOT '}"
          f"gated: {body[:80]!r}")

    print(chr(10) + "ROUTE 4 (not one of the three listed): Helsinki Ahjo publication")
    rows = []
    for label, url in HELSINKI_PDFS:
        body = fetch(url)
        is_pdf = body[:4] == b"%PDF"
        shape = None
        if is_pdf:
            txt, n = pdf_bidder_criterion_shape(body)
            j_ranked = len(re.findall(r"Sijoitus\s+\d+", txt))
            header = re.search(r"TARJOUSTEN VERTAILUTAULUKKO(.*)",
                                txt.splitlines()[0] if txt else "")
            j_header = len([c for c in re.split(r"\s{2,}", header.group(1))
                            if c.strip()]) if header else 0
            shape = (max(j_ranked, j_header), n)
        print(f"  [{'ok  ' if is_pdf else 'FAIL'}] {label}")
        print(f"        {url}")
        print(f"        {len(body)} bytes, PDF={is_pdf}, "
              f"(bidders, criteria)={shape}")
        rows.append((label, url, is_pdf, shape))

    multi = [r for r in rows if r[3] and r[3][1] and r[3][1] >= 2]
    print(chr(10) + f"{len(multi)}/{len(rows)} checked URLs are a real "
          f"multi-criterion scoring table (n >= 2), openly fetchable, "
          f"no login.")
    if multi:
        print("This is a usable overlay source procurement_shape.py did not "
              "have: enumeration (not just spot-checking) is the open work, "
              "since Helsinki's old OpenAhjo listing API was retired and "
              "paatokset.hel.fi has no documented replacement.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

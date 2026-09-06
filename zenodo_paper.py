#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Talleta paperi Zenodoon OMANA Publication-tietueenaan.

    python zenodo_paper.py                 # luo LUONNOS + lataa PDF (ei julkaise)
    python zenodo_paper.py --show <ID>     # nayta luonnoksen tila
    python zenodo_paper.py --publish <ID>  # julkaise (erillinen, tietoinen askel)

Miksi oma tietue eika versio koodiin: `10.5281/zenodo.22342751` on tyypiltaan
**Software** ja sen zipissa on 338 tiedostoa ja **0 PDF:aa** (repon .gitignore sulkee
*.pdf), joten paperi ei ole siteerattavissa silta laskeutumissivulta. Paperi ja koodi
ovat kaksi tuotosta; ne linkitetaan `isSupplementedBy`-suhteella.

Venue tarkistettu ENNEN talletusta ([[feedback-preprint-deposit-closes-venues]]):
jmlr.org/tmlr/author-guide.html sallii preprintin nimenomaisesti. TMLR-versio on
anonymisoitava eika se saa linkittaa tahan nimelliseen versioon.

EI julkaise vahingossa: julkaisu on eri komento. Zenodo-DOI on pysyva.
"""
import argparse, json, os, pathlib, sys, urllib.error, urllib.request

API = "https://zenodo.org/api/deposit/depositions"
HERE = pathlib.Path(__file__).resolve().parent
PDF = HERE / "paper.pdf"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

ABSTRACT = (
    "A benchmark leaderboard is read as a stopwatch: the rank beside a model is treated as a "
    "fact. Yet every entrant is scored on the same items and publishes which it solved, so a "
    "leaderboard is a paired experiment, and its ordering can be tested rather than trusted. We "
    "test each adjacent-rank pair for statistical separability - exact McNemar's test on "
    "discordant items for pass/fail benchmarks, a paired item-bootstrap for scored ones - across "
    "five public boards. On SWE-bench Verified, 129 of 133 adjacent ranks are not separable at "
    "the 5% level; on MTEB, 176 of 180; on HELM Lite, all 89 of 89. The benchmarks are not "
    "broken: 88% of all pairs separate cleanly. It is the neighbours they cannot order, and "
    "neighbours are what the top of a table is made of. We derive a resolution bound, "
    "delta &gt;= 2.80*sqrt(d/n) in the discordance rate d, that predicts which pairs are decidable "
    "from a benchmark's size, validate it on 12,739 pairs with zero false positives, and use "
    "LMArena - which already ships confidence intervals and ties - as a positive control for what "
    "correct reporting looks like. Every figure is recomputed from public data and passes a "
    "self-test suite; the recomputed rates match each official board exactly. We argue that a "
    "board should report the resolution of its ranking alongside the ranking, at no additional "
    "data cost."
)

METADATA = {
    "upload_type": "publication",
    "publication_type": "preprint",
    "title": ("How Much of a Leaderboard Ranking Survives Its Own Sampling Error? "
              "A Paired-Resolution Audit of Five AI Benchmarks"),
    "publication_date": "2026-09-06",
    "creators": [{"name": "Väätäinen, Ilpo",
                  "affiliation": "Independent Researcher",
                  "orcid": "0009-0007-4111-4783"}],
    "description": ABSTRACT,
    "license": "cc-by-4.0",
    "access_right": "open",
    "language": "eng",
    "keywords": ["benchmark", "leaderboard", "ranking", "McNemar test",
                 "statistical significance", "SWE-bench", "MTEB", "HELM",
                 "evaluation", "sampling error", "reproducibility"],
    "related_identifiers": [
        {"identifier": "10.5281/zenodo.22342751",
         "relation": "isSupplementedBy", "scheme": "doi",
         "resource_type": "software"},
        {"identifier": "https://github.com/ipezygj/swebench-rank-audit",
         "relation": "isSupplementedBy", "scheme": "url"},
    ],
}


def token():
    for line in (pathlib.Path.home() / ".env").read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("ZENODO_TOKEN"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("ZENODO_TOKEN puuttuu ~/.env:sta")


def call(method, url, tok, data=None, ctype="application/json", raw=False):
    body = data if raw else (json.dumps(data).encode() if data is not None else None)
    req = urllib.request.Request(url + f"?access_token={tok}", data=body, method=method)
    if body is not None:
        req.add_header("Content-Type", ctype)
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            t = r.read()
            return json.loads(t) if t else {}
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read()[:500].decode('utf-8','replace')}", file=sys.stderr)
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show")
    ap.add_argument("--publish")
    a = ap.parse_args()
    tok = token()

    if a.show:
        d = call("GET", f"{API}/{a.show}", tok)
        print(json.dumps({k: d.get(k) for k in ("id", "state", "submitted", "title", "doi",
                                                "links")}, indent=2, ensure_ascii=False)[:1400])
        print("tiedostot:", [(f["filename"], f["filesize"]) for f in d.get("files", [])])
        return 0

    if a.publish:
        # ★ Zenodo-DOI on pysyva. Tama on tietoinen, erillinen askel.
        d = call("POST", f"{API}/{a.publish}/actions/publish", tok)
        print("JULKAISTU:", d.get("doi"), "|", (d.get("links") or {}).get("record_html"))
        return 0

    if not PDF.exists():
        raise SystemExit(f"{PDF} puuttuu - aja tectonic paper.tex ensin")
    dep = call("POST", API, tok, {})
    dep_id = dep["id"]
    print("luonnos", dep_id)
    call("PUT", f"{API}/{dep_id}", tok, {"metadata": METADATA})
    bucket = dep["links"]["bucket"]
    with open(PDF, "rb") as fh:
        req = urllib.request.Request(f"{bucket}/paper.pdf?access_token={tok}",
                                     data=fh.read(), method="PUT")
        req.add_header("Content-Type", "application/octet-stream")
        urllib.request.urlopen(req, timeout=300)
    print("PDF ladattu:", PDF.stat().st_size, "tavua")
    print("EI JULKAISTU. Tarkista: python zenodo_paper.py --show", dep_id)
    print("Julkaise vasta sitten: python zenodo_paper.py --publish", dep_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())

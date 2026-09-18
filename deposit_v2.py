"""Deposit version 2 of the rank-resolution paper: corrections, no new results.

Reuses the published record's metadata, appends a version note to the abstract,
fixes the one sentence in it that the corrections falsify, and uploads the
rebuilt PDF. Publishes only with --publish.
"""
import argparse, io, json, os, sys, time
import urllib.request, urllib.error

API = "https://zenodo.org/api/deposit/depositions"
PUBLISHED_ID = "22541066"
PDF = "paper_v2_candidate.pdf"

NOTE = """

<p><strong>Version 2 (18 September 2026).</strong> No measurement changed: every quantitative
result was rerun from the deposited data and reproduced, including the three parity gates, the
preregistration scoring and all four figure coordinates. What changed are sentences about other
people's work. The result that dropping a small share of preference data can change Chatbot
Arena's top model belongs to Huang, Shen, Wei and Broderick, not to Singh et al., whose paper
documents private testing and unequal data access instead; Zhang and Hardt measure a
diversity-stability trade-off rather than proving an impossibility; Card et al. (2020) and
Mogstad et al. are now cited as the prior art for the power requirement and for simultaneous
rank confidence sets; the citation to Lunardi, Della Mea, Mizzaro and Roitero named only its
last two authors and had strengthened their hedges; LMArena's published ranking rule is one plus
the number of models significantly above, which reproduces its ranking for all 242 models, and
its ten rank-2 models sit at rating positions 2 to 9, 11 and 20. Four checkable claims about the
author's own artefacts are corrected in the same pass. All of it is listed in a new section,
"Corrections in this version".</p>"""


def call(url, token, method="GET", payload=None, raw=None, ctype="application/json"):
    data = raw if raw is not None else (json.dumps(payload).encode() if payload is not None else None)
    last = None
    for attempt in range(5 if method in {"GET", "PUT", "DELETE"} else 1):
        if attempt:
            time.sleep(5 * attempt)
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", "Bearer " + token)
        if data is not None:
            req.add_header("Content-Type", ctype)
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                body = r.read()
                return r.status, (json.loads(body) if body else {})
        except urllib.error.HTTPError as e:
            last = e.code
            if e.code in (502, 503, 504):
                continue
            return e.code, {"error": e.read().decode("utf-8", "replace")[:500]}
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = type(e).__name__
            continue
    return 504, {"error": "gave up (%s)" % last}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", default=os.environ.get("ZENODO_TOKEN"))
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--finish-draft")
    a = ap.parse_args()
    if not a.token:
        sys.exit("no token")
    if not os.path.exists(PDF):
        sys.exit("missing %s" % PDF)

    meta = json.loads(io.open("_current_meta.json", encoding="utf-8").read())
    meta.pop("doi", None)
    meta.pop("relations", None)
    rt = meta.pop("resource_type", None)
    if isinstance(rt, dict):
        meta["upload_type"] = rt.get("type", "publication")
        if rt.get("subtype"):
            meta["publication_type"] = rt["subtype"]
    if isinstance(meta.get("license"), dict):
        meta["license"] = meta["license"].get("id", "cc-by-4.0")
    meta["publication_date"] = "2026-09-18"
    meta["version"] = "2"
    # the abstract's own sentence that the corrections falsify
    meta["description"] = meta["description"].replace(
        "the recomputed rates match each official board exactly",
        "the recomputed rates match each official board to its published precision, "
        "with one mismatch on the Test split that reproduces a known duplicate-id defect "
        "in a submission file") + NOTE

    if a.finish_draft:
        dep = a.finish_draft
    else:
        st, r = call("%s/%s/actions/newversion" % (API, PUBLISHED_ID), a.token, "POST")
        print("newversion:", st)
        if st >= 300:
            sys.exit(str(r))
        dep = r["links"]["latest_draft"].rsplit("/", 1)[1]
        print("draft:", dep)

    st, r = call("%s/%s" % (API, dep), a.token)
    if st >= 300:
        sys.exit(str(r))
    for f in r.get("files", []):
        st2, _ = call("%s/%s/files/%s" % (API, dep, f["id"]), a.token, "DELETE")
        print("removed old file %s: %s" % (f.get("filename"), st2))
    bucket = r["links"]["bucket"]
    st, rr = call("%s/paper.pdf" % bucket, a.token, "PUT",
                  raw=io.open(PDF, "rb").read(), ctype="application/octet-stream")
    print("upload:", st)
    if st >= 300:
        sys.exit(str(rr))
    st, r = call("%s/%s" % (API, dep), a.token, "PUT", payload={"metadata": meta})
    print("metadata:", st, r.get("error", ""))
    if st >= 300:
        sys.exit(str(r))
    if a.publish:
        st, r = call("%s/%s/actions/publish" % (API, dep), a.token, "POST")
        print("publish:", st, r.get("doi") or r.get("error", r))
        if st < 300:
            print("PUBLISHED: https://doi.org/%s" % r.get("doi"))
    else:
        print("draft ready: https://zenodo.org/deposit/%s" % dep)
    return 0


if __name__ == "__main__":
    sys.exit(main())

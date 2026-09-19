"""Version 2: corrections to what this census says about its sources. No new measurement."""
import argparse, io, json, os, sys, time
import urllib.request, urllib.error

API = "https://zenodo.org/api/deposit/depositions"
PUBLISHED_ID = "22342752"
FILES = ["../swebench-rank-audit.zip"]

NOTE = """

<p><strong>Version 2 (19 September 2026).</strong> No measurement changed. The corrections are to
PRIOR_ART.md, the file whose whole job is to describe other people's papers accurately, and to the
absence claims in LEADERBOARD_STANDARD.md. The largest: a 34-point cross-scaffold gap was credited to
two papers, when it appears in arXiv:2605.23950 alone, which itself takes it from the Holistic Agent
Leaderboard (Kapoor et al., arXiv:2510.11977, previously uncited here); arXiv:2607.22585 contains none
of those figures and reports a token-efficiency result instead, and "nearly 48" had lost its hedge.
Three entries had hedges dropped or scope widened ("several tasks in GLUE" to "GLUE tasks", "likely to
be positively correlated" to "are", "may be skewed, clustered, or multimodal" to "are skewed or
clustered"); one attributed to a saturation index a purpose its authors assign otherwise; one rendered
a statistical criterion as a fixed number of points; and two titles were truncated. Two novelty
verdicts now name closer neighbours than before: separability with confidence (arXiv:2406.11939),
which measures the separable share rather than predicting it, and the signal-and-noise framework
(arXiv:2508.13144). Neither falsifies the closed-form claim. The standard's claim that no equivalent
exists for machine-learning benchmarks is narrowed to what it can support: benchmark quality standards
exist, including BetterBench (arXiv:2411.12990); none of them says what a published ordering must
report about its own resolution.</p>"""


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
    for f in FILES:
        if not os.path.exists(f):
            sys.exit("missing %s" % f)

    st, cur = call("%s/%s" % (API, PUBLISHED_ID), a.token)
    if st >= 300:
        sys.exit("cannot read published deposition: %s" % cur)
    meta = cur["metadata"]
    for k in ("doi", "prereserve_doi", "relations"):
        meta.pop(k, None)
    meta["publication_date"] = "2026-09-19"
    meta["version"] = "2"
    if "Version 2 (18 September 2026)" not in meta["description"]:
        meta["description"] = meta["description"] + NOTE

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
    for f in FILES:
        st, rr = call("%s/%s" % (r["links"]["bucket"], os.path.basename(f)), a.token, "PUT",
                      raw=io.open(f, "rb").read(), ctype="application/octet-stream")
        print("upload %-28s %s" % (f, st))
        if st >= 300:
            sys.exit(str(rr))
    st, r2 = call("%s/%s" % (API, dep), a.token, "PUT", payload={"metadata": meta})
    print("metadata:", st, r2.get("error", ""))
    if st >= 300:
        sys.exit(str(r2))
    if a.publish:
        st, r3 = call("%s/%s/actions/publish" % (API, dep), a.token, "POST")
        print("publish:", st, r3.get("doi") or r3.get("error", r3))
        if st < 300:
            print("PUBLISHED: https://doi.org/%s" % r3.get("doi"))
    else:
        print("draft ready: https://zenodo.org/deposit/%s" % dep)
    return 0


if __name__ == "__main__":
    sys.exit(main())

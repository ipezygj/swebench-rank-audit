"""Version 2: corrections to what this census says about its sources. No new measurement."""
import argparse, io, json, os, sys, time
import urllib.request, urllib.error

API = "https://zenodo.org/api/deposit/depositions"
PUBLISHED_ID = "22835018"
FILES = [("paper_v4_candidate.pdf", "paper.pdf")]

NOTE = """

<p><strong>Version 4 (19 September 2026).</strong> One reference. The Huang et al. entry gave the
title as "Dropping Just a Handful of Preferences Can Change Top LLM Rankings"; it is "Top Large
Language Model Rankings", and the entry now names all four authors. The same truncation was corrected
in a companion paper on 18 September, and the letter sent to the first author that day said it was
fixed, which was true of one deposit and not of this one. Nothing else changed.</p>"""


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
    for local, _ in FILES:
        if not os.path.exists(local):
            sys.exit("missing %s" % local)

    st, cur = call("%s/%s" % (API, PUBLISHED_ID), a.token)
    if st >= 300:
        sys.exit("cannot read published deposition: %s" % cur)
    meta = cur["metadata"]
    for k in ("doi", "prereserve_doi", "relations"):
        meta.pop(k, None)
    meta["publication_date"] = "2026-09-19"
    meta["version"] = "4"
    if "Version 4 (19 September 2026)" not in meta["description"]:
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
    for local, remote in FILES:
        st, rr = call("%s/%s" % (r["links"]["bucket"], remote), a.token, "PUT",
                      raw=io.open(local, "rb").read(), ctype="application/octet-stream")
        print("upload %-28s %s" % (remote, st))
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

"""Round 2 of the Ahjo enumeration (see procurement_helsinki_check.py, which
found the route; this scales it). New candidate URLs were collected 2026-08-25
via ~15 varied WebSearch queries (site:ahjojulkaisu.hel.fi filetype:pdf +
category/keyword combos: vertailutaulukko, laatuvertailu, kokonaistaloudellinen
edullisuus, siivous/ateria/vartiointi/ICT/koulutus/terveys/kuljetuspalvelu,
"Sijoitus 1", "Laatukriteeri"). This fetches each, reuses the same shape
parser, and reports which are real multi-criterion scoring tables.

    python procurement_helsinki_scale.py
"""
import re
import subprocess
import sys

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/131.0 Safari/537.36"

# (label, url) - collected 2026-08-25, see chat log for the search queries.
CANDIDATES = [
    ("27210BA1 (from 'tarjousten vertailutaulukko' search)",
     "https://ahjojulkaisu.hel.fi/27210BA1-78CB-C9CB-9764-72750B400000.pdf"),
    ("8EC11F59 'tarjousten vertailutaulukko'",
     "https://ahjojulkaisu.hel.fi/8EC11F59-E82D-C525-8D16-667C15300000.pdf"),
    ("CA165D22 Diaidea Oy kuvaustarjous, Sijoitus 1",
     "https://ahjojulkaisu.hel.fi/CA165D22-D3CA-CEBA-87B5-9204BFF00002.pdf"),
    ("E0167DBD HEL 2019-012257 Steniuksenkenttä tarjouspyyntö",
     "https://ahjojulkaisu.hel.fi/E0167DBD-E9B9-C21A-BA87-6F1938C00000.pdf"),
    ("45740180 Sosiaalipalvelut hankintasopimukset",
     "https://ahjojulkaisu.hel.fi/45740180-EE22-CE5A-A5C4-79189B500003.pdf"),
    ("4E279498 Kokonaistaloudellinen edullisuus, pisteytys",
     "https://ahjojulkaisu.hel.fi/4E279498-6DAB-C8B2-8D7F-6A7DCFC00000.pdf"),
    ("0DBFE288 (kokonaistaloudellinen edullisuus search)",
     "https://ahjojulkaisu.hel.fi/0DBFE288-A06E-C59F-86C2-9FFEE8600007.pdf"),
    ("2A6A1F80 Liite 2 (rakennusurakka search)",
     "https://ahjojulkaisu.hel.fi/2A6A1F80-D758-CA89-9353-8418BA900002.pdf"),
    ("ACE07762 Liite 3 Rakennuttaminen",
     "https://ahjojulkaisu.hel.fi/ACE07762-CD87-CB47-9757-6ECB0B600000.pdf"),
    ("2B28C571 Liite 1, Arviointitaulukko",
     "https://ahjojulkaisu.hel.fi/2B28C571-FA69-CD6C-9768-8B94E5100000.pdf"),
    ("E852FC53 HEL 2019-005067 sote tarjousvertailu",
     "https://ahjojulkaisu.hel.fi/E852FC53-3DCC-C1C6-BB08-6EFA67800000.pdf"),
    ("62932B2C HEL 2022-001280 Kumpulan kärki tarjouspyyntö",
     "https://ahjojulkaisu.hel.fi/62932B2C-6AE8-C679-B88B-7F6E2C600000.pdf"),
    ("D29D40A8 TARJOUSTEN VERTAILUTAULUKKO 58H19 Sahko-tele",
     "https://ahjojulkaisu.hel.fi/D29D40A8-6300-CB56-BBE1-6FEBDDC00000.pdf"),
    ("E0D50685 (vartiointi search)",
     "https://ahjojulkaisu.hel.fi/E0D50685-323F-C038-9520-7E2523300000.pdf"),
    ("9560AA27 (vartiointi/kuljetus search)",
     "https://ahjojulkaisu.hel.fi/9560AA27-FF3E-C569-95F4-744E70800000.pdf"),
    ("D99DE4E0 (ICT-hankinta search)",
     "https://ahjojulkaisu.hel.fi/D99DE4E0-BE3C-CABE-ACD4-A00F9C700006.pdf"),
    ("5D819142 (ateriapalvelu search)",
     "https://ahjojulkaisu.hel.fi/5D819142-AE61-C8E4-B90C-7DBDD2700000.pdf"),
]

# Round 3, same day: a second open route found mid-scaling - hel.fi/static/public/hela/
# (the old per-lautakunta decision-attachment path, distinct from ahjojulkaisu.hel.fi,
# going back to 2013). Collected via ~10 more WebSearch queries varying lautakunta name
# (Kaupunkiymparisto, Kasvatus ja koulutus, Kulttuuri ja vapaa-aika) and exact phrases
# ("TARJOUSTEN VERTAILUTAULUKKO", "Yhteiset kriteerit/tiedot", "Kelvollisia tarjouksia
# yhteensa"). Two of the ahjojulkaisu-search hits from round 2 also landed here
# (Ramboll/Siemens/Sweco and L&T/Palmia/SOL, both HKL) - kept, since this list already
# dedupes on URL at run time being irrelevant (fetch cost is cheap, correctness matters
# more than saving a curl call).
CANDIDATES += [
    ("173A5982 hammashoitoyksikko, Sote 2020-08-14",
     "https://www.hel.fi/static/public/hela/vipaU32020020VH1_Terveys-_ja_paihdepalvelujen_joht/Suomi/Paatos/2020/Sote_2020-08-14_48_Pk/173A5982-4F55-CDC2-97E0-735B79400000/Liite.pdf"),
    ("D8DED740 kaasusammutusjarjestelma, Kymp 2018-05-25",
     "https://www.hel.fi/static/public/hela/vipaU753300VH1_Yksikon_johtaja/Suomi/Paatos/2018/Kymp_2018-05-25_94_Pk/D8DED740-0194-CF4A-8EF7-638656400000/Liite.pdf"),
    ("62552BB8 Stoa siivouspalvelu, KUVA 2020-01-08",
     "https://www.hel.fi/static/public/hela/vipaU48040010VH1_Kulttuurijohtaja/Suomi/Paatos/2020/KUVA_2020-01-08_2_Pk/62552BB8-3DDE-CD67-B852-6EF4DBB00001/Liite.pdf"),
    ("A9D97FDB nayttolaitteet 312687, Nepep 2020-10-09",
     "https://www.hel.fi/static/public/hela/vipaU320200101010VH1_Neuvola-_ja_perhetyon_paallik/Suomi/Paatos/2020/Sote_2020-10-09_Nepep_4_Pk/A9D97FDB-B988-C80E-BBA3-750CA9600000/Liite.pdf"),
    ("E9606988 lumenpoisto katoilta 44H13, HKLjk 2013-11-21",
     "https://www.hel.fi/static/public/hela/Liikennelaitos_-liikelaitoksen_johtokunta_(HKL)/Suomi/Paatostiedote/2013/HKL_2013-11-21_HKLjk_14_Pt/E9606988-0044-4328-B456-041C31606187/Liite.pdf"),
    ("03299086 kirjekuoret H003-20, Keha 2020-05-04",
     "https://www.hel.fi/static/public/hela/vipaU02150015VH1_Hankintajohtaja/Suomi/Paatos/2020/Keha_2020-05-04_148_Pk/03299086-038D-C781-BA6F-71C098900000/Liite.pdf"),
    ("E00A618B vanhusten asumispalvelu 305075, Sote 2020-10-27",
     "https://www.hel.fi/static/public/hela/vipaU32020030VH1_Sairaala-_kuntoutus-_ja_hoivapalv/Suomi/Paatos/2020/Sote_2020-10-27_47_Pk/E00A618B-9D0B-C2DB-9577-756A68000000/Liite.pdf"),
    ("FDAE07AE nayttolaitteet 331557, Taj 2021-03-04",
     "https://www.hel.fi/static/public/hela/vipaU320200VH1_Sosiaali-_ja_terveystoimialan_toimi/Suomi/Paatos/2021/Sote_2021-03-04_Taj_35_Pk/FDAE07AE-C29E-CEA0-BBF7-77F741800000/Liite.pdf"),
    ("3B68ADE9 L&T/Palmia/SOL siivous, HKLjku 2020-06-04",
     "https://www.hel.fi/static/public/hela/Liikenneliikelaitoksen_johtokunta/Suomi/Paatos/2020/Kymp_2020-06-04_HKLjku_10_Pk/3B68ADE9-465A-CFCF-B862-725020700000/Liite.pdf"),
    ("56215C1F Ramboll/Siemens/Sweco sahkokonsultaatio, HKLjku 2019-12-12",
     "https://www.hel.fi/static/public/hela/Liikenneliikelaitoksen_johtokunta/Suomi/Paatos/2019/Kymp_2019-12-12_HKLjku_24_Pk/56215C1F-474C-C747-941C-6EC592C00000/Liite.pdf"),
    ("91237622 Kylk 2021-01-26 liite",
     "https://www.hel.fi/static/public/hela/Kaupunkiymparistolautakunta/Suomi/Paatos/2021/Kymp_2021-01-26_Kylk_3_Pk/91237622-7759-C3A8-99DB-766627500000/Liite.pdf"),
    ("8E789419 Kylk 2022-02-08 liite",
     "https://www.hel.fi/static/public/hela/Kaupunkiymparistolautakunta/Suomi/Paatos/2022/Kymp_2022-02-08_Kylk_5_Pk/8E789419-ED8C-C149-BB1F-7E71FE600000/Liite.pdf"),
    ("0A4876EB Kylk 2022-01-18 liite",
     "https://www.hel.fi/static/public/hela/Kaupunkiymparistolautakunta/Suomi/Paatos/2022/Kymp_2022-01-18_Kylk_2_Pk/0A4876EB-2AF9-C490-94AF-7D9E1C800000/Liite.pdf"),
    ("DD2E3C73 Vesalan nuorisotalo siivous tarjouspyynto, KUVA 2019-12-10",
     "https://www.hel.fi/static/public/hela/vipaU48040030VH1_Nuorisoasiainjohtaja/Suomi/Paatos/2019/KUVA_2019-12-10_8_Pk/DD2E3C73-4753-C302-BB68-6EEF21500000/Liite.pdf"),
    ("662FDD54 KUVA 2022-09-13 liite",
     "https://www.hel.fi/static/public/hela/Kulttuuri-_ja_vapaa-aikalautakunta/Suomi/Paatos/2022/KUVA_2022-09-13_Kuvalk_17_Pk/662FDD54-86FC-C41E-B86D-833C33200000/Liite.pdf"),
    ("D5B2AC19 Oodin julkitilakalusteet 186926, KUVA 2018-06-19",
     "https://www.hel.fi/static/public/hela/Kulttuuri-_ja_vapaa-aikalautakunta/Suomi/Paatos/2018/KUVA_2018-06-19_Kuvalk_12_Pk/D5B2AC19-6747-CCB0-8D23-63F331B00000/Liite.pdf"),
    ("5C18598A Tarjousvertailu, Sote 2020-12-18",
     "https://www.hel.fi/static/public/hela/vipaU32020030VH1_Sairaala-_kuntoutus-_ja_hoivapalv/Suomi/Paatos/2020/Sote_2020-12-18_67_Pk/5C18598A-85C7-C96C-B97C-7674D9000000/Liite.pdf"),
]

# Round 4: exact-phrase sweeps on hel.fi/static/public/hela ("LAATUPISTEYTYS",
# "HINTAPISTEYTYS", "Vertailuun valittuja tarjouksia yhteensa" - the last one's
# result snippets gave J directly, including a 76-bidder outlier worth checking).
CANDIDATES += [
    ("638BC698 raitiovaunun istuinpaalliset, Kymp 2021-07-23",
     "https://www.hel.fi/static/public/hela/vipaU7500110VH1_Yksikon_johtaja/Suomi/Paatos/2021/Kymp_2021-07-23_45_Pk/638BC698-6A6E-CE15-BBB2-7ACDC1100000/Liite.pdf"),
    ("F77AA1D5 ikaantyneiden ymparivuorokautinen palveluasuminen, 76 tarjousta, Sote 2020-10-27",
     "https://www.hel.fi/static/public/hela/vipaU32020030VH1_Sairaala-_kuntoutus-_ja_hoivapalv/Suomi/Paatos/2020/Sote_2020-10-27_47_Pk/F77AA1D5-CCC8-C9D9-8D5D-75699A400003/Ikaantyneiden_ymparivuorokautisen_palveluasumisen_.pdf"),
    ("FBA341FF Liikuntavirasto tarjouspyynto, LIV 2016-10-21",
     "https://www.hel.fi/static/public/hela/vipa471131VH1_Osastopaallikko/Suomi/Paatos/2016/LIV_2016-10-21_32_Pk/FBA341FF-F697-CE89-A47C-57E669D00001/Liite.pdf"),
]

# Round 5: more toimiala/lautakunta sweeps (Kiinteisto/Ympäristö, varhaiskasvatus,
# pelastuslaitos) plus exact phrases ("Laatukriteeri 2", year 2023).
CANDIDATES += [
    ("42472407 Taloustutka Oy vs Suomen Asiakastieto Oy, arviointikriteerit, Keha 2019-08-19",
     "https://www.hel.fi/static/public/hela/vipaU021200VH1_Elinkeinojohtaja/Suomi/Paatos/2019/Keha_2019-08-19_Ekj_84_Pk/42472407-EB42-CF26-A813-6C8AE4000000/Liite.pdf"),
    ("9623E5B6 HKLn kiinteistojen puhtaanapitopalvelut, HKLjku 2020-06-16",
     "https://www.hel.fi/static/public/hela/Liikenneliikelaitoksen_johtokunta/Suomi/Paatos/2020/Kymp_2020-06-16_HKLjku_11_Pk/9623E5B6-F005-C9FE-86CE-72EA0CE00000/HKLn_eraiden_kiinteistojen_puhtaanapitopalvelut.pdf"),
    ("D8B334F3 Stara tarjouspyynto HEL 2017-013910, Starajk 2018-05-24",
     "https://www.hel.fi/static/public/hela/Rakentamispalveluliikelaitoksen_johtokunta/Suomi/Paatos/2018/Keha_2018-05-24_Starajk_6_Pk/D8B334F3-D2BA-C495-8C1D-6363D2500000/Liite.pdf"),
    ("96CD60B8 tyohonkuntoutuspalvelut H133-16, Pesop 2016-12-12",
     "https://www.hel.fi/static/public/hela/vipa811310_Osastopaallikko/Suomi/Paatos/2016/Sote_2016-12-12_Pesop_65_Pk/96CD60B8-7AF5-C3CC-87CD-58F1FEB00000/Liite.pdf"),
    ("CDE20CF3 tarjouspyynto H030-14, Sotelk 2014-09-23",
     "https://www.hel.fi/static/public/hela/Sosiaali-_ja_terveyslautakunta/Suomi/Paatostiedote/2014/Sote_2014-09-23_Sotelk_18_Pt/CDE20CF3-46E9-44EC-91D7-64CED2BD16CC/Liite.pdf"),
    ("E399E4C3 asiantuntijapalveluiden hankinta puitejarjestely, Klk 2015-06-25",
     "https://www.hel.fi/static/public/hela/Kiinteistolautakunta/Suomi/Paatostiedote/2015/Kv_2015-06-25_Klk_13_Pt/E399E4C3-BF91-4D27-A99F-10D15BC48D97/Asiantuntijapalveluiden_hankinta_puitejarjestely.pdf"),
    ("B3AD9E82 tarjousten vertailutaulukko, Kasko 2019-07-01",
     "https://www.hel.fi/static/public/hela/vipaU420300506010VH1_Ict-kehityspaallikko/Suomi/Paatos/2019/Kasko_2019-07-01_9_Pk/B3AD9E82-ED5F-CCED-8CDD-6B9D95600000/Liite.pdf"),
]

# Round 7: pivot from keyword-varied WebSearch (saturating) to crawling
# paatokset.hel.fi case pages directly - site:paatokset.hel.fi/fi/asia searches
# surface a case page, and the case page's server-rendered HTML links every
# ahjojulkaisu.hel.fi attachment for that case (not just the one snippet-matched
# doc). paatokset_case_crawl.sh does the fetch+extract; results land in
# case_attachments_*.txt as 'hel-YYYY-NNNNNN<TAB>url' lines, loaded here.
import os as _os
for _batch in ["case_attachments_batch1.txt", "case_attachments_batch2.txt", "case_attachments_batch3.txt", "case_attachments_batch4.txt", "case_attachments_batch5.txt", "case_attachments_batch6.txt"]:
    _path = _os.path.join(_os.path.dirname(__file__), _batch)
    if _os.path.exists(_path):
        with open(_path, encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line:
                    continue
                _case, _url = _line.split("\t")
                CANDIDATES.append((f"{_case} ({_url.rsplit('/', 1)[-1][:8]})", _url))


def fetch(url: str) -> bytes:
    # -k: www.hel.fi's cert chain is incomplete (missing intermediate) and
    # Windows schannel won't fetch it via AIA like a browser does, so curl
    # fails closed with SEC_E_UNTRUSTED_ROOT on an otherwise-valid cert.
    # ahjojulkaisu.hel.fi doesn't need this. Confirmed 2026-08-25 the content
    # behind it is a normal public %PDF, not a MITM substitution.
    r = subprocess.run(["curl", "-sLk", "--max-time", "30", "-A", UA, url],
                        capture_output=True)
    return r.stdout


def pdf_bidder_criterion_shape(pdf_bytes: bytes):
    """Broadened after round 1 (only caught 'Laatukriteeri N', the one doc it
    was written against): real Ahjo vertailutaulukot instead say 'Laatupisteet'
    as a row label, or number their criteria as 'N. <name> NN pistetta' in the
    RFP's own criteria section (e.g. '1. Referenssit 50 pistetta')."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(pdf_bytes)
        path = f.name
    r = subprocess.run(["pdftotext", "-layout", "-enc", "UTF-8", path, "-"],
                        capture_output=True)
    txt = r.stdout.decode("utf-8", "replace")

    numbered = len(set(re.findall(r"^\s*(\d+)\.\s+\S.{0,60}?\d+\s*pistet",
                                   txt, re.M)))
    named = len(set(re.findall(r"Laatukriteeri\s+(\d+)", txt)))
    n_criteria = max(numbered, named)

    has_quality_row = bool(re.search(r"Laatupisteet|LAATU- JA HINTAVERTAILU|"
                                      r"[Hh]inta[- ]ja laatuvertailu", txt))
    has_price = bool(re.search(r"Kokonaishinta|HINTAPISTEYTYS|hintapisteet",
                                txt, re.I))
    if n_criteria == 0 and has_quality_row and has_price:
        n_criteria = 2  # at least price + quality, exact count unclear
    elif n_criteria and has_price:
        n_criteria += 1  # price is itself a criterion alongside numbered quality ones
    elif n_criteria == 0 and has_price:
        n_criteria = 1

    j_ranked = len(set(re.findall(r"Sijoitus\s+\d+", txt)))
    j_declared = re.search(r"(?:Kelvollisia|Vertailuun valittuja) tarjouksia"
                            r" yhteens.?:\s*(\d+)", txt)
    # Fallback: 'TARJOUSTEN VERTAILUTAULUKKO <Bidder 1>  <Bidder 2>  ...' is
    # often literally the first line, bidder names separated by runs of
    # whitespace from the -layout column spacing (procurement_helsinki_check.py's
    # original heuristic - reused here since round 3's docs don't declare a
    # count but do print the header row this way, e.g. 3B68ADE9).
    # Some docs title it 'VERTAILUTAULUKKO 28H21 <bidders...>' (a case code
    # between the label and the names, not 'TARJOUSTEN VERTAILUTAULUKKO') -
    # 638BC698 is the doc this broke on. j_header is a rough signal either
    # way; MULTI hits still get manually re-read before going in the overlay.
    header = re.search(r"VERTAILUTAULUKKO(?:\s+\S+)?(\s{2,}.*)",
                        txt.splitlines()[0] if txt else "")
    j_header = len([c for c in re.split(r"\s{2,}", header.group(1))
                    if c.strip()]) if header else 0
    j = int(j_declared.group(1)) if j_declared else max(j_ranked, j_header)
    return txt, n_criteria, j


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    rows = []
    for label, url in CANDIDATES:
        body = fetch(url)
        is_pdf = body[:4] == b"%PDF"
        shape = None
        if is_pdf:
            txt, n, j = pdf_bidder_criterion_shape(body)
            shape = (j, n)
        print(f"[{'ok  ' if is_pdf else 'FAIL'}] {label}")
        print(f"      {url}")
        print(f"      {len(body)} bytes, PDF={is_pdf}, (bidders_ranked, criteria)={shape}")
        rows.append((label, url, is_pdf, shape))

    multi = [r for r in rows if r[3] and r[3][1] and r[3][1] >= 2]
    print(f"\n{len(multi)}/{len(rows)} new candidates are real multi-criterion "
          f"scoring tables (n>=2 criteria detected).")
    for label, url, _, shape in multi:
        print(f"  MULTI {shape}: {label}\n        {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

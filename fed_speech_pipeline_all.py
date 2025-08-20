# powell_speech_probabilities.py
# ------------------------------------------------------------
# Collects Jerome Powell speeches (2018->today), builds a text corpus,
# computes word probabilities (All vs Formal policy-related),
# and compares those probabilities to market-implied odds
# from a markets JSON (Kalshi-like) you provide.
#
# Outputs:
#   data/scraped_index.csv                 (index of speeches)
#   data/speeches/*.txt                    (cached cleaned texts)
#   out/word_probs_all.csv                 (P(word|All Speeches))
#   out/word_probs_formal.csv              (P(word|Formal Speeches))
#   out/contract_compare.csv               (historical vs market; mispricing flags)
#   out/top_tokens_all.csv                 (summary token counts)
#   out/top_tokens_formal.csv              (summary token counts)
#   out/summary.txt                        (quick stats)
#
# Dependencies:
#   pip install requests beautifulsoup4 lxml pdfminer.six pandas regex unidecode python-dateutil tqdm
#
# Notes:
# - Scrapes official FRB "Speeches" pages by year and filters to Powell.
# - Also handles PDF speeches with pdfminer.six.
# - "Formal policy-related" heuristic: title/body contains any of:
#       ["jackson hole", "monetary policy", "economic outlook", "price stability"]
#   You can adjust the heuristics easily (see IS_FORMAL function).
# - Contract word mapping is controlled via WORD_PATTERNS below. Adjust freely.
# ------------------------------------------------------------

from __future__ import annotations

import os
import re
import json
import time
import math
import pathlib
import logging
import datetime as dt
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from dateutil import parser as dateparser
from pdfminer.high_level import extract_text as pdf_extract_text
import pandas as pd
from tqdm import tqdm
from unidecode import unidecode

# -----------------------
# Config
# -----------------------
BASE_YEAR_START = 2018
BASE_YEAR_END = dt.date.today().year
FRB_YEAR_URL = "https://www.federalreserve.gov/newsevents/speech/{year}-speeches.htm"
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; PowellSpeechScraper/1.0; +https://www.federalreserve.gov/)"}
SLEEP_BETWEEN_REQUESTS = (0.4, 1.1)  # (min, max) seconds polite delay

DATA_DIR = pathlib.Path("data")
OUT_DIR = pathlib.Path("out")
SPEECH_DIR = DATA_DIR / "speeches"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
SPEECH_DIR.mkdir(parents=True, exist_ok=True)

LOGGING_LEVEL = logging.INFO
logging.basicConfig(level=LOGGING_LEVEL, format="%(asctime)s %(levelname)s: %(message)s")

# If you prefer to paste your `markets` JSON directly, set MARKETS_JSON = r""" ... """ and leave MARKETS_JSON_PATH=None
MARKETS_JSON_PATH = "markets.json"
MARKETS_JSON: Optional[str] = None  # r"""PASTE THE RAW JSON FROM YOUR PROMPT HERE IF YOU DON'T WANT TO USE A FILE"""

# -----------------------
# Helper: polite sleep
# -----------------------
import random


def _nap():
    time.sleep(random.uniform(*SLEEP_BETWEEN_REQUESTS))


# -----------------------
# Word patterns (contract → regex or (regex, threshold))
# The key should match the "name" field in your markets JSON.
# Threshold variants (e.g., "Labor (30+ times)") defined explicitly below.
# -----------------------
WORD_PATTERNS: Dict[str, re.Pattern] = {
    "Trump": re.compile(r"\btrump\b", re.I),
    "Projection": re.compile(r"\bprojection(s)?\b", re.I),
    "Good afternoon": re.compile(r"\bgood afternoon\b", re.I),
    "Russia": re.compile(r"\brussia(n)?\b", re.I),
    "Pandemic": re.compile(r"\bpandemic(s)?\b", re.I),
    "Median": re.compile(r"\bmedian(s)?\b", re.I),
    "Administration": re.compile(r"\badministration\b", re.I),
    "Tariff": re.compile(r"\btariff(s)?\b", re.I),
    "Renovation": re.compile(r"\brenovation(s)?\b", re.I),
    "Regulator/ regulatory / regulation": re.compile(r"\bregulat(?:or|ory|ion|ions|ors)\b", re.I),
    "Overheat": re.compile(r"\boverheat(?:ed|ing)?\b", re.I),
    "Michelle / Bowman": re.compile(r"\bmichelle\b|\bbowman\b", re.I),
    "Layoff": re.compile(r"\blayoff(s)?\b|\blay off\b", re.I),
    "Good morning": re.compile(r"\bgood morning\b", re.I),
    "Energy": re.compile(r"\benergy\b", re.I),
    "Dollar": re.compile(r"\bdollar(s)?\b", re.I),
    "Dissent": re.compile(r"\bdissent(s|ed|ing)?\b", re.I),
    "Cut": re.compile(r"\bcut(s|ting)?\b", re.I),
    "Crypto / Bitcoin": re.compile(r"\b(bitcoin|crypto(?:currency)?|stablecoin(s)?)\b", re.I),
    "Credit": re.compile(r"\bcredit\b", re.I),
    "Consumer confidence": re.compile(r"\bconsumer confidence\b", re.I),
    "Balance of risks": re.compile(r"\bbalance of risks\b", re.I),
    "Anchor": re.compile(r"\banchor(?:ed|ing)?\b", re.I),
    "Transitory": re.compile(r"\btransitory\b", re.I),
    "Symposium": re.compile(r"\bsymposium\b", re.I),
    "Meeting": re.compile(r"\bmeeting(s)?\b", re.I),
    "Chair": re.compile(r"\bchair\b", re.I),
    "Dual": re.compile(r"\bdual\b.{0,20}\bmandate\b|\bdual mandate\b", re.I),
}

# Threshold contracts (count >= N of a base token)
THRESHOLD_CONTRACTS = {
    "Tariff (10+ times)": (re.compile(r"\btariff(s)?\b", re.I), 10),
    "Labor (30+ times)": (re.compile(r"\blabor\b", re.I), 30),
    "Labor (40+ times)": (re.compile(r"\blabor\b", re.I), 40),
}


# -----------------------
# Scraping & parsing
# -----------------------
def get_html(url: str) -> str:
    r = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
    r.raise_for_status()
    return r.text


def get_binary(url: str) -> bytes:
    r = requests.get(url, headers=REQUEST_HEADERS, timeout=60)
    r.raise_for_status()
    return r.content


@dataclass
class SpeechItem:
    date: dt.date
    title: str
    url: str
    is_pdf: bool


def find_powell_items_for_year(year: int) -> List[SpeechItem]:
    url = FRB_YEAR_URL.format(year=year)
    try:
        html = get_html(url)
    except Exception as e:
        logging.warning(f"Failed to fetch year page {url}: {e}")
        return []
    soup = BeautifulSoup(html, "lxml")

    # Year pages list many officials. We filter entries where the speaker contains "Powell".
    items: List[SpeechItem] = []
    # Each entry often sits in a <div class="row"> with date, title link, and speaker line.
    for block in soup.select("div.row"):
        text = " ".join(block.stripped_strings)
        if "Powell" not in text:
            continue
        # Date usually appears at the start; title link is <a> inside
        a = block.find("a", href=True)
        if not a:
            continue
        title = a.get_text(strip=True)
        href = a["href"]
        # Build absolute URL
        if href.startswith("/"):
            url_abs = "https://www.federalreserve.gov" + href
        else:
            url_abs = href

        # Parse date: often the year page has a sibling date, or embed in block
        date_str = None
        # Try to find a span or strong that looks like a date
        for maybe in block.find_all(["span", "strong", "em"]):
            t = maybe.get_text(" ", strip=True)
            if re.search(r"\b\d{1,2}/\d{1,2}/\d{4}\b", t):
                date_str = re.search(r"\b\d{1,2}/\d{1,2}/\d{4}\b", t).group(0)
                break
        if not date_str:
            # fallback: try to parse from text
            m = re.search(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w* \d{1,2}, \d{4}\b", text)
            if m:
                date_str = m.group(0)
        if not date_str:
            # last resort: infer from URL structure powellYYYYMMDD*.htm
            m = re.search(r"powell(\d{8})", url_abs)
            if m:
                d = m.group(1)
                date_str = f"{d[4:6]}/{d[6:8]}/{d[0:4]}"
            else:
                # otherwise just set Jan 1 of year
                date_str = f"01/01/{year}"

        try:
            d = dateparser.parse(date_str).date()
        except Exception:
            d = dt.date(year, 1, 1)

        is_pdf = url_abs.lower().endswith(".pdf")
        items.append(SpeechItem(date=d, title=title, url=url_abs, is_pdf=is_pdf))

    return items


def extract_text_from_html(url: str, html: str, try_pdf_first: bool = True) -> str:
    """
    Parse FRB speech pages correctly:
    - Prefer the official transcript PDF if a link exists.
    - Otherwise, extract actual speech paragraphs from #article and discard page chrome.
    """
    soup = BeautifulSoup(html, "lxml")

    # ---------- Prefer the transcript PDF if present ----------
    if try_pdf_first:
        # Common places a transcript link appears
        pdf_link = (
            soup.select_one('a[href$=".pdf"][href*="/newsevents/speech/files/"]')
            or soup.select_one('#videoDetails45469 a[href$=".pdf"]')  # id varies across pages, this is fine as a fallback
            or soup.select_one('#content a[href$=".pdf"]')
        )
        if pdf_link and pdf_link.get("href"):
            pdf_url = urljoin(url, pdf_link["href"])
            try:
                b = get_binary(pdf_url)
                text_pdf = extract_text_from_pdf_bytes(b)
                # If we got a plausible transcript, return it
                if text_pdf and len(text_pdf) > 200:
                    return text_pdf
            except Exception as e:
                logging.warning(f"PDF fallback failed ({pdf_url}): {e}")

    # ---------- HTML fallback: extract real speech text ----------
    # Anchor to the article container first, then tidy it
    art = soup.select_one("#article") or soup.select_one("#content") or soup

    # Remove obvious non-body elements
    for sel in [
        ".page-header",
        ".header-group",
        ".breadcrumb",
        ".shareDL",
        ".watchLive",
        ".panel-related",
        ".panel",
        ".embed-responsive",
        ".video-js",
        "script",
        "style",
        ".sr-only",
        "#videoDetails45469",
        ".heading",
        ".list-unstyled",
        "noscript",
    ]:
        for node in art.select(sel):
            node.decompose()

    # Collect paragraphs that are not meta
    paras = []
    for p in art.select("p"):
        classes = set(p.get("class", []))
        # Drop meta lines (date/speaker/location etc.)
        if {"article__time", "speaker", "location"} & classes:
            continue
        t = p.get_text(" ", strip=True)
        # Skip empty or boilerplate
        if not t:
            continue
        # Skip single-word ‘Share’ etc.
        if t.lower() in {"share", "watch live"}:
            continue
        paras.append(t)

    # If we got paragraphs, join them; otherwise fall back to all text in art
    if paras:
        text = "\n\n".join(paras)
    else:
        text = art.get_text(" ", strip=True)

    # Clean up
    text = unidecode(text)
    text = re.sub(r"[ \t]+", " ", text).strip()

    # IMPORTANT: remove the previous "first sentence" heuristic — it was causing truncation.
    return text


def extract_text_from_pdf_bytes(b: bytes) -> str:
    # pdfminer.six on temp file
    tmp = SPEECH_DIR / "_tmp.pdf"
    tmp.write_bytes(b)
    try:
        text = pdf_extract_text(str(tmp)) or ""
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
    text = unidecode(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -----------------------
# Formal policy-related heuristic
# -----------------------
FORMAL_KEYWORDS = [
    "jackson hole",
    "monetary policy",
    "economic outlook",
    "price stability",
]


def IS_FORMAL(title: str, body: str) -> bool:
    t = title.lower()
    b = body.lower()
    return any(k in t for k in FORMAL_KEYWORDS) or any(k in b for k in FORMAL_KEYWORDS)


# -----------------------
# Corpus build
# -----------------------
def build_powell_corpus() -> pd.DataFrame:
    rows = []
    for year in range(BASE_YEAR_START, BASE_YEAR_END + 1):
        logging.info(f"Fetching year {year} index …")
        items = find_powell_items_for_year(year)
        for it in items:
            rows.append(
                {
                    "date": it.date.isoformat(),
                    "title": it.title,
                    "url": it.url,
                    "is_pdf": it.is_pdf,
                }
            )
        _nap()
    df = pd.DataFrame(rows).drop_duplicates(subset=["url"]).sort_values("date")
    df.to_csv(DATA_DIR / "scraped_index.csv", index=False)
    logging.info(f"Indexed {len(df)} Powell items.")
    return df


def fetch_and_cache_text(url: str, is_pdf: bool) -> str:
    """
    Fetch page and return full transcript text.
    - If the URL is a PDF (or marked is_pdf), parse via pdfminer.
    - If HTML, prefer the embedded transcript PDF when present, otherwise parse paragraphs.
    Caches the final cleaned text in data/speeches/.
    """
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", url.strip("/"))[:150]
    path = SPEECH_DIR / f"{slug}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8", errors="ignore")

    logging.info(f"Downloading: {url}")
    text = ""
    try:
        if is_pdf or url.lower().endswith(".pdf"):
            b = get_binary(url)
            text = extract_text_from_pdf_bytes(b)
        else:
            html = get_html(url)
            # NOTE: pass try_pdf_first=True so HTML pages still prefer the transcript PDF
            text = extract_text_from_html(url, html, try_pdf_first=True)
    except Exception as e:
        logging.warning(f"Failed to fetch/parse {url}: {e}")
        text = ""

    text = (text or "").strip()
    path.write_text(text, encoding="utf-8")
    _nap()
    return text


# -----------------------
# Analytics
# -----------------------
STOPWORDS = set(
    """
a an the and or but if while of to for in on by with from as at this that those these is are was were be been being it its into not no we you i our their his her they them there here such
""".split()
)

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]+")


def tokenize(text: str) -> List[str]:
    return [w.lower() for w in TOKEN_RE.findall(text)]


def count_pattern(text: str, pat: re.Pattern) -> int:
    return len(re.findall(pat, text))


def compute_word_probs(df: pd.DataFrame, subset_mask: pd.Series) -> pd.DataFrame:
    # Coerce mask to a boolean Series aligned to df.index
    if not isinstance(subset_mask, pd.Series):
        subset_mask = pd.Series(subset_mask, index=df.index)
    elif not subset_mask.index.equals(df.index):
        subset_mask = subset_mask.reindex(df.index, fill_value=False)

    sub = df[subset_mask].copy()
    N = len(sub)
    rows = []
    for name, pat in WORD_PATTERNS.items():
        hits = 0
        total_mentions = 0
        for t in sub["text"]:
            c = count_pattern(t, pat)
            total_mentions += c
            if c > 0:
                hits += 1
        p = hits / N if N else float("nan")
        rows.append(
            {
                "contract": name,
                "speeches": N,
                "hit_speeches": hits,
                "mentions_total": total_mentions,
                "p_hist": p,
            }
        )

    for name, (pat, k) in THRESHOLD_CONTRACTS.items():
        hits = 0
        total_mentions = 0
        for t in sub["text"]:
            c = count_pattern(t, pat)
            total_mentions += c
            if c >= k:
                hits += 1
        p = hits / N if N else float("nan")
        rows.append(
            {
                "contract": name,
                "speeches": N,
                "hit_speeches": hits,
                "mentions_total": total_mentions,
                "p_hist": p,
            }
        )
    return pd.DataFrame(rows).sort_values(["contract"])


def summarize_tokens(df: pd.DataFrame, subset_mask: pd.Series, topn: int = 200) -> pd.DataFrame:
    sub = df[subset_mask].copy()
    counts: Dict[str, int] = {}
    for t in sub["text"]:
        for tok in tokenize(t):
            if tok in STOPWORDS:
                continue
            counts[tok] = counts.get(tok, 0) + 1
    top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:topn]
    return pd.DataFrame(top, columns=["token", "count"])


# -----------------------
# Markets parsing & comparison
# -----------------------
def load_markets() -> List[Dict]:
    if MARKETS_JSON is not None:
        data = json.loads(MARKETS_JSON)
    else:
        with open(MARKETS_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    # Expect an object with key "markets": [ ... ]
    if isinstance(data, dict) and "markets" in data:
        return data["markets"]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError("Unexpected markets JSON structure. Provide {'markets': [...]} or a list.")


def implied_prob(mkt: Dict) -> Optional[float]:
    yes_bid = mkt.get("yes_bid")
    yes_ask = mkt.get("yes_ask")
    last_price = mkt.get("last_price")
    # Use mid of bid/ask if both present, else last_price
    if isinstance(yes_bid, (int, float)) and isinstance(yes_ask, (int, float)) and yes_bid >= 0 and yes_ask > 0:
        return (0.5 * (yes_bid + yes_ask)) / 100.0
    if isinstance(last_price, (int, float)) and last_price >= 0:
        return last_price / 100.0
    return None


def normalize_contract_name(raw: str) -> str:
    # Markets "name" field should match our keys; if not, try to map lightly
    # e.g., "Regulator/ regulatory / regulation" might come as that exact string.
    return raw.strip()


def compare_to_markets(hist_df: pd.DataFrame, markets: List[Dict]) -> pd.DataFrame:
    # Make lookup from contract → p_hist
    pmap = {row["contract"]: row["p_hist"] for _, row in hist_df.iterrows()}
    rows = []
    for m in markets:
        cname = normalize_contract_name(m.get("name", m.get("title", "")))
        if not cname:
            continue
        p_hist = pmap.get(cname)
        p_mkt = implied_prob(m)
        if p_mkt is None or p_hist is None or math.isnan(p_hist):
            continue
        diff = p_hist - p_mkt
        rows.append(
            {
                "contract": cname,
                "p_hist": p_hist,
                "p_mkt": p_mkt,
                "hist_minus_mkt": diff,
                "ticker": m.get("ticker_name"),
                "id": m.get("id"),
            }
        )
    out = pd.DataFrame(rows).sort_values("hist_minus_mkt")
    return out


# -----------------------
# Main
# -----------------------
def main():
    # 1) Build index & fetch texts
    idx = build_powell_corpus()
    logging.info("Fetching & caching speech texts …")
    texts = []
    for row in tqdm(idx.itertuples(index=False), total=len(idx)):
        txt = fetch_and_cache_text(row.url, bool(row.is_pdf))
        texts.append(txt)
    idx["text"] = texts

    # basic stats
    idx["n_words"] = idx["text"].apply(lambda s: len(tokenize(s)))
    idx["formal"] = idx.apply(lambda r: IS_FORMAL(r["title"], r["text"]), axis=1)

    idx.to_csv(DATA_DIR / "powell_speech_corpus.csv", index=False)

    # 2) Word probabilities (All vs Formal)
    mask_all = pd.Series(True, index=idx.index)
    mask_formal = idx["formal"].fillna(False)

    probs_all = compute_word_probs(idx, mask_all)
    probs_formal = compute_word_probs(idx, mask_formal)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    probs_all.to_csv(OUT_DIR / "word_probs_all.csv", index=False)
    probs_formal.to_csv(OUT_DIR / "word_probs_formal.csv", index=False)

    # 3) Summary tokens
    top_all = summarize_tokens(idx, mask_all, topn=300)
    top_formal = summarize_tokens(idx, mask_formal, topn=300)
    top_all.to_csv(OUT_DIR / "top_tokens_all.csv", index=False)
    top_formal.to_csv(OUT_DIR / "top_tokens_formal.csv", index=False)

    # 4) Compare to markets (use All by default for mispricing; you can also compare with Formal)
    markets = load_markets()
    comp_all = compare_to_markets(probs_all, markets)
    comp_formal = compare_to_markets(probs_formal, markets)

    comp_all.to_csv(OUT_DIR / "contract_compare_ALL.csv", index=False)
    comp_formal.to_csv(OUT_DIR / "contract_compare_FORMAL.csv", index=False)

    # 5) Quick human-readable summary
    with open(OUT_DIR / "summary.txt", "w", encoding="utf-8") as f:
        f.write("Powell Speeches Corpus Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Total speeches indexed: {len(idx)}\n")
        if len(idx):
            f.write(f"Date range: {idx['date'].min()} → {idx['date'].max()}\n")
            f.write(f"Formal subset size: {mask_formal.sum()} (heuristic)\n")
            f.write(f"Avg words per speech (All): {idx['n_words'].mean():,.0f}\n")
            f.write("\nTop likely contracts historically (All):\n")
            f.write(probs_all.sort_values("p_hist", ascending=False).head(10).to_string(index=False))
            f.write("\n\nMost underpriced by history (hist << market) — ALL:\n")
            f.write(comp_all.sort_values("hist_minus_mkt").head(10).to_string(index=False))
            f.write("\n\nMost overpriced by history (hist >> market) — ALL:\n")
            f.write(comp_all.sort_values("hist_minus_mkt", ascending=False).head(10).to_string(index=False))
            f.write("\n")

    print("\nDone. See the 'out/' folder for CSVs and summary.\n")


if __name__ == "__main__":
    main()

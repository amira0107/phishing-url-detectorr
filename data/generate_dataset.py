"""
generate_dataset.py
--------------------
Builds a labeled dataset of URLs (legitimate=0 / phishing=1) for
training the phishing detector.

NOTE ON DATA SOURCE:
In a real-world deployment you would use public datasets such as:
  - PhishTank (https://phishtank.org/developer_info.php)
  - UCI Phishing Websites Dataset
  - OpenPhish feeds
  - Tranco / Alexa top domains for legitimate examples

This script generates a realistic SYNTHETIC dataset using
rule-based templates (legit brand-like domains vs. phishing-style
lookalike domains with typosquatting, IPs, suspicious keywords,
extra subdomains, etc.) so the whole project runs fully offline
and is reproducible. Swap `build_dataset()` with a loader for a
real CSV (e.g. from PhishTank) when you have network access -
the rest of the pipeline (features -> model -> API) stays the same.
"""

import random
import csv
import os

random.seed(42)

LEGIT_BRANDS = [
    "google", "youtube", "facebook", "amazon", "wikipedia", "twitter",
    "instagram", "linkedin", "microsoft", "apple", "netflix", "github",
    "reddit", "yahoo", "ebay", "paypal", "spotify", "dropbox", "adobe",
    "salesforce", "zoom", "slack", "airbnb", "booking", "coursera"
]

LEGIT_TLDS = [".com", ".org", ".net", ".io", ".co", ".edu"]

LEGIT_PATHS = [
    "", "/", "/home", "/about", "/products", "/blog/2024/article",
    "/user/settings", "/docs/api", "/contact", "/help/faq",
    "/search?q=example", "/watch?v=abc123", "/pricing"
]

PHISHING_KEYWORDS = [
    "login", "verify", "secure", "update", "confirm", "account",
    "signin", "webscr", "banking", "suspend", "alert", "urgent",
    "billing", "recover", "unlock"
]

TYPO_SUFFIXES = ["-security", "-verify", "-support", "-team", "-online", "-alert"]

SHORTENERS = ["bit.ly", "tinyurl.com", "goo.gl", "t.co", "is.gd", "cutt.ly"]

RANDOM_TLDS = [".tk", ".xyz", ".top", ".ru", ".info", ".click", ".gq", ".cf"]


def random_ip():
    return ".".join(str(random.randint(1, 254)) for _ in range(4))


def typosquat(brand):
    """Generate a lookalike misspelling of a brand name."""
    ops = random.choice(["dup", "swap", "insert", "hyphen", "homoglyph"])
    b = list(brand)
    if ops == "dup" and len(b) > 2:
        i = random.randint(0, len(b) - 1)
        b.insert(i, b[i])
    elif ops == "swap" and len(b) > 3:
        i = random.randint(0, len(b) - 2)
        b[i], b[i + 1] = b[i + 1], b[i]
    elif ops == "insert":
        i = random.randint(0, len(b))
        b.insert(i, random.choice("qxz01"))
    elif ops == "hyphen":
        i = random.randint(1, len(b) - 1)
        b.insert(i, "-")
    elif ops == "homoglyph":
        b = [c.replace("o", "0").replace("l", "1") for c in b]
    return "".join(b)


def make_legit_url():
    brand = random.choice(LEGIT_BRANDS)
    tld = random.choice(LEGIT_TLDS)
    path = random.choice(LEGIT_PATHS)
    use_www = random.random() < 0.5
    sub = "www." if use_www else ""
    return f"https://{sub}{brand}{tld}{path}"


def make_phishing_url():
    style = random.choice(["typosquat", "ip", "keyword_subdomain",
                            "shortener", "random_tld_keywords", "at_symbol"])
    brand = random.choice(LEGIT_BRANDS)
    keyword = random.choice(PHISHING_KEYWORDS)

    if style == "typosquat":
        fake_brand = typosquat(brand)
        suffix = random.choice(TYPO_SUFFIXES)
        return f"http://{fake_brand}{suffix}.com/{keyword}"

    if style == "ip":
        ip = random_ip()
        return f"http://{ip}/{keyword}/{brand}-account"

    if style == "keyword_subdomain":
        tld = random.choice(RANDOM_TLDS)
        return f"http://{keyword}.{brand}-{keyword}{tld}/{keyword}.php"

    if style == "shortener":
        short = random.choice(SHORTENERS)
        rand_code = "".join(random.choices("abcdefghijkmnpqrstuvwxyz23456789", k=7))
        return f"http://{short}/{rand_code}"

    if style == "random_tld_keywords":
        tld = random.choice(RANDOM_TLDS)
        return f"http://{brand}-{keyword}-{random.randint(10,999)}{tld}/{keyword}"

    if style == "at_symbol":
        tld = random.choice(RANDOM_TLDS)
        return f"http://{brand}.com@{keyword}{tld}/{keyword}"

    return f"http://{brand}-{keyword}.tk/{keyword}"


def build_dataset(n_per_class=1500):
    rows = []
    seen = set()

    while sum(1 for r in rows if r[1] == 0) < n_per_class:
        u = make_legit_url()
        if u not in seen:
            seen.add(u)
            rows.append((u, 0))

    while sum(1 for r in rows if r[1] == 1) < n_per_class:
        u = make_phishing_url()
        if u not in seen:
            seen.add(u)
            rows.append((u, 1))

    random.shuffle(rows)
    return rows


def save_dataset(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "label"])
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {path}")


if __name__ == "__main__":
    dataset = build_dataset(n_per_class=1500)
    out_path = os.path.join(os.path.dirname(__file__), "urls_dataset.csv")
    save_dataset(dataset, out_path)

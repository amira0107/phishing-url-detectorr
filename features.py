"""
features.py
------------
Feature extraction module for the Phishing URL Detector.

Given a raw URL string, this module extracts a fixed-size numeric
feature vector that a machine learning model can consume.

The features are inspired by well-known research on phishing
detection (UCI Phishing Websites dataset, PhishTank studies) and
rely only on the URL string itself (no need to fetch the page),
which keeps the pipeline fast and safe (we never actually visit
potentially malicious links).
"""

import re
import math
from urllib.parse import urlparse

SUSPICIOUS_WORDS = [
    "login", "verify", "update", "secure", "account", "banking",
    "confirm", "signin", "password", "bank", "paypal", "ebay",
    "webscr", "suspend", "alert", "urgent", "click", "free",
    "gift", "prize", "bonus"
]

SHORTENER_DOMAINS = [
    "bit.ly", "goo.gl", "tinyurl.com", "t.co", "ow.ly", "is.gd",
    "buff.ly", "adf.ly", "shorte.st", "cutt.ly"
]

FEATURE_NAMES = [
    "url_length",
    "hostname_length",
    "path_length",
    "num_dots",
    "num_hyphens",
    "num_underscores",
    "num_slashes",
    "num_digits",
    "num_special_chars",
    "num_subdomains",
    "has_ip",
    "has_at_symbol",
    "has_https",
    "has_https_token_in_domain",
    "has_double_slash_redirect",
    "num_suspicious_words",
    "is_shortened",
    "domain_entropy",
    "digit_letter_ratio",
]


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    probs = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs)


def _has_ip_address(hostname: str) -> int:
    ip_pattern = re.compile(
        r"^(\d{1,3}\.){3}\d{1,3}$|"
        r"^0x[0-9a-fA-F]{1,2}(\.0x[0-9a-fA-F]{1,2}){3}$"
    )
    return 1 if hostname and ip_pattern.match(hostname) else 0


def extract_features(url: str) -> dict:
    """Extract a dict of numeric features from a single URL string."""
    url = url.strip()
    if not re.match(r"^https?://", url, re.IGNORECASE):
        # normalize so urlparse behaves consistently
        parsed_url_for_parse = "http://" + url
    else:
        parsed_url_for_parse = url

    parsed = urlparse(parsed_url_for_parse)
    hostname = parsed.hostname or ""
    path = parsed.path or ""

    num_subdomains = max(hostname.count(".") - 1, 0) if hostname else 0
    suspicious_hits = sum(1 for w in SUSPICIOUS_WORDS if w in url.lower())
    is_shortened = 1 if any(s in hostname for s in SHORTENER_DOMAINS) else 0

    digits = sum(c.isdigit() for c in url)
    letters = sum(c.isalpha() for c in url)
    digit_letter_ratio = digits / letters if letters > 0 else 0.0

    features = {
        "url_length": len(url),
        "hostname_length": len(hostname),
        "path_length": len(path),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_underscores": url.count("_"),
        "num_slashes": url.count("/"),
        "num_digits": digits,
        "num_special_chars": len(re.findall(r"[^a-zA-Z0-9\.\-/_:]", url)),
        "num_subdomains": num_subdomains,
        "has_ip": _has_ip_address(hostname),
        "has_at_symbol": 1 if "@" in url else 0,
        "has_https": 1 if parsed.scheme == "https" else 0,
        "has_https_token_in_domain": 1 if "https" in hostname.lower() else 0,
        "has_double_slash_redirect": 1 if url.rfind("//") > 7 else 0,
        "num_suspicious_words": suspicious_hits,
        "is_shortened": is_shortened,
        "domain_entropy": round(_shannon_entropy(hostname), 4),
        "digit_letter_ratio": round(digit_letter_ratio, 4),
    }
    return features


def extract_features_vector(url: str) -> list:
    """Return the feature values in the fixed FEATURE_NAMES order."""
    feats = extract_features(url)
    return [feats[name] for name in FEATURE_NAMES]


if __name__ == "__main__":
    test_urls = [
        "https://www.google.com",
        "http://192.168.1.1/login/verify-account",
        "http://secure-paypal-update.com/confirm@login",
        "https://bit.ly/3xyzAbc",
    ]
    for u in test_urls:
        print(u)
        print(extract_features(u))
        print("-" * 40)

# 🛡️ Phishing URL Detector (ML)

A machine learning system that detects phishing URLs using only the
URL string itself — no need to visit the page, which keeps analysis
fast and safe. Built with **scikit-learn** (Random Forest) and served
through a **Flask** REST API with a minimal web UI.

## Why this approach

Instead of relying on blocklists (which are always a step behind new
phishing domains), this project extracts **19 structural features**
from a URL — length, number of hyphens/dots, presence of an IP
address, suspicious keywords ("login", "verify", "secure"...), use of
URL shorteners, domain entropy, etc. — and trains a classifier to
recognize the *patterns* phishing URLs tend to share.

## Project structure

```
phishing-url-detector/
├── app/
│   ├── app.py              # Flask API + web UI
│   └── features.py         # URL -> feature vector extraction
├── data/
│   ├── generate_dataset.py # Builds the labeled training dataset
│   └── urls_dataset.csv    # Generated dataset (3000 URLs)
├── models/
│   ├── train_model.py      # Trains & evaluates the RandomForest model
│   ├── phishing_model.joblib   # Trained model (generated)
│   └── training_report.json    # Metrics from the last training run
├── requirements.txt
└── README.md
```

## How it works

1. **`features.py`** turns any URL into a numeric feature vector
   (URL length, hyphen count, HTTPS usage, suspicious keywords,
   domain entropy, IP address detection, shortener detection, etc.)
2. **`generate_dataset.py`** builds a labeled dataset of ~3000 URLs
   (legitimate look-alikes of real brands + rule-based phishing
   patterns: typosquatting, IP hosts, suspicious subdomains, URL
   shorteners, `@`-symbol tricks...).
   > ⚠️ For production use, replace this with a real dataset such as
   > [PhishTank](https://phishtank.org/developer_info.php) or the
   > [UCI Phishing Websites dataset](https://archive.ics.uci.edu/dataset/327/phishing+websites).
   > The rest of the pipeline (features → model → API) stays identical.
3. **`train_model.py`** trains a `RandomForestClassifier` and reports
   accuracy / precision / recall / F1 / confusion matrix.
4. **`app.py`** loads the trained model and exposes it via a REST API
   and a small web form.

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) regenerate the dataset
python3 data/generate_dataset.py

# 3. Train the model
python3 models/train_model.py

# 4. Run the web app
cd app
python3 app.py
# -> open http://localhost:5000
```

## API usage

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"url": "http://secure-paypal-update.tk/confirm-account"}'
```

Response:
```json
{
  "url": "http://secure-paypal-update.tk/confirm-account",
  "is_phishing": true,
  "phishing_probability": 1.0,
  "features": { "...": "..." }
}
```

## Model performance (on synthetic held-out test set)

| Metric    | Score |
|-----------|-------|
| Accuracy  | 1.00  |
| Precision | 1.00  |
| Recall    | 1.00  |
| F1-score  | 1.00  |

Top predictive features: `has_https`, `num_hyphens`,
`num_suspicious_words`, `hostname_length`, `domain_entropy`.

> Note: the synthetic dataset used here has clearly separable
> patterns by design, so scores are near-perfect. On real-world data
> (PhishTank/UCI), expect ~95-97% accuracy — still worth mentioning
> honestly in an interview if asked.

## Possible extensions (good talking points for interviews)

- Swap the synthetic dataset for a real PhishTank/OpenPhish feed
- Add WHOIS/domain-age features (newly registered domains are riskier)
- Add a browser extension front-end calling the `/predict` endpoint
- Deploy with Docker + CI/CD (GitHub Actions) for a full DevSecOps demo
- Add SHAP explainability to show *why* a URL was flagged

## Tech stack

`Python` · `scikit-learn` · `pandas` · `Flask` · `joblib`

## License

MIT — free to use for learning and portfolio purposes.

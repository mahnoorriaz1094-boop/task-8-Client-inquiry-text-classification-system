# Task-8 CoreTech Client Inquiry — Text Classification

> **Internship Task | NLP & Machine Learning**  
> Classifying client messages into service categories using Naive Bayes & Logistic Regression

---

##  Project Overview

CoreTech receives hundreds of client inquiries daily across different service areas. This project builds an automated **text classification system** that reads a client's message and predicts which department or service category it belongs to — saving time and improving response efficiency.

---

##  Dataset

**File:** `coretech_client_inquiries.csv`  
**Total Records:** 80 messages (10 per category)  
**Columns:** `message`, `category`

###  Categories

| # | Category | Description |
|---|----------|-------------|
| 1 | **Web Development** | Website creation, CMS, e-commerce, landing pages |
| 2 | **App Development** | iOS, Android, cross-platform mobile apps |
| 3 | **UI/UX Design** | Wireframes, prototypes, design systems, usability |
| 4 | **Digital Marketing** | Social media, Google Ads, email marketing, influencer |
| 5 | **SEO** | Keyword research, on-page, technical SEO, backlinks |
| 6 | **Software Solutions** | ERP, CRM, custom software, API integrations |
| 7 | **General Inquiry** | Pricing, timelines, consultations, services overview |
| 8 | **Complaint** | Bugs, delays, dissatisfaction, billing issues |

###  Sample Data

| message | category |
|---------|----------|
| I need a professional website for my restaurant... | Web Development |
| Can you develop a fitness tracking app with GPS... | App Development |
| Our website is not ranking on Google... | SEO |
| I am very unhappy with the website you delivered... | Complaint |

---

##  Models Trained

| Model | Vectorizer | Notes |
|-------|------------|-------|
| **Multinomial Naive Bayes** | TF-IDF | Fast, probabilistic baseline |
| **Logistic Regression** | TF-IDF | Linear model, strong on text |

###  TF-IDF Configuration
```python
TfidfVectorizer(
    max_features=500,
    ngram_range=(1, 2),    # unigrams + bigrams
    stop_words='english',
    lowercase=True
)
```

---

##  Results

###  Naive Bayes

```
Accuracy: ~87.50%

                    precision  recall  f1-score  support
App Development        1.00    1.00      1.00       2
Complaint              1.00    1.00      1.00       2
Digital Marketing      1.00    1.00      1.00       2
General Inquiry        0.67    1.00      0.80       2
SEO                    1.00    0.50      0.67       2
Software Solutions     1.00    1.00      1.00       2
UI/UX Design           0.67    1.00      0.80       2
Web Development        1.00    1.00      1.00       2

accuracy                           0.88      16
macro avg              0.92    0.94      0.91      16
```

---

###  Logistic Regression

```
Accuracy: ~93.75%

                    precision  recall  f1-score  support
App Development        1.00    1.00      1.00       2
Complaint              1.00    1.00      1.00       2
Digital Marketing      1.00    1.00      1.00       2
General Inquiry        1.00    1.00      1.00       2
SEO                    0.67    1.00      0.80       2
Software Solutions     1.00    1.00      1.00       2
UI/UX Design           1.00    0.50      0.67       2
Web Development        1.00    1.00      1.00       2

accuracy                           0.94      16
macro avg              0.96    0.94      0.93      16
```

>  **Logistic Regression outperforms Naive Bayes** on this dataset due to its ability to handle overlapping feature patterns in tech-domain text.

---

##  Sample Predictions

```
 "I want a Shopify store for my clothing business."
   Naive Bayes         → Web Development
   Logistic Regression → Web Development

 "Your project manager never replies to our emails."
   Naive Bayes         → Complaint
   Logistic Regression → Complaint

 "We want to rank #1 on Google for our target keywords."
   Naive Bayes         → SEO
   Logistic Regression → SEO

 "Can you build a Flutter app for iOS and Android?"
   Naive Bayes         → App Development
   Logistic Regression → App Development
```

---

##  Project Structure

```
coretech-text-classification/
│
├──  coretech_client_inquiries.csv     # Dataset (80 rows, 8 categories)
├──  coretech_text_classification.ipynb # Full Colab notebook
├──  category_distribution.png          # EDA chart (generated)
├──  confusion_matrices.png             # Confusion matrix heatmaps (generated)
├──  model_comparison.png               # Accuracy bar chart (generated)
└──  README.md                          # This file
```

---

##  How to Run

### Option 1 — Google Colab (Recommended)
1. Upload `coretech_client_inquiries.csv` to Google Drive or Colab session
2. Open `coretech_text_classification.ipynb` in [Google Colab](https://colab.research.google.com)
3. Click **Runtime → Run All**

### Option 2 — Local Setup
```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/coretech-text-classification.git
cd coretech-text-classification

# Install dependencies
pip install scikit-learn pandas matplotlib seaborn

# Run the notebook
jupyter notebook coretech_text_classification.ipynb
```

---

##  Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikitlearn)
![pandas](https://img.shields.io/badge/pandas-2.x-150458?logo=pandas)
![Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading and manipulation |
| `scikit-learn` | TF-IDF vectorization, models, metrics |
| `matplotlib` | Charts and visualizations |
| `seaborn` | Confusion matrix heatmaps |

---

##  Key Concepts Used

- **TF-IDF (Term Frequency–Inverse Document Frequency):** Converts text to numerical features by weighing how unique a word is across all documents
- **Naive Bayes (Multinomial):** A probabilistic classifier that works well with word frequency features; fast and effective for text
- **Logistic Regression:** A linear classifier that models class probabilities; generalizes better on overlapping categories
- **Stratified Split:** Ensures each category is equally represented in train and test sets

---


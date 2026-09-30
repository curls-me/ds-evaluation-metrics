# Evaluation Metrics Cheat Sheet

Quick reference for the metrics covered in `1_confusion_matrix.ipynb`, `2_classification_metrics.ipynb`, `3_regression_metrics.ipynb`, and `slides/Lecture_slides-evaluation-metrics.html`.

## Rule #1: Fit a Baseline First

A score means nothing until you know what you'd get for free. Always fit the dumbest possible model before trusting any real one.

```python
from sklearn.dummy import DummyClassifier, DummyRegressor

baseline_clf = DummyClassifier(strategy="most_frequent")  # always predicts the majority class
baseline_clf.fit(X_train, y_train)
baseline_clf.score(X_test, y_test)

baseline_reg = DummyRegressor(strategy="mean")  # always predicts the mean
baseline_reg.fit(X_train, y_train)
```

This is the **accuracy paradox**: when one class dominates (e.g. only 21.6% of cases are "positive"), a model that ignores every feature can still score ~78% accuracy. If your real model only beats that by a couple of points, it may not be doing much. `DummyRegressor(strategy="mean")` scores R² = 0.0 by construction — the baseline every regression R² should be compared against.

### Worked example: the accuracy paradox

100 wine bottles: 80 ordinary, 20 good. A "model" predicts **all 100 as ordinary** (i.e. it never predicts the positive class at all — exactly what `DummyClassifier(strategy="most_frequent")` would do here).

|                     | Predicted Ordinary | Predicted Good |
|---------------------|---------------------|-----------------|
| **Actually Ordinary** (80) | TN = 80 | FP = 0 |
| **Actually Good** (20) | FN = 20 | TP = 0 |

sklearn format: `[[80, 0], [20, 0]]`

Every good bottle lands in FN, and precision becomes 0/0 (undefined — sklearn raises a warning and returns 0):

- **Accuracy** = (0 + 80) / 100 = **0.80** — looks fine!
- **Recall** = 0 / (0 + 20) = **0.0** — caught zero good bottles
- **Precision** = 0 / (0 + 0) = **undefined → 0**
- **F1** = **0.0**

80% accuracy from a model that identifies precisely nothing useful — this is why accuracy alone is misleading on imbalanced data, and why you always compare against the baseline before trusting a score.

---

## Confusion Matrix

|                     | Predicted Negative | Predicted Positive |
|---------------------|---------------------|---------------------|
| **Actual Negative** | TN | FP |
| **Actual Positive** | FN | TP |

- **TP** — predicted positive, actually positive (a catch)
- **FP** — predicted positive, actually negative (a false alarm / Type I error)
- **FN** — predicted negative, actually positive (a miss / Type II error)
- **TN** — predicted negative, actually negative (a correct all-clear)

```python
from sklearn.metrics import confusion_matrix

confusion_matrix(y_true, y_pred)
# array([[TN, FP],
#        [FN, TP]])
```

From-scratch version:

```python
import numpy as np

def find_TP(y_true, y_pred):
    return sum((y_true == 1) & (y_pred == 1))

def find_FN(y_true, y_pred):
    return sum((y_true == 1) & (y_pred == 0))

def find_FP(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 1))

def find_TN(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 0))
```

> **Gotcha:** sklearn's rows are the truth, columns are the prediction — `[[TN, FP], [FN, TP]]`. Many textbooks and blog posts draw it transposed. Read it mirrored and precision/recall swap places, silently reversing every conclusion. Sanity-check one cell you can reason about first (e.g. the largest count should be the majority-class-predicted-correctly cell) before trusting the rest.

**Which error matters more depends on the business context** — e.g. fraud detection: a missed fraud (FN) usually costs more than a wrongly flagged transaction (FP); spam filtering: losing a real invoice to the spam folder (FP) is usually worse than one spam email getting through (FN).

---

## Classification Metrics

All formulas below use `TP`, `FP`, `FN`, `TN` from the confusion matrix.

| Metric | Definition | Formula | Question it answers |
|---|---|---|---|
| **Accuracy** | Fraction of all predictions that were correct | `(TP + TN) / (TP + TN + FP + FN)` | Of all observations, how many did the model get right? |
| **Balanced Accuracy** | Average of per-class recall | `(Recall₊ + Recall₋) / 2` | Same as accuracy, but each class counts equally regardless of size |
| **Recall** (Sensitivity / TPR) | Fraction of actual positives correctly identified | `TP / (TP + FN)` | Of all actual positives, how many did the model catch? |
| **Precision** | Fraction of predicted positives that are actually positive | `TP / (TP + FP)` | Of everything flagged positive, how many really were? |
| **F1 Score** | Harmonic mean of precision and recall | `2 * (precision * recall) / (precision + recall)` | Single score balancing precision and recall |
| **FPR** (False Positive Rate) | Fraction of actual negatives incorrectly flagged positive | `FP / (FP + TN)` | Used with TPR to build the ROC curve |

```python
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    recall_score, precision_score, f1_score,
)

accuracy_score(y_true, y_pred)
balanced_accuracy_score(y_true, y_pred)
recall_score(y_true, y_pred)
precision_score(y_true, y_pred)
f1_score(y_true, y_pred)
```

From-scratch versions (reuse `find_TP` etc. above):

```python
def my_accuracy_score(y_true, y_pred):
    TP, FN, FP, TN = find_TP(y_true, y_pred), find_FN(y_true, y_pred), find_FP(y_true, y_pred), find_TN(y_true, y_pred)
    return (TP + TN) / (TP + FN + FP + TN)

def my_recall_score(y_true, y_pred):
    TP, FN = find_TP(y_true, y_pred), find_FN(y_true, y_pred)
    return TP / (TP + FN)

def my_precision_score(y_true, y_pred):
    TP, FP = find_TP(y_true, y_pred), find_FP(y_true, y_pred)
    return TP / (TP + FP)

def my_f1_score(y_true, y_pred):
    precision = my_precision_score(y_true, y_pred)
    recall = my_recall_score(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall)
```

**Accuracy pitfall:** with imbalanced classes, a model that always predicts the majority class can still score high accuracy while catching zero cases of the rare class you actually care about. Check accuracy against the baseline (above), and prefer recall/precision/F1/balanced accuracy when classes are imbalanced or errors have unequal costs.

**Precision vs. recall trade-off:** lowering the classification threshold raises recall (catches more positives) but usually lowers precision (more false positives), and vice versa. There is no threshold where both are maximized at once — that's the trade-off, not a bug. 0.5 is a *default* threshold, not a law; pick it based on which mistake costs more.

```python
proba = model.predict_proba(X_test)[:, 1]        # P(positive class)
y_pred_030 = (proba >= 0.30).astype(int)          # a lower threshold → more predicted positives
```

### ROC Curve & AUC

Unlike the metrics above, ROC/AUC take **predicted probabilities**, not predicted labels — and they evaluate performance **across all thresholds** rather than one fixed threshold.

- **TPR** (True Positive Rate) = Recall = `TP / (TP + FN)`
- **FPR** (False Positive Rate) = `FP / (FP + TN)`
- **ROC curve** plots TPR (y-axis) vs. FPR (x-axis) as the threshold varies
- **AUC** (Area Under the Curve) summarizes the ROC curve in one number: 1.0 = perfect, 0.5 = random guessing. Interpretation: pick one random positive and one random negative case — AUC is the probability the model scores the positive one higher.

> **Common mistake to avoid:** passing `y_pred` (binary 0/1 predictions) instead of `proba` (predicted probabilities). Because the ROC curve needs to sweep across thresholds, it requires probability scores — feeding it binary predictions collapses the curve to a single point and silently gives a misleading result.

```python
from sklearn.metrics import roc_curve, roc_auc_score

proba = model.predict_proba(X_test)[:, 1]   # P(positive class) — NOT model.predict(X_test)
fpr, tpr, thresholds = roc_curve(y_true, proba)
auc = roc_auc_score(y_true, proba)
```

Use AUC when you want to compare models independent of a specific decision threshold (e.g. before a business owner has decided what threshold to use).

> **Watch out on imbalanced data:** FPR divides by the size of the (huge) negative class, so a few hundred false positives barely move it — ROC can look great even when the model isn't finding many of the rare positives. When positives are rare, use the **precision-recall curve** instead.

### Precision-Recall Curve (for imbalanced / rare-positive problems)

Plots **precision** (y-axis) against **recall** (x-axis) as the threshold sweeps. Unlike ROC, precision divides by what was actually flagged positive, so it can't be flattered by a large negative class.

```python
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
```

- The "no skill" baseline here is the **positive rate** (share of all observations that are actually positive) — not 0.5 like the ROC diagonal.
- Prefer this over ROC whenever positives are rare (e.g. fraud, rare disease detection).

---

## Regression Metrics

| Metric | Definition | Formula | Notes |
|---|---|---|---|
| **MAE** (Mean Absolute Error) | Average absolute difference between prediction and actual | `mean(\|y_true - y_pred\|)` | Same units as target; treats every error linearly — a 20-unit miss is exactly twice as bad as a 10-unit miss |
| **MSE** (Mean Squared Error) | Average squared difference | `mean((y_true - y_pred)^2)` | Penalizes large errors much more; units are squared |
| **RMSE** (Root Mean Squared Error) | Square root of MSE | `sqrt(MSE)` | Back in original units; **always ≥ MAE** — a much larger RMSE than MAE signals a few large outlier errors rather than uniform small misses |
| **R²** (R-squared / coefficient of determination) | Proportion of variance explained by the model | `1 - (SS_res / SS_tot)` | 1 = perfect fit, 0 = no better than always predicting the mean (the baseline `DummyRegressor` score), negative = worse than the mean |
| **MAPE** (Mean Absolute Percentage Error) | Average absolute error as a % of the actual value | `mean(\|y_true - y_pred\| / \|y_true\|)` | Unit-free, so it's comparable across different targets/scales — but unstable (blows up) if any actual value is near zero |
| **Median Absolute Error** | Middle absolute error | `median(\|y_true - y_pred\|)` | Ignores outliers entirely; compare to MAE the same way you compare MAE to RMSE — a big gap flags outliers |

Where `SS_res = sum((y_true - y_pred)^2)` and `SS_tot = sum((y_true - mean(y_true))^2)`.

```python
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, root_mean_squared_error,
    r2_score, mean_absolute_percentage_error, median_absolute_error,
)

mean_absolute_error(y_true, y_pred)
mean_squared_error(y_true, y_pred)
root_mean_squared_error(y_true, y_pred)
r2_score(y_true, y_pred)
mean_absolute_percentage_error(y_true, y_pred)
median_absolute_error(y_true, y_pred)
```

From-scratch versions:

```python
import numpy as np

def mae_function(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mse_function(y_true, y_pred, root=False):
    squared_error = np.power(y_true - y_pred, 2)
    return np.sqrt(np.mean(squared_error)) if root else np.mean(squared_error)

def r2_function(y_true, y_pred):
    ssr = np.sum(np.power(y_true - y_pred, 2))
    sst = np.sum(np.power(y_true - np.mean(y_true), 2))
    return 1 - (ssr / sst)
```

**When MAE and RMSE (or MAE and median absolute error) disagree a lot:** don't just report the smaller number — go look at which rows the model is failing on. That gap is diagnostic, not noise.

---

## Recap: One Page

Two kinds of problem, four counts, and every formula that falls out of them. (The "Today" column below is the worked example from the slide deck: predicting wine quality/alcohol on the `wine-quality.csv` dataset — kept here as a concrete anchor, not the only valid values.)

**Classification** — target is a *category*, a prediction is right or wrong.
**Regression** — target is a *number*, a prediction is near or far.

Classification — the confusion matrix, and the four counts every classification metric is built from:

|                     | predicted negative | predicted positive |
|---------------------|---------------------|---------------------|
| **actually negative** | TN — true negative — a correct all-clear | FP — false positive — a false alarm |
| **actually positive** | FN — false negative — a miss | TP — true positive — a catch |

Six ways of dividing those four counts:

| Metric | Formula | What it says | Today |
|---|---|---|---|
| Accuracy | `(TP + TN) / (TP + FP + TN + FN)` | of all predictions, the share that were right | 0.807 |
| Recall | `TP / (TP + FN)` | of the real positives, the share we found | 0.291 |
| Precision | `TP / (TP + FP)` | of everything we flagged, the share that was right | 0.616 |
| F1 | `2 · (Precision · Recall) / (Precision + Recall)` | the harmonic mean, pulled toward the weaker half | 0.395 |
| ROC AUC | area under recall plotted against `FPR = FP / (FP + TN)` | the ranking, judged at every threshold at once | 0.784 |
| Balanced accuracy | `(Recall₊ + Recall₋) / 2` | accuracy with each class weighted equally | 0.620 |

Regression — nothing is right or wrong here, only near or far, so we measure the size of the miss:

| Metric | Formula | What it says | Today |
|---|---|---|---|
| MAE | `(1/n) · Σ \|yᵢ − ŷᵢ\|` | the typical miss, in the target's own units | 0.292 |
| MSE / RMSE | `(1/n) · Σ (yᵢ − ŷᵢ)²` · `RMSE = √MSE` | squares the miss, so a few big errors dominate | 0.382 |
| R² | `1 − (SSres / SStot)` | the variance explained, versus always guessing the mean | 0.904 |

*Symbols: TP true positive · TN true negative · FP false positive · FN false negative · Recall₊ recall on the positive class · Recall₋ recall on the negative class · n number of observations · yᵢ actual value · ŷᵢ predicted value · SSres Σ(yᵢ − ŷᵢ)² · SStot Σ(yᵢ − ȳ)²*

**None of those numbers means anything on its own.** In the deck's example, a `DummyClassifier` that ignores every feature scored 0.784 accuracy against the real model's 0.807, and balanced accuracy only 0.500; `DummyRegressor` scores R² = 0.0 by construction (see [Rule #1](#rule-1-fit-a-baseline-first) above). The decision threshold is a dial, not a law — moving it from 0.50 to 0.25 bought recall 0.291 → 0.702 and sold precision 0.616 → 0.449. **No metric is "best."** The right one is decided by what your model predicts and by the consequences of its mistakes.

---

## Quick Decision Guide

| When... | reach for | because... | example |
|---|---|---|---|
| Balanced classes, all errors equal cost | Accuracy | Simple and interpretable when there's no skew | — |
| Missing a positive is costly (fraud, disease detection, screening) | Recall | Counts what you missed (FN) | **Intensive care.** A model flags patients developing sepsis. A miss can be fatal; a false alarm just costs one more blood test — optimize for catching every real case. |
| A false alarm is costly (spam filter, loan approval) | Precision | Counts what you falsely flagged (FP) | — |
| Need one number balancing both | F1 | Harmonic mean — can't hide a weak side behind a strong one | — |
| Comparing models, threshold not chosen yet | ROC AUC | Scores the ranking across every threshold at once | **Churn.** Two candidate models, and marketing hasn't yet decided how many customers it can afford to target (i.e. hasn't picked a threshold) — AUC lets you compare them anyway. |
| Positives are rare / classes are imbalanced | Precision-recall curve, Balanced accuracy | ROC and plain accuracy get flattered by the large majority class | **Hiring.** A CV screener is 95% accurate, but only 4% of applicants are qualified — "reject everyone" alone would score ~96%, and the cost of a false negative (an applicant who never hears back) is easy to miss if you only look at accuracy.<br>**Card fraud.** A bank auto-blocks transactions it thinks are stolen; only 2 in 1,000 really are fraudulent, and wrongly blocking a legitimate one loses a customer. With such a large negative class, the ROC curve looks good almost regardless — the precision-recall curve is the honest read. |
| Regression, outliers should be punished harder | RMSE | Squaring makes big misses dominate | **House prices.** A handful of mansions cost twenty times the median — RMSE lets those large errors dominate the score. (Worth asking: does the answer change if those mansions are the clients actually paying for the model?) |
| Regression, all errors should count equally | MAE | Linear penalty, easy to interpret in target units | **Delivery times.** The app predicts arrival in minutes, and being 20 minutes out is exactly twice as annoying as being 10 minutes out — a plain average miss (MAE) matches that intuition. |
| Regression, need a scale-free error to compare across targets | MAPE | Percentage-based, but avoid when actuals can be ~0 | — |
| Regression, want "goodness of fit" vs. the mean baseline | R² | 0 = no better than guessing the mean, negative = worse | — |

**Whoever picks the metric is making a decision about people, not statistics** — behind every one of these examples is a real cost (a missed diagnosis, a lost customer, an applicant who never hears back), and the "right" metric depends on who bears that cost.

Always **pick the metric before fitting the model**, based on what a mistake actually costs — not after, based on which number looks best. For the full run-through, see `slides/Lecture_slides-evaluation-metrics.html`; open it in a browser (arrow keys to navigate) — it also has a quiz.

---

## Glossary

- **Classification** — a task whose target is a category, so a prediction is simply right or wrong.
- **Regression** — a task whose target is a number, so a prediction is near or far and the size of the miss is what we measure.
- **Positive class** — the outcome you are trying to detect. Naming it is a modelling choice, and it decides what TP, FP and FN mean.
- **Class imbalance** — one class far outnumbering the other, so a model can score well by ignoring the rare one.
- **Confusion matrix** — the 2×2 table of TP, FP, TN, FN. In scikit-learn the truth runs down the side and the prediction across the top.
- **True positive (TP)** — predicted positive, and it really was positive. A catch.
- **False positive (FP)** — predicted positive, actually negative. A false alarm.
- **True negative (TN)** — predicted negative, and it really was negative. A correct all-clear.
- **False negative (FN)** — predicted negative, actually positive. A miss.
- **Accuracy** — of all predictions, the share that were right.
- **Accuracy paradox** — when one class dominates, accuracy mostly measures the class balance rather than the model.
- **Baseline** — the dumbest defensible model, fitted first, so a score has something to beat.
- **Recall** — of the real positives, the share found. Also called sensitivity, or the true positive rate.
- **Precision** — of everything flagged positive, the share that really was.
- **Decision threshold** — the probability cut-off, 0.50 by default, that turns a predicted score into a predicted class.
- **F1-score** — the harmonic mean of precision and recall.
- **Harmonic mean** — an average pulled toward the smaller value, so one weak half cannot hide behind a strong one.
- **ROC curve** — recall against false positive rate, traced across every threshold.
- **False positive rate (FPR)** — of the real negatives, the share wrongly flagged positive.
- **ROC AUC** — area under the ROC curve. 0.5 is guessing, 1.0 is perfect.
- **Balanced accuracy** — the mean of the per-class recalls, so each class counts equally whatever its size.
- **Precision-recall curve** — precision against recall; the honest curve when positives are rare.
- **Positive rate** — the share of all observations that are positive. It is the no-skill line of a precision-recall curve.
- **MAE** — mean absolute error, the typical miss, in the target's own units.
- **MSE** — mean squared error; squares the miss, so big errors dominate.
- **RMSE** — the root of MSE, back in the target's units. Always at least MAE.
- **R²** — the share of variance explained versus always guessing the mean. 0 is the mean, negative is worse.
- **MAPE** — mean absolute percentage error; unit-free, but unstable when any actual value is near zero.
- **Median absolute error** — the middle miss; ignores outliers entirely.

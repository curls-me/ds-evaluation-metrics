# Evaluation Metrics

A hands-on introduction to the metrics used to judge classification and regression models. You practice the standard scikit-learn tools and build a deeper intuition by implementing some of the metrics yourself.

## Learning Objectives

By the end of this repository, you should be able to:

- Build a confusion matrix and interpret its four components (true/false positives and negatives).
- Calculate and apply the core classification metrics (accuracy, precision, recall, F1-score) using `sklearn.metrics`.
- Read an ROC curve and use the ROC AUC score to compare classifiers.
- Decide which type of error (false positives or false negatives) matters most for a given business context, and choose a metric accordingly.
- Implement common regression metrics (MAE, MSE, R-squared) from scratch and verify them against scikit-learn.

## Learning Path

Work through the notebooks in order:

| File / Folder                                                       | Description                                                                   |
| ------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| [**1 - Confusion Matrix**](1_confusion_matrix.ipynb)             | Build and read a confusion matrix, the foundation for classification metrics. |
| [**2 - Classification Metrics**](2_classification_metrics.ipynb) | Precision, recall, F1-score, accuracy, and the ROC curve.                     |
| [**3 - Regression Metrics**](3_regression_metrics.ipynb)         | MAE, MSE, RMSE, and R-squared for continuous targets.                         |

### Additional Folders and Files

| File / Folder                           | Description                             |
| --------------------------------------- | --------------------------------------- |
| [**Data**](data/)                    | Datasets used across the notebooks.     |
| [**Assets**](assets/)                | Images used in the notebooks.           |
| [**Solutions**](solutions/)          | Reference solutions.                    |
| [**pyproject.toml**](pyproject.toml) | Project configuration and dependencies. |
| [**uv.lock**](uv.lock)               | Dependency lock file.                   |

## Setup

> [!NOTE]
> Throughout these steps, text in angle brackets like `<repo-name>` is a
> **placeholder**. Replace it, including the `< >` brackets, with your own
> value. For example, `cd <repo-name>` becomes `cd ds-evaluation-metrics`.

### 1. Create the Repository from the Template

Click **Use this template** on GitHub.

When creating the repository:

- Set yourself as the **Owner**
- Choose a repository name
- Disable **Include all branches**
- Click **Create repository**

> [!IMPORTANT]
> If you are working in pairs or groups, only **one person** should complete this step.

---

### 2. Add Collaborators (Pairs/Groups Only)

If working with teammates:

1. Open the repository on GitHub
2. Go to **Settings → Collaborators**
3. Add your teammates as collaborators
4. Share the repository link with your team

Teammates should accept the invitation before continuing.

---

### 3. Clone the Repository

Copy the SSH URL from the **Code** button on GitHub, then run:

```bash
git clone <copied-ssh-url>
```

The copied SSH URL will look like `git@github.com:<your-username>/<repo-name>.git`.

---

### 4. Move into the Project Folder and Install Dependencies

This installs all dependencies and creates a virtual environment in `.venv/`.

```bash
cd <repo-name>
uv sync
```

---

### 5. Open the Notebooks

> [!NOTE]
> Make sure you open VS Code from the project root so it automatically detects the environment created by `uv sync`.

Launch VS Code in the project root folder:

```bash
code .
```

Then open a notebook and select the Python environment created by `uv sync` as the kernel.

## References & Further Reading

- [**Scikit-learn: Metrics and scoring**](https://scikit-learn.org/stable/modules/model_evaluation.html): The canonical reference for classification and regression metrics.
- [**Scikit-learn: Confusion matrix example**](https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html): A worked example of building and reading a confusion matrix.
- [**Google ML Crash Course: ROC and AUC**](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc): A clear explanation of the ROC curve and the area under it.

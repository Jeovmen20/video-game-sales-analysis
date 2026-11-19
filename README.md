# Video Game Sales Trends Over the Decades

An exploratory data analysis and machine learning project examining historical video game sales data across platforms, genres, publishers, and global regions.

## Project Overview

This repository contains a Jupyter Notebook that investigates commercial trends in the video game industry over a 30-year span using the `vgsales.csv` dataset (16,598 records). The analysis progresses through exploratory data summaries, unsupervised clustering, and supervised regression modeling to identify the primary drivers of a game's global commercial success.

### Research Questions
* **Primary Question:** How have video game sales trends changed over the last 30 years across platforms and genres?
* **Sub-question 1:** Which genres have grown or declined most in popularity over time?
* **Sub-question 2:** What factors are most associated with a game's commercial success?

---

## Dataset (`vgsales.csv`)

The dataset ranks video games from best-selling to worst-selling and includes the following features:
* `Rank`: Overall sales ranking
* `Name`: Title of the video game
* `Platform`: Console or system the game was released on (e.g., Wii, NES, GB)
* `Year`: Year of release
* `Genre`: Game category (e.g., Sports, Platform, Racing, Role-Playing)
* `Publisher`: Company that published the game
* `NA_Sales`, `EU_Sales`, `JP_Sales`, `Other_Sales`: Regional sales figures (in millions)
* `Global_Sales`: Total worldwide sales (in millions)

*Note: Some records contain missing values (such as release years or region-specific sales for titles not released globally), which are handled via `dropna()` prior to modeling.*

---

## Methodology & Key Findings

1. **Exploratory Data Analysis (Lab 3)**
   * Computed summary statistics across regional and global sales metrics.
   * Identified missing values resulting from region-exclusive releases and evaluated their impact on downstream analysis.

2. **Unsupervised Clustering (Lab 4)**
   * Applied K-Means clustering to the numerical sales features.
   * **Finding:** K-Means clustering did not yield meaningful groupings for this dataset, as answering the core research questions relies heavily on categorical variables (`Platform`, `Genre`, `Publisher`) alongside numerical metrics.

3. **Feature Engineering & Linear Regression (Lab 5)**
   * Applied one-hot encoding (`pd.get_dummies`) to convert categorical features (`Platform`, `Genre`, `Publisher`) into numerical representations, expanding the feature space to 623 columns.
   * Trained a `LinearRegression` model (80/20 train-test split) to predict `Global_Sales`.
   * **Evaluation:** Achieved an $R^2$ score of `1.0000` and an RMSE of `0.0054`. Because `Global_Sales` is the direct sum of the regional sales columns (`NA_Sales`, `EU_Sales`, `JP_Sales`, `Other_Sales`), those features dominated the model coefficients ($\approx 1.0$), masking the true predictive weight of `Genre` and `Publisher` and indicating severe overfitting/data leakage.

---

## Dependencies & Setup

To run the notebook locally, ensure you have Python 3 installed along with the following libraries:

```bash
pip install pandas numpy scikit-learn seaborn matplotlib eli5 jupyter

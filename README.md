# Rate My Professors Data Analysis

## Overview
This project analyzes Rate My Professors (RMP) data to explore trends in professor ratings, gender differences, difficulty perception, and potential biases using statistical and machine learning techniques. The dataset includes numerical attributes (e.g., average ratings, number of ratings) and categorical attributes (e.g., university, state, professor tags).

## Data Sources
The project utilizes three datasets:
- `rmpCapstoneNum.csv`: Contains numerical features such as average rating and number of ratings.
- `rmpCapstoneQual.csv`: Contains qualitative features like university and state.
- `rmpCapstoneTags.csv`: Contains binary indicator variables representing different professor tags (e.g., "tough_grader", "inspirational").

## Research Questions
1. **Gender Differences in Ratings:** Do male and non-male professors receive significantly different ratings?
2. **Distribution & Spread Analysis:** How do cumulative distribution functions (CDFs) and bootstrap variance analysis reflect gender-based rating differences?
3. **Effect Size Analysis:** What is the Cohen's d effect size for mean differences in ratings between male and non-male professors?
4. **Tag Differences Across Genders:** Which professor tags show significant gender-based differences?
5. **Gender Differences in Difficulty Ratings:** Do students perceive male and non-male professors as having different difficulty levels?
6. **Effect Size for Difficulty Ratings:** What is the Cohen’s d effect size for difficulty ratings between male and non-male professors?
7. **Regression on Average Ratings:** Can we predict average ratings using numerical features?
8. **Feature Importance in Rating Prediction:** Which features are most important in predicting ratings?
9. **Prediction of Difficulty Ratings:** Can we predict difficulty ratings based on professor tags?
10. **Classification Model for "Received a Pepper" Feature:** Can we classify whether a professor received a chili pepper (indicating attractiveness) using available features?

## Methodology
### Data Processing
- Missing values are handled using support vector regression (SVR) imputation.
- Features are standardized using `StandardScaler`.
- High multicollinearity features are removed using Variance Inflation Factor (VIF).

### Statistical Analysis
- **Kolmogorov-Smirnov (KS) Test**: To check significant differences between gender groups.
- **Mann-Whitney U Test**: To compare distributions of difficulty ratings.
- **Bootstrapping**: Used for variance estimation and effect size confidence intervals.
- **Cohen’s d**: Measures effect size for rating differences.

### Machine Learning Models
- **Linear Regression**: To predict average ratings.
- **Random Forest Regressor**: To improve accuracy and feature importance interpretation.
- **Support Vector Classifier (SVC)**: To classify whether a professor received a chili pepper.
- **Cross-validation (K-Fold)**: Applied for model performance evaluation.

### Visualizations
- **KDE Plots**: To visualize distributions of ratings and difficulty.
- **CDFs**: To analyze cumulative probability differences between groups.
- **Heatmaps**: To show correlation matrices of features.
- **Feature Importance Bar Charts**: For linear regression and random forest models.

## Results
- There are significant gender-based differences in professor ratings and difficulty perception.
- Some professor tags are more gender-associated than others.
- Feature importance analysis reveals key predictors for ratings and difficulty scores.
- Classification models can predict whether a professor received a chili pepper with reasonable accuracy.

## Repository Structure
```
|-- data/                         # Contains RMP datasets                   
|-- scripts/                       # Python scripts for data preprocessing and modeling
|-- README.md                      # Project documentation
|-- requirements.txt                # Python dependencies
```

## Setup & Usage
### Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/rmp-analysis.git
   cd rmp-analysis
   ```
2. Create a virtual environment (optional but recommended):
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use 'venv\Scripts\activate'
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Run analysis scripts:
   ```sh
   python scripts/data_analysis.py
   ```

## Dependencies
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scipy`
- `sklearn`
- `statsmodels`



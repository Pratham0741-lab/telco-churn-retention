# Telco churn: choosing who gets a retention offer by expected value

Predicts which telecom customers will leave, then picks the decision threshold by **expected business value** instead of accuracy. Includes a calibration check, a sensitivity analysis of the cost assumptions, SHAP explanations, an error analysis, a fairness check, survival analysis, a model card and a what-if app. The test set is scored once, at the end.

## Problem

A retention offer costs money for everyone who receives it, but it only pays off for customers who would otherwise have left and are persuaded to stay. Choosing who gets the offer is a cost decision, so the threshold should maximise expected value, not F1 or accuracy.

## Data source and license

- **IBM Telco Customer Churn** (sample dataset), via Kaggle: <https://www.kaggle.com/datasets/blastchar/telco-customer-churn>
- 7,043 customers, 21 columns (services, contract, billing, demographics, `Churn`). 26.5% churned.
- **License: not clearly stated.** Kaggle's metadata lists "Data files © Original Authors". It is an IBM sample published for learning, so this repo uses it for non-commercial education only. Raw data is not committed.

## What's in the project

| File | What it is |
| --- | --- |
| `analysis.ipynb` | All the analysis, from cleaning to the model card. Settings and cost assumptions are in `CONFIG` at the top |
| `app.py` | What-if Streamlit app: edit a customer and watch the churn score and its drivers |
| `requirements.txt` | Pinned dependencies (pandas 2.3.3, because lifelines needs pandas < 3) |

The notebook is saved with its outputs (tables and charts), so you can read it on GitHub without running it:

1. **Clean, split and explore**: `TotalCharges` blanks are tenure-0 customers, set to 0; stratified 60/20/20 split before any preprocessing; EDA on the training split only.
2. **Model comparison**: logistic regression, random forest and gradient boosting as `Pipeline`s, tuned with 5-fold CV (PR-AUC), each with a calibrated twin.
3. **Threshold by expected value** (offer $100, customer value $1,000, 40% success, all assumed), a 144-cell sensitivity grid, and per-customer values.
4. **Test evaluation, run once**, and a fairness check by senior citizen, gender and contract with bootstrap intervals.
5. **SHAP** on validation and an error analysis of missed churners.
6. **Survival analysis** (Kaplan-Meier, log-rank, Cox model stratified on contract).
7. **Model card** generated from the saved results.

## Results

**Model comparison, validation, threshold 0.5**

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC | Brier |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic regression (unweighted) | 0.661 | 0.537 | 0.593 | 0.643 | 0.837 | 0.138 |
| Logistic regression (balanced) | 0.512 | 0.786 | 0.620 | 0.643 | 0.837 | 0.168 |
| Random forest (balanced) | 0.513 | 0.786 | 0.621 | 0.642 | 0.838 | 0.164 |
| Gradient boosting (balanced) | 0.503 | 0.783 | 0.613 | 0.642 | 0.838 | 0.165 |
| Gradient boosting, calibrated | 0.684 | 0.481 | 0.565 | 0.651 | 0.841 | 0.137 |

- **All models are effectively tied** on ranking quality (PR-AUC 0.641-0.651). Tree models don't beat logistic regression on this data.
- Accuracy is not used to choose: predicting "no churn" for everyone already scores 0.735.

**Threshold by expected value**: the class-weighted random forest at **threshold 0.45** had the highest validation EV ($61,900). But a 1,000-sample bootstrap shows every other model's EV is within noise: every 95% interval for the gap includes 0. The random forest also had the best F1, so here the cost-based choice and the F1 choice agree.

**Test set, run once**

| | Value |
| --- | --- |
| Precision / recall / F1 at 0.45 | 0.502 / 0.826 / 0.625 |
| PR-AUC / ROC-AUC | 0.647 / 0.845 |
| Customers targeted | 615 of 1,409 (TP 309, FP 306, FN 65) |
| Estimated EV at 0.45 | **$62,100** (55.3% of a perfect model's $112,200) |
| EV at default 0.50 / target everyone | $62,300 / $8,700 |

**Drivers and misses**: month-to-month contract dominates the SHAP ranking, followed by no online security, tenure and fiber optic. 41 of the 61 churners missed on validation had one- or two-year contracts; missed churners have a median tenure of 49 months vs 8 for those caught.

**Per-customer value**: instead of a flat $1,000, a lost customer is worth their own monthly bill x 15 months (validation mean $981, range $280-$1,754). Targeting by *expected value at risk* (p x value >= $430) beats targeting by probability alone (p >= 0.45):

| Rule (chosen on validation) | Validation value | Test value | Customers targeted (test) |
| --- | ---: | ---: | ---: |
| Probability: p >= 0.45 | $81,831 | $76,180 | 615 |
| Value: p x value >= $430 | **$86,896** | **$79,837** | 605 |

On validation the gap held in 99% of 1,000 bootstrap resamples (95% interval $734 to $10,238) and at 6, 15 and 30 months of value. The value rule targets slightly fewer customers with higher bills (median $88.42 vs $81.05 per month). On test it captured 63.4% of a perfect model's $125,889, against 60.5% for the probability rule.

**Sensitivity: offer cost x customer value**:

Across 144 cost/value combinations the best threshold ranges from 0.02 to 0.92, and in 16 cells no threshold makes money. In 15 of those a saved customer is worth no more than the offer; in 1 the margin is positive but too thin. The optimal threshold tracks the break-even probability closely for the calibrated model (mean gap -0.00) but sits +0.18 above it for the class-weighted model, so a textbook break-even rule only works on calibrated scores.

**Fairness on test**:

| Group | Miss rate (FNR) of churners | Gap (95% interval) |
| --- | --- | --- |
| Gender | Female 18.7%, Male 16.0% | +2.6 pts (-5.5 to +9.7): no clear gap |
| Senior citizen | No 21.0%, Yes 7.1% | +13.9 pts (+6.4 to +20.4) |
| Contract | Month-to-month 8.2%, One year 80.6%, Two year 100% | the model's main blind spot |

Seniors are flagged more often (68.5% vs 39.0%), in line with their higher churn rate (44.1% vs 23.3%). The gap that matters runs the other way: non-senior churners are missed three times as often. Options are discussed, not applied.

**Survival analysis**:

After 12 months, 70.1% of month-to-month customers remain vs 99.3% on one-year and 100% on two-year contracts. The median month-to-month tenure is 35 months. The one-year contract term violates proportional hazards (p < 0.001), so the Cox model is stratified by contract. Fiber optic has a hazard ratio of 2.14 vs DSL, electronic check 1.74 and mailed check 1.92 vs bank transfer (concordance 0.710). Five covariates still fail the PH test, and the report says so.

**What-if app** (`streamlit run app.py`): pick a validation customer, change contract, charges, tenure and services, and watch the score, the decision and the SHAP drivers update.

**Churn is not uplift.** The model predicts who will leave, not who an offer will save. Measuring that needs a randomised test: offer the discount to a random half of the at-risk group and compare how many stay (about 1,085 customers per arm to detect a 50% → 44% drop). This dataset has no treatment data, so no uplift model is built.

The model card is Part 7 of the notebook.

## What didn't work

- **The EV-chosen threshold didn't beat the default on test.** 0.45 earned $62,100 against $62,300 at 0.50. The EV curve is flat near the top, so the threshold choice matters less than the cost assumptions.
- **Tree models didn't beat logistic regression.** Random forest and gradient boosting tied with it (validation PR-AUC 0.641-0.651), so the extra complexity bought nothing here.
- **The textbook break-even threshold.** The theory says target when p > 0.25, but the chosen class-weighted model's best thresholds sit about 0.18 above break-even across the sensitivity grid; only the calibrated model follows the rule.
- **The first Cox model.** Contract violated proportional hazards (one-year term, p < 0.001), and the "No internet service" dummies were exact duplicates, which gave three identical meaningless hazard ratios. The final model drops the duplicates and stratifies on contract; five covariates still fail the PH test.
- **Dependency clash.** lifelines 0.30.3 doesn't support pandas 3, so the repo is pinned to pandas 2.3.3. Rerunning on 2.3.3 gave identical results.

## Limitations

- **Fairness numbers are noisy:** 222 seniors and 9 two-year churners in the test set. The gaps are real enough to monitor, not precise.
- **Survival analysis uses one snapshot,** so tenure mixes customers who joined at different times. Five Cox covariates still violate proportional hazards after stratifying on contract; their hazard ratios are averages over tenure.
- **pandas is pinned to 2.3.3** (not 3.x) because lifelines 0.30.3 requires pandas < 3. Rerunning on 2.3.3 gave identical results and plots.

- **Cost assumptions drive every dollar figure.** They are illustrative, not measured. The EV numbers are offline estimates, not savings.
- **The EV-chosen threshold gave no gain on test** over the default 0.5 ($62,100 vs $62,300). The EV curve is flat near the optimum, so the method's value here is in knowing the curve's shape and when *not* to target, not in a precise threshold.
- **Model differences are noise-level** on a 1,409-row validation set. Picking the top EV model is a fixed rule, not evidence it is truly best. Logistic regression would be a reasonable production choice.
- **SHAP explains the model, not causes.** It shows what the model uses, not what would change a customer's decision.
- **Blind spot:** churners on longer contracts are mostly missed (all 13 two-year-contract churners on validation).
- **No time dimension** in the data, so there's no out-of-time test; real deployment would need one.
- **Per-customer value is still a rough proxy**: monthly bill x 15 months ignores margins and how long each customer would really stay.
- **SHAP is computed on the uncalibrated random forest** (the chosen model). Calibration would rescale scores but not change which features drive them.

## How to run

Needs Python 3.11.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Get the data (one CSV) into `data/`:

```bash
curl -L -o data/telco.zip https://www.kaggle.com/api/v1/datasets/download/blastchar/telco-customer-churn
cd data && unzip telco.zip && rm telco.zip && cd ..
```

To try other business assumptions, edit `economics` in `CONFIG` at the top of the notebook and run it again.

Run the analysis (or open `analysis.ipynb` and run all cells):

```bash
jupyter nbconvert --to notebook --execute --inplace analysis.ipynb
```

Then start the app:

```bash
streamlit run app.py
```

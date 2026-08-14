# Predicting Airbnb Listing Prices in Brisbane, Australia

A regression project that predicts the nightly price (AUD) of Airbnb listings in Brisbane from listing-level characteristics, run as a Kaggle-style prediction competition for a university Business Analytics course.

**About this project.** This was a group assignment (team of five). It's shared here as a portfolio piece to show the full end-to-end workflow, with my individual contribution concentrated in **Task 3: Model Fitting, Tuning, and Prediction** — selecting the candidate models, designing the cross-validation strategy, tuning ElasticNet, Random Forest, Gradient Boosting, Histogram Gradient Boosting and Extra Trees, building the voting ensemble, and producing the final Kaggle submission plus residual/error analysis. Tasks 1 and 2 (EDA and feature engineering) reflect collaborative team work and are retained to give full context for the modelling decisions in Task 3.

---

## Summary

- **Problem:** predict nightly listing price from 64 raw Airbnb fields (property, host, review, and booking attributes), using 3,735 training listings and 1,601 test listings.
- **Approach:** cleaned and engineered the raw fields down to 27 features (capacity, location, host, review, amenity, and text-derived signals), then compared a regularised linear baseline against four tree-based ensembles under a shared 10-fold cross-validation setup, scored on Mean Absolute Percentage Error (MAPE) — the same metric used by the competition leaderboard.
- **Result:** a tuned Random Forest was the best model (CV MAPE 5.82%), scoring **≈6.9% MAPE** on the final Kaggle submission. A top-3 voting ensemble was also tested but slightly underperformed the single Random Forest (6.12% CV MAPE), because its component models had errors too correlated to benefit from averaging.

## Tools

- **Language / core libraries:** Python, pandas, NumPy
- **Modelling:** scikit-learn — `ElasticNet`, `RandomForestRegressor`, `GradientBoostingRegressor`, `HistGradientBoostingRegressor`, `ExtraTreesRegressor`, `VotingRegressor`, `Pipeline`, `RobustScaler`, `KFold`, `GridSearchCV`
- **Visualisation:** matplotlib, seaborn, Plotly (interactive geographic maps)
- **Other:** `KMeans` for location clustering, `re`/`html` for text cleaning

## Visuals

The notebook includes:
- Price distribution (raw vs. log-transformed) and skewness diagnostics
- An interactive map of listing density and price tiers across Brisbane
- A feature-correlation heatmap and amenity price-lift chart (justifying which amenities became features)
- Hyperparameter tuning curves for each model
- Random Forest feature-importance and residual-diagnostic plots
- A final model-comparison chart (CV MAPE across all five models plus the voting ensemble)

## Key Insights

- **Random Forest was the clear winner** (5.82% CV MAPE) over ElasticNet (22.37%), Gradient Boosting (7.27%), Histogram Gradient Boosting (6.92%), and Extra Trees (6.86%) — evidence that price is driven by non-linear interactions (e.g. amenities matter more for entire homes than private rooms) that a linear model structurally can't capture.
- **Ensembling didn't help:** all tree-based models leaned on similar signal and produced correlated errors, so a top-3 voting ensemble (6.12%) landed between the best and weaker members rather than beating the leader — a useful negative result on when averaging pays off.
- **The real bottleneck was data, not modelling:** two engineered features — `implied_price` (derived from revenue ÷ occupancy) and `room_type` — accounted for ~74% of the Random Forest's decisions, and every model tried (including XGBoost, LightGBM, and CatBoost outside the five compared here) converged to the same ~6–7% error band. That convergence points to a ceiling set by the available features rather than the algorithm choice.
- **Business takeaway:** despite the ceiling, the model is accurate enough to support data-driven pricing — helping hosts benchmark listings against comparable properties and identify under- or over-priced ones.
- **Future work** identified in the notebook: time-based features (seasonality, day-of-week, booking lead time), features that don't just re-encode price (e.g. sentiment from reviews, themes from listing descriptions), and estimating a realistic "noise floor" from near-identical listings before investing further in modelling.

---

*Notebook: `Airbnb_PricePrediction_Portfolio.ipynb`*

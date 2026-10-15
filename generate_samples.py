"""Recreate the included synthetic datasets. No external downloads required."""

from pathlib import Path
import numpy as np
import pandas as pd


def generate_samples():
    output = Path(__file__).resolve().parent / "sample_data"
    output.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)  # A seed makes the examples repeatable.

    # Classification: predict a category (customer churn).
    count = 240
    months = rng.integers(1, 72, count)
    charges = rng.uniform(20, 120, count).round(2)
    tickets = rng.integers(0, 7, count)
    contract = rng.choice(["monthly", "annual"], count)
    score = -1.3 + 0.018 * charges + 0.35 * tickets - 0.045 * months + 0.6 * (contract == "monthly")
    churn = rng.random(count) < 1 / (1 + np.exp(-score))
    customers = pd.DataFrame({
        "tenure_months": months, "monthly_charge": charges,
        "support_tickets": tickets, "contract": contract,
        "churn": np.where(churn, "yes", "no"),
    })
    customers.loc[rng.choice(count, 12, replace=False), "monthly_charge"] = np.nan
    customers.loc[rng.choice(count, 6, replace=False), "contract"] = np.nan
    customers = pd.concat([customers, customers.iloc[:4]], ignore_index=True)
    customers.to_csv(output / "classification.csv", index=False)

    # Regression: predict a number (fictional house price, in currency units).
    count = 250
    area = rng.integers(500, 3000, count)
    bedrooms = rng.integers(1, 6, count)
    age = rng.integers(0, 60, count)
    neighborhood = rng.choice(["central", "suburban", "rural"], count)
    premium = np.where(neighborhood == "central", 90000, np.where(neighborhood == "suburban", 35000, 0))
    price = 40000 + area * 160 + bedrooms * 12000 - age * 1200 + premium + rng.normal(0, 25000, count)
    houses = pd.DataFrame({"area_sqft": area, "bedrooms": bedrooms, "age_years": age,
                           "neighborhood": neighborhood, "price": price.round(2)})
    houses["area_sqft"] = houses["area_sqft"].astype(float)
    houses.loc[rng.choice(count, 10, replace=False), "area_sqft"] = np.nan
    houses.to_csv(output / "regression.csv", index=False)

    # Clustering: three groups created around different centers; no target.
    groups = []
    for income, spend, visits in [(30, 25, 4), (70, 75, 12), (110, 40, 7)]:
        groups.append(pd.DataFrame({
            "annual_income_thousands": rng.normal(income, 7, 80).round(2),
            "spending_score": np.clip(rng.normal(spend, 8, 80), 1, 100).round(2),
            "monthly_visits": np.maximum(0, rng.normal(visits, 1.4, 80)).round(1),
        }))
    shoppers = pd.concat(groups, ignore_index=True).sample(frac=1, random_state=42)
    shoppers.to_csv(output / "clustering.csv", index=False)


if __name__ == "__main__":
    generate_samples()
    print("Created three synthetic CSV files in sample_data/.")

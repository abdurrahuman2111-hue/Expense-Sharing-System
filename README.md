# Uber Ride Analysis + Demand Forecasting (using `uberrproject.ipynb`)

This project analyzes the **Uber Drive-style** dataset and builds a **predictive model** for ride demand forecasting.

> Source for analysis/plots: `uberrproject.ipynb` (reads `uberdrive.csv`, performs preprocessing, generates EDA plots, and derives peak windows).

---

## 1) Dataset
The notebook expects a CSV similar to the common Uber Drive dataset with columns such as:
- `START_DATE*` (start timestamp)
- `END_DATE*` (end timestamp) — used to compute ride duration
- `MILES*` (distance traveled)
- `PURPOSE*`
- `CATEGORY*`
- `START*`, `STOP*`

The notebook loads:
- `uber_df = pd.read_csv(r"C:\\Users\\abudr\\uberdrive.csv")`

---

## 2) Data Preprocessing (performed in the notebook)
Key preprocessing steps from `uberrproject.ipynb`:

1. **Initial cleanup**
- Drops the last row: `uber_df.drop(uber_df.index[-1], inplace=True)`
- Checks duplicates and removes them: `uber_df.drop_duplicates(inplace=True)`
- Basic missing value handling:
  - Fills missing `PURPOSE*` with `'not'`

2. **Datetime parsing + feature extraction**
- Converts timestamps:
  - `uber_df["START_DATE*"] = pd.to_datetime(uber_df["START_DATE*"], errors='coerce')`
  - `uber_df["END_DATE*"] = pd.to_datetime(uber_df["END_DATE*"], errors='coerce')`
- Creates:
  - `date` from `START_DATE*`
  - `time` (hour) from `START_DATE*`
  - `day-night` using hour bins:
    - Morning (0–10)
    - Afternoon (10–15)
    - Evening (15–19)
    - Night (19–24)

3. **Columns normalization (later in notebook)**
- Renames columns by stripping `*`, removing spaces, lowercasing:
  - `uber_df.columns = uber_df.columns.str.replace('*', '', regex=False).str.replace(' ', '').str.lower().str.strip()`

4. **Additional engineered variables**
- `month` from `start_date`
- `ride_duration` (in minutes):
  - `ride_duration = (end_date - start_date).dt.total_seconds() / 60`
- `weekday` from `start_date`

5. **Handling missing ride duration**
- Drops rows missing required fields for some computations:
  - `valid_uber_df = uber_df.dropna(subset=['purpose','ride_duration'])`
- Fills missing durations with median:
  - `uber_df['ride_duration'].fillna(uber_df['ride_duration'].median(), inplace=True)`

---

## 3) EDA Visualizations (present in `uberrproject.ipynb`)
The notebook generates many plots. The most relevant for the assignment are:

### A) Ride patterns
- `countplot` for:
  - Top start locations (`START*`)
  - `CATEGORY*` and `PURPOSE*`
- Distribution comparisons:
  - Boxplots / violin plots of `MILES*` by `day-night`
  - Faceted boxplots by `CATEGORY*`

### B) Peak hours
- Time of day distribution (hour-level):
  - `sorted_time_counts = uber_df['time'].value_counts().sort_values(ascending=False)`
  - `sns.countplot(... x='time', order=sorted_time_counts.index)`
- Day-part distribution:
  - `sns.countplot(uber_df['day-night'], ...)`

### C) Fare trends
- The notebook does not explicitly show a `fare` column plot; instead it uses:
  - `MILES*` trends
  - `ride_duration` trends (via `MINUTES` / `ride_duration` computations)
- Example trend plot:
  - `sns.lineplot(... x=MINUTES in minutes, y=MILES*)`

### D) Demand forecasting (predictive model)
- The notebook derives demand-related signals from:
  - `MILES*` patterns across time/category/purpose
  - aggregated counts by month/purpose/weekday
  - computed duration and miles by day-night

---

## 4) Predictive Modeling for Ride Demand Forecasting
Although the provided notebook code is primarily an EDA notebook, demand forecasting can be framed using the available numeric measures.

Common demand proxy used:
- **`MILES*`**: treated as a strong proxy for demand intensity in ride-level data.

A predictive pipeline typically:
1. Uses engineered time features (hour, day-night, weekday, month)
2. Uses categorical trip context (purpose, category, start/stop)
3. Trains a regression model to predict the demand proxy (miles)
4. Converts predictions into peak-hour forecasts (top time windows)

In this repo, the predictive modeling code is provided separately in:
- **`uber_predict.py`**

---

## 5) Key Insights to Extract (from outputs of `uberrproject.ipynb`)
The notebook computes/prints/plots multiple summaries. Use them to write insights such as:

- **Most common areas (start locations)** based on `START*` frequency.
- **Top categories/purposes** based on `CATEGORY*` / `PURPOSE*` counts.
- **Peak demand periods**:
  - highest `countplot` hours
  - strongest `day-night` segments
- **Miles/duration patterns**:
  - `MILES*` vs `day-night`
  - miles by category and purpose
  - ride duration statistics by purpose

---

## 6) How to Run
1. Ensure required libraries are installed:
```bash
pip install pandas numpy seaborn matplotlib scikit-learn joblib
```

2. Update dataset path in the notebook (`r"C:\\Users\\abudr\\uberdrive.csv"`) if needed.

3. Run `uberrproject.ipynb`.

4. (Optional) Run predictive model script in this repo:
```bash
python uber_predict.py
```

---

## Notes / Limitations
- The notebook uses **distance (`MILES*`)** and **duration** as proxies for “demand” because the dataset commonly lacks a direct “ride count” target per future window.
- The EDA is strong for extracting peak patterns, but forecasting requires modeling and careful definition of the prediction horizon.

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import io
from datetime import datetime


ANALYTICAL_BASELINE = pd.Timestamp("2020-01-01")

METRIC_NAME_HINTS = [
    "revenue", "sales", "amount", "total", "price", "cost", "premium", "value",
    "spend", "expense", "income", "profit", "quantity", "qty", "units", "volume",
    "balance", "payment", "charge", "fee", "gwp"
]

TIMELINE_NAME_HINTS = [
    "date", "time", "effective", "created", "order", "transaction", "posted",
    "period", "timestamp", "inception", "renewal"
]

ENTITY_NAME_HINTS = [
    "id", "code", "key", "customer", "client", "account", "member", "employee",
    "order", "ref", "number", "no", "sku", "user", "patient", "student", "policy"
]

# Columns matching these are usually surrogate keys, not something worth treating as a
# grouping dimension even if their cardinality happens to look categorical.
ID_LIKE_EXCLUDE_HINTS = ["id", "code", "key", "no", "number", "ref", "sku", "uuid", "guid"]

MAX_DIMENSION_CANDIDATES = 15
DEFAULT_DIMENSION_COUNT = 5


class BookOfBusinessAnalyzer:
    """
    Loads a single uploaded tabular file (CSV/XLSX/XLSM) and provides generic schema
    inference, filtering, KPI computation, seasonality-aware forecasting, and multi-goal
    pacing analysis — works on any file with a numeric metric and a date column, not just
    a specific domain's data.

    One instance corresponds to one uploaded file for the lifetime of a session; `self.df`
    holds the cleaned raw data and all analysis methods operate on filtered copies of it.
    """

    def __init__(self, file_bytes: bytes, file_name: str):
        self.file_name = file_name
        name_lower = (file_name or "").lower()

        if name_lower.endswith(".csv"):
            self.df = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
        elif name_lower.endswith((".xls", ".xlsx", ".xlsm")):
            self.df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            # Unknown extension — try CSV first, then Excel, rather than giving up outright.
            try:
                self.df = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
            except Exception:
                self.df = pd.read_excel(io.BytesIO(file_bytes))

        self.df.columns = [str(c).strip() for c in self.df.columns]

        for col in self.df.select_dtypes(include=["object"]).columns:
            self.df[col] = self.df[col].astype(str).str.strip()

    # ----------------------------------------------------------------------- #
    # Generic value/column helpers
    # ----------------------------------------------------------------------- #

    def _normalize_categorical_value(self, x):
        try:
            if isinstance(x, (float, int)) and not pd.isna(x) and x == int(x):
                return str(int(x))
        except (ValueError, TypeError):
            pass

        return str(x).strip()

    def get_unique_column_values(self, col: str, limit: int = 500) -> list:
        if not col or col not in self.df.columns:
            return []

        raw = self.df[col].dropna()
        cleaned = [self._normalize_categorical_value(x) for x in raw]
        cleaned = [v for v in cleaned if v and v.lower() != "nan"]

        return sorted(set(cleaned))[:limit]

    def get_date_range(self, time_col: str) -> dict:
        if not time_col or time_col not in self.df.columns:
            return {"min_date": None, "max_date": None}

        parsed = pd.to_datetime(self.df[time_col], errors="coerce").dropna()
        parsed = parsed[parsed >= ANALYTICAL_BASELINE]

        if parsed.empty:
            return {"min_date": None, "max_date": None}

        return {
            "min_date": parsed.min().strftime("%Y-%m-%d"),
            "max_date": parsed.max().strftime("%Y-%m-%d")
        }

    # ----------------------------------------------------------------------- #
    # Schema inference — generic column-role detection
    # ----------------------------------------------------------------------- #

    def _column_numeric_fraction(self, col: str) -> float:
        series = self.df[col]
        non_null = series.dropna()

        if len(non_null) == 0:
            return 0.0

        coerced = pd.to_numeric(non_null, errors="coerce")
        return float(coerced.notna().mean())

    def _column_date_fraction(self, col: str) -> float:
        series = self.df[col]
        non_null = series.dropna()

        if len(non_null) == 0:
            return 0.0

        if pd.api.types.is_datetime64_any_dtype(series):
            return 1.0

        coerced = pd.to_datetime(non_null, errors="coerce")
        return float(coerced.notna().mean())

    def _detect_column_roles(self) -> dict:
        """
        Classify every column into numeric / date / categorical candidate buckets, purely
        from the data itself (type-coercion success rate + cardinality), independent of any
        domain-specific naming assumptions. Column-name hints are only used to *break ties*
        among otherwise-equal candidates, not to decide the bucket itself.
        """
        columns = self.df.columns.tolist()
        n_rows = max(1, len(self.df))

        numeric_cols = []
        date_cols = []
        id_like_cols = []
        categorical_candidates = []

        for col in columns:
            series = self.df[col]
            non_null = series.dropna()

            if len(non_null) == 0:
                continue

            unique_ratio = non_null.nunique() / len(non_null)
            col_lower = str(col).lower()
            looks_id_like_name = any(kw in col_lower for kw in ID_LIKE_EXCLUDE_HINTS)

            is_datetime_dtype = pd.api.types.is_datetime64_any_dtype(series)
            is_numeric_dtype = pd.api.types.is_numeric_dtype(series)

            if is_datetime_dtype:
                date_cols.append(col)
                continue

            if is_numeric_dtype:
                # Numeric dtype columns are never date candidates here: pd.to_datetime on a
                # numeric Series reinterprets the numbers as epoch timestamps and "succeeds"
                # almost universally, which would otherwise misclassify every numeric metric
                # column (revenue, units, etc.) as a date.
                #
                # Uniqueness ratio alone is NOT a reliable signal for "this is a surrogate
                # key" — a continuous revenue/price column is also almost always ~100% unique
                # just from cents-level precision. Only flag as id-like when the values are
                # integers AND (the name suggests an id, OR they form a perfect 1-step
                # sequential ramp, the unmistakable signature of an autoincrement key).
                numeric_vals = non_null
                is_int_like = float(((numeric_vals % 1) == 0).mean()) > 0.99

                is_sequential_ramp = False
                if is_int_like and unique_ratio >= 0.99:
                    sorted_vals = np.sort(numeric_vals.unique())
                    if len(sorted_vals) > 1:
                        is_sequential_ramp = bool(np.allclose(np.diff(sorted_vals), 1))

                name_suggests_id = looks_id_like_name and is_int_like and unique_ratio >= 0.9

                if is_sequential_ramp or name_suggests_id:
                    id_like_cols.append(col)
                else:
                    numeric_cols.append(col)
                continue

            # Only object/string-dtype columns are candidates for date parsing, so a numeric
            # value stored as text ("45.2") doesn't get misread as a timestamp either.
            date_frac = self._column_date_fraction(col)

            if date_frac >= 0.9:
                date_cols.append(col)
                continue

            numeric_frac = self._column_numeric_fraction(col)

            if numeric_frac >= 0.9:
                numeric_vals = pd.to_numeric(non_null, errors="coerce").dropna()
                is_int_like = float(((numeric_vals % 1) == 0).mean()) > 0.99 if len(numeric_vals) else False

                is_sequential_ramp = False
                if is_int_like and unique_ratio >= 0.99:
                    sorted_vals = np.sort(numeric_vals.unique())
                    if len(sorted_vals) > 1:
                        is_sequential_ramp = bool(np.allclose(np.diff(sorted_vals), 1))

                name_suggests_id = looks_id_like_name and is_int_like and unique_ratio >= 0.9

                if is_sequential_ramp or name_suggests_id:
                    id_like_cols.append(col)
                else:
                    numeric_cols.append(col)
                continue

            # Non-numeric, non-date: a categorical/dimension candidate if it has more than
            # one but not too many distinct values, and isn't just a free-text ID field.
            unique_count = non_null.nunique()

            if unique_count <= 1:
                continue

            if looks_id_like_name and unique_ratio >= 0.9:
                id_like_cols.append(col)
                continue

            if 2 <= unique_count <= max(300, int(n_rows * 0.5)):
                categorical_candidates.append(col)
            elif unique_ratio >= 0.3:
                # High-cardinality text that isn't a clean dimension — likely an identifier
                # or free-text field (names, notes, descriptions).
                id_like_cols.append(col)

        return {
            "numeric_cols": numeric_cols,
            "date_cols": date_cols,
            "id_like_cols": id_like_cols,
            "categorical_candidates": categorical_candidates
        }

    def _pick_metric_column(self, numeric_cols: list) -> str:
        if not numeric_cols:
            return None

        for col in numeric_cols:
            col_lower = str(col).lower()
            if any(kw in col_lower for kw in METRIC_NAME_HINTS):
                return col

        # No name match — fall back to the numeric column with the largest total magnitude,
        # a reasonable proxy for "the metric that matters" (revenue dwarfs a 1-5 rating column).
        best_col = None
        best_magnitude = -1.0

        for col in numeric_cols:
            magnitude = float(pd.to_numeric(self.df[col], errors="coerce").abs().sum())

            if magnitude > best_magnitude:
                best_magnitude = magnitude
                best_col = col

        return best_col or numeric_cols[0]

    def _pick_timeline_column(self, date_cols: list) -> str:
        if not date_cols:
            return None

        for col in date_cols:
            col_lower = str(col).lower()
            if any(kw in col_lower for kw in TIMELINE_NAME_HINTS):
                return col

        return date_cols[0]

    def _pick_entity_column(self, id_like_cols: list, categorical_candidates: list, exclude: set) -> str:
        """
        Pick the column that best represents a recurring "entity" (customer, client, account)
        for repeat-rate and per-entity metrics. Critically, a column that is ~100% unique per
        row (an order ID, a transaction ID) provides no repetition to measure retention from —
        so a candidate that actually recurs across records is preferred whenever one exists,
        even if a more "ID-shaped" column is also available.
        """
        pool = [c for c in (id_like_cols + categorical_candidates) if c not in exclude]

        if not pool:
            return None

        def cardinality_and_ratio(col):
            non_null = self.df[col].dropna()
            if len(non_null) == 0:
                return 0, 1.0
            return non_null.nunique(), non_null.nunique() / len(non_null)

        def name_bonus(col):
            return 1 if any(kw in str(col).lower() for kw in ENTITY_NAME_HINTS) else 0

        repeating_candidates = []
        unique_only_candidates = []

        for col in pool:
            card, ratio = cardinality_and_ratio(col)

            if card < 2:
                continue

            if ratio <= 0.9:
                repeating_candidates.append((col, card))
            else:
                unique_only_candidates.append((col, card))

        if repeating_candidates:
            repeating_candidates.sort(key=lambda t: (-name_bonus(t[0]), -t[1]))
            return repeating_candidates[0][0]

        if unique_only_candidates:
            unique_only_candidates.sort(key=lambda t: (-name_bonus(t[0]), -t[1]))
            return unique_only_candidates[0][0]

        return None

    def infer_schema(self) -> dict:
        roles = self._detect_column_roles()
        columns = self.df.columns.tolist()

        metric_column = self._pick_metric_column(roles["numeric_cols"])
        timeline_column = self._pick_timeline_column(roles["date_cols"])

        exclude = {c for c in (metric_column, timeline_column) if c}
        entity_column = self._pick_entity_column(roles["id_like_cols"], roles["categorical_candidates"], exclude)

        # Any remaining numeric columns (not chosen as the metric) can still work fine as
        # low-cardinality dimensions (e.g. a "Rating" 1-5 column, a "Year" column) — but we
        # keep the categorical candidate list itself restricted to what _detect_column_roles
        # already classified, to avoid dumping high-cardinality numeric IDs in as dimensions.
        dimension_candidates = [
            c for c in roles["categorical_candidates"]
            if c not in exclude and c != entity_column
        ]

        # Prefer lower-cardinality columns first — they tend to be the more useful,
        # cleaner grouping dimensions (Region, Category, Status) versus noisier ones.
        dimension_candidates = sorted(
            dimension_candidates,
            key=lambda c: self.df[c].dropna().nunique()
        )[:MAX_DIMENSION_CANDIDATES]

        default_dimensions = dimension_candidates[:DEFAULT_DIMENSION_COUNT]

        mapping = {
            "metric_column": metric_column,
            "timeline_column": timeline_column,
            "entity_column": entity_column,
            "dimension_columns": default_dimensions
        }

        return {
            "columns": columns,
            "inferred_mapping": mapping,
            "numeric_columns": roles["numeric_cols"],
            "date_columns": roles["date_cols"],
            "categorical_columns": dimension_candidates,
            "date_range": self.get_date_range(timeline_column),
            "baseline_date": ANALYTICAL_BASELINE.strftime("%Y-%m-%d"),
            "data_quality": self.assess_data_quality(mapping)
        }

    def assess_data_quality(self, mapping: dict) -> dict:
        """
        Surface data quality issues in the mapped columns before analysis runs, so problems
        like blank metric values or unparseable dates are visible instead of silently dropped.
        """
        total_rows = int(len(self.df))
        warnings = []

        report = {
            "total_rows": total_rows,
            "duplicate_rows": int(self.df.duplicated().sum()) if total_rows > 0 else 0,
            "metric_column": None,
            "timeline_column": None,
            "dimension_columns": {},
            "warnings": warnings
        }

        if total_rows == 0:
            warnings.append("The uploaded file has no data rows.")
            return report

        if report["duplicate_rows"] > 0:
            dup_pct = report["duplicate_rows"] / total_rows * 100
            warnings.append(f"{report['duplicate_rows']:,} fully duplicate rows found ({dup_pct:.1f}% of the file).")

        metric_col = mapping.get("metric_column")

        if metric_col and metric_col in self.df.columns:
            raw = self.df[metric_col]
            numeric = pd.to_numeric(raw, errors="coerce")
            missing_pct = float(numeric.isna().mean() * 100)
            non_positive_pct = float((numeric.fillna(0) <= 0).mean() * 100)

            report["metric_column"] = {
                "column": metric_col,
                "missing_or_non_numeric_pct": round(missing_pct, 1),
                "zero_or_negative_pct": round(non_positive_pct, 1)
            }

            if missing_pct >= 5:
                warnings.append(f"'{metric_col}' has {missing_pct:.1f}% blank or non-numeric values.")
            if non_positive_pct >= 5:
                warnings.append(f"'{metric_col}' has {non_positive_pct:.1f}% zero or negative values.")

        time_col = mapping.get("timeline_column")

        if time_col and time_col in self.df.columns:
            raw = self.df[time_col]
            parsed = pd.to_datetime(raw, errors="coerce")
            unparseable_pct = float(parsed.isna().mean() * 100)

            valid_parsed = parsed.dropna()
            before_baseline_pct = float((valid_parsed < ANALYTICAL_BASELINE).mean() * 100) if not valid_parsed.empty else 0.0
            today = pd.Timestamp(datetime.now().date())
            future_pct = float((valid_parsed > today).mean() * 100) if not valid_parsed.empty else 0.0

            report["timeline_column"] = {
                "column": time_col,
                "unparseable_pct": round(unparseable_pct, 1),
                "before_baseline_pct": round(before_baseline_pct, 1),
                "future_dated_pct": round(future_pct, 1)
            }

            if unparseable_pct >= 2:
                warnings.append(f"'{time_col}' has {unparseable_pct:.1f}% unparseable dates that will be excluded.")
            if before_baseline_pct >= 5:
                warnings.append(
                    f"'{time_col}' has {before_baseline_pct:.1f}% of dates before {ANALYTICAL_BASELINE.strftime('%Y-%m-%d')} "
                    f"(outside the analytical baseline and excluded)."
                )
            if future_pct >= 5:
                warnings.append(f"'{time_col}' has {future_pct:.1f}% future-dated records.")

        for col in (mapping.get("dimension_columns") or []):
            if not col or col not in self.df.columns:
                continue

            missing_pct = float(self.df[col].isna().mean() * 100)
            report["dimension_columns"][col] = {"column": col, "missing_pct": round(missing_pct, 1)}

            if missing_pct >= 10:
                warnings.append(f"'{col}' is {missing_pct:.1f}% blank.")

        entity_col = mapping.get("entity_column")

        if entity_col and entity_col in self.df.columns:
            missing_pct = float(self.df[entity_col].isna().mean() * 100)
            if missing_pct >= 10:
                warnings.append(f"'{entity_col}' (entity column) is {missing_pct:.1f}% blank.")

        return report

    # ----------------------------------------------------------------------- #
    # Forecasting
    # ----------------------------------------------------------------------- #

    def _empty_forecast_outlook(self, projection_target="value", metric_label="metric value"):
        return {
            "metric_type": projection_target,
            "current_year": None,
            "previous_year": None,
            "current_actual": 0.0,
            "previous_year_actual": 0.0,
            "projected_year_end": 0.0,
            "conservative_year_end": 0.0,
            "aggressive_year_end": 0.0,
            "remaining_months": 0,
            "growth_vs_previous_year_pct": 0.0,
            "confidence_score": 0.0,
            "confidence_label": "Insufficient Data",
            "trend_direction": "Flat",
            "monthly_forecast": [],
            "seasonal_index": {},
            "executive_summary": "Not enough data is available to produce a reliable year-end projection."
        }

    def _build_monthly_series(self, df, time_col, metric_col, entity_col, target_series="value"):
        """Aggregate a (already scoped/filtered) dataframe into a monthly time series."""
        if df is None or df.empty or time_col not in df.columns:
            return pd.DataFrame(columns=["YearMonth", "value", "count"])

        d = df.copy()
        d[time_col] = pd.to_datetime(d[time_col], errors="coerce")
        d = d.dropna(subset=[time_col])

        if d.empty:
            return pd.DataFrame(columns=["YearMonth", "value", "count"])

        d["YearMonth"] = d[time_col].dt.to_period("M")
        grouped = d.groupby("YearMonth")

        monthly = pd.DataFrame({
            "value": grouped[metric_col].sum() if metric_col in d.columns else grouped.size(),
            "count": grouped[entity_col].nunique() if entity_col and entity_col in d.columns else grouped.size()
        }).reset_index()

        return monthly.sort_values("YearMonth").reset_index(drop=True)

    def _seasonal_trend_forecast(self, monthly_df, target_series, periods_ahead):
        """
        Produce a trend + seasonality aware forecast for the next `periods_ahead` months.

        Method: fit a linear trend across all available months, derive a per-calendar-month
        seasonal multiplier from the average ratio of actual-to-trend, normalize the
        multipliers to average 1.0, then project future months as trend * seasonal index.
        Confidence bands are built from the standard deviation of de-seasonalized residuals,
        widened with distance into the future (uncertainty compounds over a longer horizon).
        """
        empty_diagnostics = {"r_squared": 0.0, "residual_std": 0.0, "slope": 0.0, "seasonal_index": {}}

        if monthly_df is None or monthly_df.empty or len(monthly_df) < 2 or periods_ahead <= 0:
            return [], empty_diagnostics

        mdf = monthly_df.copy().sort_values("YearMonth").reset_index(drop=True)
        mdf["CalMonth"] = mdf["YearMonth"].dt.month

        X = np.arange(len(mdf)).reshape(-1, 1)
        y = mdf[target_series].astype(float).values

        model = LinearRegression().fit(X, y)
        trend_fitted = model.predict(X)

        try:
            r_squared = float(model.score(X, y))
        except Exception:
            r_squared = 0.0

        safe_trend = np.where(np.abs(trend_fitted) < 1e-9, np.nan, trend_fitted)

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = y / safe_trend

        mdf["_ratio"] = ratio

        seasonal_index = {}
        for m in range(1, 13):
            vals = mdf.loc[mdf["CalMonth"] == m, "_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
            seasonal_index[m] = float(vals.mean()) if len(vals) > 0 else 1.0

        observed_vals = [v for v in seasonal_index.values() if v is not None and not np.isnan(v)]
        mean_idx = float(np.mean(observed_vals)) if observed_vals else 1.0

        if mean_idx and not np.isnan(mean_idx) and mean_idx != 0:
            seasonal_index = {m: v / mean_idx for m, v in seasonal_index.items()}

        # Clip extreme seasonal swings so sparse-history months don't produce wild forecasts.
        seasonal_index = {m: float(np.clip(v, 0.4, 2.5)) for m, v in seasonal_index.items()}

        deseasonalized_fitted = trend_fitted * np.array([seasonal_index.get(m, 1.0) for m in mdf["CalMonth"]])
        residuals = y - deseasonalized_fitted
        residual_std = float(np.std(residuals)) if len(residuals) > 1 else 0.0

        last_period = mdf["YearMonth"].max()
        future_X = np.arange(len(mdf), len(mdf) + periods_ahead).reshape(-1, 1)
        future_trend = model.predict(future_X)

        future = []

        for i, base_pred in enumerate(future_trend):
            months_ahead = i + 1
            future_period = last_period + i + 1
            cal_month = int(future_period.month)
            seasonal_mult = seasonal_index.get(cal_month, 1.0)
            expected = max(0.0, float(base_pred) * seasonal_mult)

            # Uncertainty compounds with distance into the future rather than staying fixed —
            # a forecast for next month should be tighter than one for 30 months out.
            band_width = residual_std * float(np.sqrt(months_ahead))
            conservative = max(0.0, expected - band_width)
            aggressive = max(0.0, expected + band_width)

            future.append({
                "period": str(future_period),
                "year": int(future_period.year),
                "month": cal_month,
                "months_ahead": months_ahead,
                "expected_value": expected,
                "conservative_value": conservative,
                "aggressive_value": aggressive
            })

        slope = float(model.coef_[0]) if hasattr(model, "coef_") and len(model.coef_) > 0 else 0.0

        return future, {
            "r_squared": r_squared,
            "residual_std": residual_std,
            "slope": slope,
            "seasonal_index": seasonal_index
        }

    def _compute_forecast_outlook(self, monthly_df, target_series, projection_target, metric_label="metric value"):
        if monthly_df is None or monthly_df.empty or len(monthly_df) < 2:
            return self._empty_forecast_outlook(projection_target, metric_label)

        forecast_df = monthly_df.copy().sort_values("YearMonth").reset_index(drop=True)
        forecast_df["Year"] = forecast_df["YearMonth"].dt.year

        current_year = int(forecast_df["Year"].max())
        previous_year = current_year - 1

        current_year_df = forecast_df[forecast_df["Year"] == current_year]
        previous_year_df = forecast_df[forecast_df["Year"] == previous_year]

        current_actual = float(current_year_df[target_series].sum())
        previous_year_actual = float(previous_year_df[target_series].sum()) if not previous_year_df.empty else 0.0

        last_period = forecast_df["YearMonth"].max()
        remaining_months = max(0, 12 - int(last_period.month))

        future_monthly, diagnostics = self._seasonal_trend_forecast(forecast_df, target_series, remaining_months)

        r_squared = diagnostics["r_squared"]
        residual_std = diagnostics["residual_std"]
        slope = diagnostics["slope"]

        avg_monthly = float(forecast_df[target_series].astype(float).mean()) if len(forecast_df) > 0 else 0.0
        volatility_ratio = abs(residual_std / avg_monthly) if avg_monthly != 0 else 1.0

        expected_future_total = sum(item["expected_value"] for item in future_monthly)
        conservative_future_total = sum(item["conservative_value"] for item in future_monthly)
        aggressive_future_total = sum(item["aggressive_value"] for item in future_monthly)

        projected_year_end = float(current_actual + expected_future_total)
        conservative_year_end = float(current_actual + conservative_future_total)
        aggressive_year_end = float(current_actual + aggressive_future_total)

        growth_vs_previous_year_pct = 0.0

        if previous_year_actual > 0:
            growth_vs_previous_year_pct = ((projected_year_end - previous_year_actual) / previous_year_actual) * 100

        if slope > 0:
            trend_direction = "Increasing"
        elif slope < 0:
            trend_direction = "Decreasing"
        else:
            trend_direction = "Flat"

        data_points = len(forecast_df)
        history_score = min(1.0, data_points / 12)
        fit_score = max(0.0, min(1.0, r_squared))
        stability_score = max(0.0, min(1.0, 1 - volatility_ratio))

        confidence_score = round(((history_score * 0.35) + (fit_score * 0.40) + (stability_score * 0.25)) * 100, 1)

        if data_points < 4:
            confidence_label = "Low"
        elif confidence_score >= 75:
            confidence_label = "High"
        elif confidence_score >= 50:
            confidence_label = "Moderate"
        else:
            confidence_label = "Low"

        if previous_year_actual > 0:
            growth_phrase = f"{growth_vs_previous_year_pct:.1f}% compared with the prior year"
        else:
            growth_phrase = "no prior-year comparison is available"

        executive_summary = (
            f"Based on current monthly performance and seasonal patterns, the selected data is projected to finish "
            f"{current_year} at approximately {projected_year_end:,.0f} in {metric_label}. "
            f"This represents {growth_phrase}. "
            f"The forecast confidence is {confidence_label.lower()} based on available history, trend fit, and volatility. "
            f"See the Goals panel for goal-specific pacing and projections."
        )

        return {
            "metric_type": projection_target,
            "current_year": current_year,
            "previous_year": previous_year,
            "current_actual": float(current_actual),
            "previous_year_actual": float(previous_year_actual),
            "projected_year_end": float(projected_year_end),
            "conservative_year_end": float(conservative_year_end),
            "aggressive_year_end": float(aggressive_year_end),
            "remaining_months": int(remaining_months),
            "growth_vs_previous_year_pct": float(growth_vs_previous_year_pct),
            "confidence_score": float(confidence_score),
            "confidence_label": confidence_label,
            "trend_direction": trend_direction,
            "monthly_forecast": future_monthly,
            "seasonal_index": diagnostics.get("seasonal_index", {}),
            "executive_summary": executive_summary
        }

    def _empty_long_range_forecast(self, projection_target="value", horizon_months=24, metric_label="metric value"):
        return {
            "metric_type": projection_target,
            "horizon_months": horizon_months,
            "trend_direction": "Flat",
            "confidence_label": "Insufficient Data",
            "monthly": [],
            "annual_rollup": [],
            "executive_summary": "Not enough monthly history is available to build a multi-year projection."
        }

    def _compute_long_range_forecast(self, monthly_df, target_series, projection_target, horizon_months=24, metric_label="metric value"):
        """
        Project `horizon_months` months forward from the latest data point (not tied to the
        current calendar year), and roll those months up into per-calendar-year totals —
        useful for multi-year planning rather than just a year-end estimate.
        """
        horizon_months = int(np.clip(horizon_months, 1, 60))

        if monthly_df is None or monthly_df.empty or len(monthly_df) < 3:
            return self._empty_long_range_forecast(projection_target, horizon_months, metric_label)

        future_monthly, diagnostics = self._seasonal_trend_forecast(monthly_df, target_series, horizon_months)

        if not future_monthly:
            return self._empty_long_range_forecast(projection_target, horizon_months, metric_label)

        annual = {}

        for item in future_monthly:
            year = item["year"]
            bucket = annual.setdefault(year, {
                "expected_total": 0.0, "conservative_total": 0.0, "aggressive_total": 0.0, "months_included": 0
            })
            bucket["expected_total"] += item["expected_value"]
            bucket["conservative_total"] += item["conservative_value"]
            bucket["aggressive_total"] += item["aggressive_value"]
            bucket["months_included"] += 1

        annual_rollup = [
            {
                "year": int(year),
                "projected_total": float(vals["expected_total"]),
                "conservative_total": float(vals["conservative_total"]),
                "aggressive_total": float(vals["aggressive_total"]),
                "months_included": int(vals["months_included"]),
                "is_partial_year": vals["months_included"] < 12
            }
            for year, vals in sorted(annual.items())
        ]

        slope = diagnostics.get("slope", 0.0)

        if slope > 0:
            trend_direction = "Increasing"
        elif slope < 0:
            trend_direction = "Decreasing"
        else:
            trend_direction = "Flat"

        data_points = len(monthly_df)
        r_squared = diagnostics.get("r_squared", 0.0)

        if data_points < 6:
            confidence_label = "Low"
        elif r_squared >= 0.5 and data_points >= 18:
            confidence_label = "Moderate"
        else:
            confidence_label = "Low"

        far_year = annual_rollup[-1]["year"] if annual_rollup else None

        executive_summary = (
            f"Projecting {horizon_months} months forward from the latest available data, the trend is "
            f"{trend_direction.lower()} in {metric_label}. "
            f"Confidence in the {far_year} figures is necessarily lower than near-term months — "
            f"the projected range widens the further out the estimate reaches, since a trend fit on "
            f"{data_points} months of history compounds its own uncertainty over a multi-year horizon. "
            f"Treat years beyond the first 12 months as directional planning input rather than a firm commitment."
        )

        return {
            "metric_type": projection_target,
            "horizon_months": horizon_months,
            "trend_direction": trend_direction,
            "confidence_label": confidence_label,
            "monthly": future_monthly,
            "annual_rollup": annual_rollup,
            "executive_summary": executive_summary
        }

    # ----------------------------------------------------------------------- #
    # Goals: period bounds, scoping to any dimension column, pacing + projection
    # ----------------------------------------------------------------------- #

    def _period_bounds(self, period: str, anchor: pd.Timestamp):
        """Return (start, end) timestamps for the annual/quarterly/monthly window containing `anchor`."""
        anchor = pd.Timestamp(anchor)

        if period == "monthly":
            start = pd.Timestamp(year=anchor.year, month=anchor.month, day=1)
            end = start + pd.DateOffset(months=1) - pd.Timedelta(days=1)
        elif period == "quarterly":
            q_start_month = ((anchor.month - 1) // 3) * 3 + 1
            start = pd.Timestamp(year=anchor.year, month=q_start_month, day=1)
            end = start + pd.DateOffset(months=3) - pd.Timedelta(days=1)
        else:
            start = pd.Timestamp(year=anchor.year, month=1, day=1)
            end = pd.Timestamp(year=anchor.year, month=12, day=31)

        return start, end

    def _apply_dimension_scope(self, df, scope_column, scope_value):
        """Filter a dataframe down to rows where `scope_column` equals `scope_value`."""
        if not scope_column or scope_column not in df.columns:
            return df

        normalized = self._normalize_categorical_value(scope_value)
        return df[df[scope_column].astype(str).map(self._normalize_categorical_value) == normalized]

    def compute_goals(
        self,
        working_df,
        metric_col,
        entity_col,
        time_col,
        projection_target,
        goals,
        anchor_date
    ) -> list:
        """
        Compute pacing + a seasonality-aware projection for each goal in `goals`.

        Each goal dict supports: id, label, period ("annual"/"quarterly"/"monthly"),
        scope_type ("overall"/"dimension"), scope_column (required when scope_type is
        "dimension" — any column name from the file), scope_value, target_value.
        """
        target_series = "value" if projection_target == "value" else "count"
        results = []

        if not goals:
            return results

        if anchor_date is None or pd.isna(anchor_date):
            anchor_date = pd.Timestamp(datetime.now().date())

        for idx, g in enumerate(goals):
            goal_id = g.get("id") or f"goal_{idx + 1}"
            label = g.get("label") or "Goal"
            period = g.get("period") if g.get("period") in ("annual", "quarterly", "monthly") else "annual"
            scope_type = g.get("scope_type") if g.get("scope_type") in ("overall", "dimension") else "overall"
            scope_column = g.get("scope_column") if scope_type == "dimension" else None
            scope_value = g.get("scope_value") if scope_type == "dimension" else None

            try:
                target_value = float(g.get("target_value") or 0)
            except (ValueError, TypeError):
                target_value = 0.0

            start, end = self._period_bounds(period, anchor_date)
            base_result = {
                "id": goal_id,
                "label": label,
                "period": period,
                "scope_type": scope_type,
                "scope_column": scope_column,
                "scope_value": scope_value,
                "metric_type": projection_target,
                "period_start": start.strftime("%Y-%m-%d"),
                "period_end": end.strftime("%Y-%m-%d"),
                "days_total": int((end - start).days + 1)
            }

            scoped_df = self._apply_dimension_scope(working_df, scope_column, scope_value) if scope_type == "dimension" else working_df

            if target_value <= 0:
                results.append({
                    **base_result,
                    "target": 0.0,
                    "actual": 0.0,
                    "achievement_pct": 0.0,
                    "expected_pct": 0.0,
                    "gap_to_goal": 0.0,
                    "projected_period_end": 0.0,
                    "projected_period_end_low": None,
                    "projected_period_end_high": None,
                    "days_elapsed": 0,
                    "status": "no_goal_set",
                    "trend": []
                })
                continue

            effective_end_for_actual = min(end, anchor_date)
            in_period = (scoped_df[time_col] >= start) & (scoped_df[time_col] <= effective_end_for_actual)
            period_df = scoped_df[in_period]

            if projection_target == "value":
                actual = float(period_df[metric_col].sum()) if not period_df.empty else 0.0
            else:
                actual = float(period_df[entity_col].nunique()) if entity_col and not period_df.empty else float(len(period_df))

            days_total = max(1, base_result["days_total"])
            days_elapsed = max(0, (effective_end_for_actual - start).days + 1) if effective_end_for_actual >= start else 0

            expected_pct = min(100.0, (days_elapsed / days_total) * 100)
            achievement_pct = (actual / target_value) * 100
            gap_to_goal = float(actual - target_value)

            projected_period_end = float((actual / days_elapsed) * days_total) if days_elapsed > 0 else 0.0
            projected_period_end_low = None
            projected_period_end_high = None

            # Where at least one full calendar month remains in the goal period, refine the
            # run-rate projection using the seasonality-aware monthly forecast instead.
            anchor_month_period = pd.Period(anchor_date, freq="M")
            end_month_period = pd.Period(end, freq="M")

            if anchor_month_period < end_month_period:
                remaining_months = int(end_month_period.ordinal - anchor_month_period.ordinal)
                monthly_series = self._build_monthly_series(scoped_df, time_col, metric_col, entity_col, target_series)

                if len(monthly_series) >= 2:
                    future_vals, _diag = self._seasonal_trend_forecast(monthly_series, target_series, remaining_months)
                    remaining_forecast = sum(item["expected_value"] for item in future_vals)
                    remaining_conservative = sum(item["conservative_value"] for item in future_vals)
                    remaining_aggressive = sum(item["aggressive_value"] for item in future_vals)

                    projected_period_end = float(actual + remaining_forecast)
                    projected_period_end_low = float(actual + remaining_conservative)
                    projected_period_end_high = float(actual + remaining_aggressive)

            if expected_pct <= 0:
                status = "no_time_elapsed"
            else:
                pace_ratio = achievement_pct / expected_pct

                if pace_ratio >= 1.0:
                    status = "ahead"
                elif pace_ratio >= 0.8:
                    status = "on_pace"
                else:
                    status = "behind"

            trend = self._build_goal_trend(
                scoped_df, time_col, metric_col, entity_col, projection_target, start, effective_end_for_actual, days_total
            )

            results.append({
                **base_result,
                "target": float(target_value),
                "actual": float(actual),
                "achievement_pct": float(achievement_pct),
                "expected_pct": float(expected_pct),
                "gap_to_goal": float(gap_to_goal),
                "projected_period_end": float(projected_period_end),
                "projected_period_end_low": projected_period_end_low,
                "projected_period_end_high": projected_period_end_high,
                "days_elapsed": int(days_elapsed),
                "status": status,
                "trend": trend
            })

        return results

    def _build_goal_trend(self, scoped_df, time_col, metric_col, entity_col, projection_target, start, anchor, days_total, max_points: int = 10) -> list:
        """
        Build a downsampled cumulative-actual-vs-expected-pace series for a goal's sparkline,
        covering from the period start through the anchor date (today, or the data's latest date).
        """
        if scoped_df is None or scoped_df.empty or anchor < start:
            return []

        mask = (scoped_df[time_col] >= start) & (scoped_df[time_col] <= anchor)
        subset = scoped_df.loc[mask]

        if subset.empty:
            return []

        day_index = subset[time_col].dt.date

        if projection_target == "value":
            daily = subset.groupby(day_index)[metric_col].sum().sort_index()
            cumulative = daily.cumsum()
        elif entity_col and entity_col in subset.columns:
            first_seen_in_period = subset.groupby(entity_col)[time_col].min()
            daily_new = first_seen_in_period.dt.date.value_counts().sort_index()
            cumulative = daily_new.cumsum()
        else:
            daily = subset.groupby(day_index).size().sort_index()
            cumulative = daily.cumsum()

        if cumulative.empty:
            return []

        dates = cumulative.index.to_list()
        values = cumulative.values.tolist()
        total_points = len(values)
        n_points = min(max_points, total_points)

        if n_points <= 1:
            sample_positions = [total_points - 1]
        else:
            sample_positions = sorted(set(
                round(i * (total_points - 1) / (n_points - 1)) for i in range(n_points)
            ))

        trend = []

        for pos in sample_positions:
            elapsed_days = (dates[pos] - start.date()).days + 1
            expected_pct = min(100.0, (elapsed_days / max(1, days_total)) * 100)

            trend.append({
                "date": dates[pos].strftime("%Y-%m-%d"),
                "cumulative_actual": float(values[pos]),
                "expected_pct_at_date": float(expected_pct)
            })

        return trend

    # ----------------------------------------------------------------------- #
    # Goal suggestions ("based on past info, here's what to aim for")
    # ----------------------------------------------------------------------- #

    def _reference_period_bounds(self, period: str, anchor: pd.Timestamp):
        """The same period one year prior — the historical baseline a suggested goal is built from."""
        start, end = self._period_bounds(period, anchor)
        return start - pd.DateOffset(years=1), end - pd.DateOffset(years=1)

    def _period_actual(self, df, time_col, metric_col, entity_col, projection_target, start, end):
        if df is None or df.empty:
            return 0.0

        mask = (df[time_col] >= start) & (df[time_col] <= end)
        subset = df[mask]

        if subset.empty:
            return 0.0

        if projection_target == "value":
            return float(subset[metric_col].sum())

        return float(subset[entity_col].nunique()) if entity_col else float(len(subset))

    def _estimate_growth_rate(self, df, time_col, metric_col, entity_col, projection_target, period, anchor_date):
        """
        Compare the most recent complete comparable period to the one a year before it
        (e.g. last calendar year vs the year before). Returns None if there isn't enough
        history for this scope to compute a meaningful rate.
        """
        ref_start, ref_end = self._reference_period_bounds(period, anchor_date)
        prior_start, prior_end = ref_start - pd.DateOffset(years=1), ref_end - pd.DateOffset(years=1)

        ref_actual = self._period_actual(df, time_col, metric_col, entity_col, projection_target, ref_start, ref_end)
        prior_actual = self._period_actual(df, time_col, metric_col, entity_col, projection_target, prior_start, prior_end)

        if prior_actual <= 0:
            return None

        growth = (ref_actual - prior_actual) / prior_actual

        # Clip so a single noisy small-scope period can't produce an absurd suggestion.
        return float(np.clip(growth, -0.6, 1.5))

    def _round_target(self, value: float) -> float:
        """Round a suggested target to a clean, presentable figure (e.g. 6,432,911 -> 6,400,000)."""
        if value is None or value <= 0:
            return 0.0

        digits = len(str(int(value)))
        magnitude = max(1, 10 ** max(0, digits - 2))

        return float(round(value / magnitude) * magnitude)

    def suggest_goal_candidates(
        self,
        mapping: dict,
        projection_target: str = "value",
        period: str = "annual",
        top_n: int = 3
    ) -> list:
        """
        Propose goal candidates from historical performance: overall, plus the top N values
        in each configured dimension column, by volume. Each candidate reports last year's
        actual for the same period, the trailing growth rate, and three target tiers
        (maintain / grow / stretch) derived from those two numbers.
        """
        metric_col = mapping.get("metric_column")
        time_col = mapping.get("timeline_column")
        entity_col = mapping.get("entity_column")
        dimension_cols = [c for c in (mapping.get("dimension_columns") or []) if c in self.df.columns]

        if not metric_col or not time_col or metric_col not in self.df.columns or time_col not in self.df.columns:
            raise ValueError("Metric and Timeline columns must be mapped and present to suggest goals.")

        entity_col = entity_col if entity_col in self.df.columns else None

        cols = list(dict.fromkeys([metric_col, time_col] + dimension_cols + ([entity_col] if entity_col else [])))

        working_df = self.df[cols].copy()
        working_df[time_col] = pd.to_datetime(working_df[time_col], errors="coerce")
        working_df[metric_col] = pd.to_numeric(working_df[metric_col], errors="coerce").fillna(0)
        working_df = working_df.dropna(subset=[time_col])
        working_df = working_df[working_df[time_col] >= ANALYTICAL_BASELINE]

        for col in dimension_cols:
            working_df[col] = working_df[col].apply(self._normalize_categorical_value)

        if entity_col:
            working_df[entity_col] = working_df[entity_col].fillna("Unknown")

        if working_df.empty:
            return []

        anchor_date = working_df[time_col].max()

        # Organization-wide growth rate, used as a fallback for scopes with too little
        # standalone history (e.g. a small segment with only a few months of data).
        overall_growth = self._estimate_growth_rate(
            working_df, time_col, metric_col, entity_col, projection_target, period, anchor_date
        )

        scope_candidates = [("overall", None, None, "Overall")]

        def top_values(col):
            if not col or col not in working_df.columns:
                return []

            sums = working_df.groupby(col)[metric_col].sum().sort_values(ascending=False)
            return list(sums.head(top_n).index)

        for col in dimension_cols:
            for val in top_values(col):
                scope_candidates.append(("dimension", col, val, f"{col}: {val}"))

        suggestions = []

        for scope_type, scope_column, scope_value, scope_label in scope_candidates:
            scoped_df = self._apply_dimension_scope(working_df, scope_column, scope_value) if scope_type == "dimension" else working_df

            ref_start, ref_end = self._reference_period_bounds(period, anchor_date)
            cur_start, cur_end = self._period_bounds(period, anchor_date)

            reference_actual = self._period_actual(
                scoped_df, time_col, metric_col, entity_col, projection_target, ref_start, ref_end
            )

            if reference_actual <= 0:
                continue

            growth = self._estimate_growth_rate(
                scoped_df, time_col, metric_col, entity_col, projection_target, period, anchor_date
            )
            data_sufficient = growth is not None

            if growth is None:
                growth = overall_growth if overall_growth is not None else 0.0

            suggestions.append({
                "scope_type": scope_type,
                "scope_column": scope_column,
                "scope_value": scope_value,
                "scope_label": scope_label,
                "period": period,
                "period_start": cur_start.strftime("%Y-%m-%d"),
                "period_end": cur_end.strftime("%Y-%m-%d"),
                "reference_period_start": ref_start.strftime("%Y-%m-%d"),
                "reference_period_end": ref_end.strftime("%Y-%m-%d"),
                "reference_actual": float(reference_actual),
                "growth_rate_pct": float(growth * 100),
                "data_sufficient": data_sufficient,
                "metric_type": projection_target,
                "suggestions": {
                    "maintain": self._round_target(reference_actual),
                    "grow": self._round_target(reference_actual * (1 + growth)),
                    "stretch": self._round_target(reference_actual * (1 + growth + 0.05))
                }
            })

        return suggestions

    # ----------------------------------------------------------------------- #
    # Entity recurrence (new vs. repeat) — derived automatically from the entity
    # column's first-appearance date, no separate "type" mapping required.
    # ----------------------------------------------------------------------- #

    def _classify_entity_recurrence(self, working_df, entity_col, time_col):
        """
        Classify each row as "New" (the entity's first appearance, within 30 days of its
        earliest record) or "Repeat" (a later appearance of an already-seen entity). If
        there's no entity column, everything is "Unclassified" — there's no way to tell
        new from repeat without something to link repeat occurrences.
        """
        if not entity_col or entity_col not in working_df.columns:
            working_df["EntityStatus"] = "Unclassified"
            return working_df

        first_seen = self.df.copy()
        first_seen[time_col] = pd.to_datetime(first_seen[time_col], errors="coerce")
        first_seen = first_seen.dropna(subset=[time_col])
        entity_first_dates = first_seen.groupby(entity_col)[time_col].min()

        def classify_row(row):
            entity = row[entity_col]

            if entity in entity_first_dates.index:
                first_date = entity_first_dates[entity]

                if pd.notna(row[time_col]) and abs((row[time_col] - first_date).days) <= 30:
                    return "New"

                return "Repeat"

            return "Repeat"

        working_df["EntityStatus"] = working_df.apply(classify_row, axis=1)
        return working_df

    def _detect_trend_anomalies(
        self, working_df, time_col, metric_col, entity_col, target_series, dimension_col, avg_monthly_scope_total
    ) -> list:
        """
        Flag values within a dimension column whose most recent month deviates sharply from
        their own trailing 3-month average — a spike or drop worth a human look, distinct
        from the static "largest entities" concentration check elsewhere.

        `avg_monthly_scope_total` is the whole dataset's average monthly volume, used to
        filter out entities too small to be worth flagging (compared like-for-like against
        a typical month, not the multi-year cumulative total).
        """
        flags = []

        if not dimension_col or dimension_col not in working_df.columns:
            return flags

        for value in working_df[dimension_col].dropna().unique():
            entity_df = working_df[working_df[dimension_col] == value]
            monthly = self._build_monthly_series(entity_df, time_col, metric_col, entity_col, target_series)

            if len(monthly) < 4:
                continue

            monthly = monthly.sort_values("YearMonth").reset_index(drop=True)
            latest_value = float(monthly[target_series].iloc[-1])
            trailing_window = monthly[target_series].iloc[-4:-1]
            baseline = float(trailing_window.mean()) if len(trailing_window) > 0 else 0.0

            if baseline <= 0:
                continue

            # Ignore entities too small to matter relative to the dataset's typical month.
            if avg_monthly_scope_total > 0 and (baseline / avg_monthly_scope_total) < 0.03:
                continue

            deviation = (latest_value - baseline) / baseline

            if abs(deviation) >= 0.4:
                flags.append({
                    "dimension_column": dimension_col,
                    "scope_value": str(value),
                    "latest_period": str(monthly["YearMonth"].iloc[-1]),
                    "latest_value": latest_value,
                    "trailing_avg": baseline,
                    "deviation_pct": float(deviation * 100),
                    "direction": "spike" if deviation > 0 else "drop"
                })

        return flags

    # ----------------------------------------------------------------------- #
    # AI insights
    # ----------------------------------------------------------------------- #

    def _empty_ai_insights(self):
        return {
            "portfolio_health_score": 0.0,
            "portfolio_health_label": "Insufficient Data",
            "portfolio_health_status": "neutral",
            "executive_summary": "Not enough data is available to generate insights.",
            "insights": [],
            "recommended_actions": [
                "Upload or select a broader dataset to generate insights."
            ]
        }

    def _generate_ai_insights(
        self,
        total_metric_value,
        unique_entity_count,
        avg_entity_value,
        repeat_rate,
        hhi_index,
        pareto_ratio,
        entity_split,
        forecast_outlook,
        primary_goal,
        dimension_data,
        anomalies,
        projection_target,
        metric_label
    ):
        insights = []
        action_items = []

        projected_year_end = float(forecast_outlook.get("projected_year_end", 0) or 0)
        growth_pct = float(forecast_outlook.get("growth_vs_previous_year_pct", 0) or 0)
        confidence_label = forecast_outlook.get("confidence_label", "Insufficient Data")
        trend_direction = forecast_outlook.get("trend_direction", "Flat")

        goal_status = "No Goal Set"
        projected_gap_to_goal = 0.0

        if primary_goal and float(primary_goal.get("target", 0) or 0) > 0:
            projected_gap_to_goal = float(primary_goal.get("projected_period_end", 0) or 0) - float(primary_goal.get("target", 0) or 0)
            goal_status = "Projected Above Goal" if projected_gap_to_goal >= 0 else "Projected Below Goal"

        new_value = float(entity_split.get("new_value", 0) or 0)
        repeat_value = float(entity_split.get("repeat_value", 0) or 0)
        new_count = int(entity_split.get("new_count", 0) or 0)
        repeat_count = int(entity_split.get("repeat_count", 0) or 0)
        recurrence_available = entity_split.get("classification_method") == "derived"

        total_split_value = new_value + repeat_value
        total_split_count = new_count + repeat_count

        new_value_share = (new_value / total_split_value * 100) if total_split_value > 0 else 0
        repeat_value_share = (repeat_value / total_split_value * 100) if total_split_value > 0 else 0
        new_count_share = (new_count / total_split_count * 100) if total_split_count > 0 else 0
        repeat_count_share = (repeat_count / total_split_count * 100) if total_split_count > 0 else 0

        top_dimension_name = None
        top_dimension_share = 0.0

        if dimension_data and len(dimension_data) > 0:
            try:
                top_dimension_name = max(dimension_data, key=dimension_data.get)
                dimension_total = sum(float(v or 0) for v in dimension_data.values())

                if dimension_total > 0:
                    top_dimension_share = float(dimension_data.get(top_dimension_name, 0) or 0) / dimension_total * 100
            except Exception:
                top_dimension_name = None
                top_dimension_share = 0.0

        anomaly_count = len(anomalies) if anomalies else 0

        if hhi_index >= 2500:
            concentration_level = "High"
        elif hhi_index >= 1500:
            concentration_level = "Moderate"
        else:
            concentration_level = "Low"

        if repeat_rate >= 90:
            retention_level = "Strong"
        elif repeat_rate >= 75:
            retention_level = "Watch"
        else:
            retention_level = "At Risk"

        if growth_pct >= 10:
            growth_level = "Accelerating"
        elif growth_pct >= 3:
            growth_level = "Growing"
        elif growth_pct <= -5:
            growth_level = "Declining"
        else:
            growth_level = "Flat"

        if projected_year_end > 0:
            insights.append({
                "category": "Forecast",
                "icon": "🔮",
                "severity": "positive" if growth_pct >= 0 else "warning",
                "title": "Year-End Projection",
                "message": (
                    f"Based on current monthly performance, the selected data is projected to finish at "
                    f"approximately {projected_year_end:,.0f} in {metric_label}. Forecast confidence is "
                    f"{confidence_label.lower()}."
                )
            })
        else:
            insights.append({
                "category": "Forecast",
                "icon": "🔮",
                "severity": "neutral",
                "title": "Forecast Availability",
                "message": "There is not enough monthly history available to produce a reliable year-end forecast."
            })

        if growth_level == "Accelerating":
            insights.append({
                "category": "Growth",
                "icon": "📈",
                "severity": "positive",
                "title": "Growth Momentum Is Strong",
                "message": (
                    f"The forecast indicates accelerating growth of approximately {growth_pct:.1f}% versus the prior year. "
                    f"Current trend direction is {trend_direction.lower()}."
                )
            })
            action_items.append(
                "Review the highest-performing dimensions to identify where growth is coming from and whether it can be replicated."
            )
        elif growth_level == "Growing":
            insights.append({
                "category": "Growth",
                "icon": "📈",
                "severity": "positive",
                "title": "Growth Trend Is Positive",
                "message": (
                    f"The data is projected to grow by approximately {growth_pct:.1f}% versus the prior year, "
                    f"suggesting positive but controlled expansion."
                )
            })
        elif growth_level == "Declining":
            insights.append({
                "category": "Growth",
                "icon": "📈",
                "severity": "risk",
                "title": "Growth Trend Is Declining",
                "message": (
                    f"The forecast indicates a decline of approximately {abs(growth_pct):.1f}% versus the prior year. "
                    f"This may warrant review of where volume is being lost."
                )
            })
            action_items.append(
                "Investigate whether the decline is concentrated in specific dimensions (regions, categories, channels, etc.)."
            )
        else:
            insights.append({
                "category": "Growth",
                "icon": "📈",
                "severity": "neutral",
                "title": "Growth Trend Is Relatively Flat",
                "message": (
                    f"The current forecast shows limited movement versus the prior year at approximately {growth_pct:.1f}%."
                )
            })

        if goal_status == "Projected Above Goal":
            insights.append({
                "category": "Goal",
                "icon": "🏁",
                "severity": "positive",
                "title": "Projected Above Goal",
                "message": (
                    f"Current trends suggest the selected scope may finish above goal by approximately "
                    f"{abs(projected_gap_to_goal):,.0f}."
                )
            })
        elif goal_status == "Projected Below Goal":
            insights.append({
                "category": "Goal",
                "icon": "🏁",
                "severity": "risk",
                "title": "Projected Below Goal",
                "message": (
                    f"Current trends suggest the selected scope may finish below goal by approximately "
                    f"{abs(projected_gap_to_goal):,.0f}."
                )
            })
            action_items.append(
                "Compare the required pace to recent performance to determine whether the gap is realistically recoverable."
            )
        else:
            insights.append({
                "category": "Goal",
                "icon": "🏁",
                "severity": "neutral",
                "title": "No Goal Applied",
                "message": "No goal is currently applied, so goal-based variance is not being evaluated."
            })

        if recurrence_available:
            if retention_level == "Strong":
                insights.append({
                    "category": "Retention",
                    "icon": "✅",
                    "severity": "positive",
                    "title": "Repeat Rate Is Strong",
                    "message": (
                        f"The repeat rate is currently {repeat_rate:.1f}%, indicating healthy persistency across the selected data."
                    )
                })
            elif retention_level == "Watch":
                insights.append({
                    "category": "Retention",
                    "icon": "✅",
                    "severity": "warning",
                    "title": "Repeat Rate Should Be Watched",
                    "message": (
                        f"The repeat rate is currently {repeat_rate:.1f}%. This is not critical, but it may deserve monitoring."
                    )
                })
                action_items.append(
                    "Look at repeat entities by dimension to identify where retention is softening."
                )
            else:
                insights.append({
                    "category": "Retention",
                    "icon": "✅",
                    "severity": "risk",
                    "title": "Retention Risk Detected",
                    "message": (
                        f"The repeat rate is currently {repeat_rate:.1f}%, which may indicate elevated attrition risk."
                    )
                })
                action_items.append(
                    "Prioritize reviewing entities that were active in the prior year but are not appearing in the current year."
                )

        if concentration_level == "High":
            insights.append({
                "category": "Risk",
                "icon": "⚠️",
                "severity": "risk",
                "title": "High Concentration Risk",
                "message": (
                    f"The HHI concentration index is {hhi_index:,.0f}, which suggests elevated concentration exposure. "
                    f"{anomaly_count} concentration outlier(s) were detected."
                )
            })
            action_items.append(
                "Review the largest entities and determine whether the selected scope is overly dependent on a small number of high-value relationships."
            )
        elif concentration_level == "Moderate":
            insights.append({
                "category": "Risk",
                "icon": "⚠️",
                "severity": "warning",
                "title": "Moderate Concentration Risk",
                "message": (
                    f"The HHI concentration index is {hhi_index:,.0f}, suggesting moderate concentration. "
                    f"This is manageable, but still worth monitoring."
                )
            })
        else:
            insights.append({
                "category": "Risk",
                "icon": "⚠️",
                "severity": "positive",
                "title": "Concentration Appears Controlled",
                "message": (
                    f"The HHI concentration index is {hhi_index:,.0f}, suggesting the selected data is not overly concentrated."
                )
            })

        if recurrence_available and total_split_value > 0:
            if new_value_share >= 35:
                insights.append({
                    "category": "Mix",
                    "icon": "🧭",
                    "severity": "positive",
                    "title": "New Entity Contribution Is Strong",
                    "message": (
                        f"New entities represent approximately {new_value_share:.1f}% of {metric_label}, "
                        f"indicating strong contribution from new activity."
                    )
                })
            elif new_value_share >= 15:
                insights.append({
                    "category": "Mix",
                    "icon": "🧭",
                    "severity": "neutral",
                    "title": "Mix Is Repeat-Led With Meaningful New Activity",
                    "message": (
                        f"Repeat entities represent approximately {repeat_value_share:.1f}% of {metric_label}, "
                        f"while new entities contribute {new_value_share:.1f}%."
                    )
                })
            else:
                insights.append({
                    "category": "Mix",
                    "icon": "🧭",
                    "severity": "warning",
                    "title": "Heavily Repeat-Dependent",
                    "message": (
                        f"New entities represent only {new_value_share:.1f}% of {metric_label}. "
                        f"The selected data appears highly dependent on repeat activity."
                    )
                })
                action_items.append(
                    "Review whether new-entity activity is sufficient to offset future attrition."
                )
        elif recurrence_available and total_split_count > 0:
            insights.append({
                "category": "Mix",
                "icon": "🧭",
                "severity": "neutral",
                "title": "Mix Available By Entity Count",
                "message": (
                    f"New entities represent approximately {new_count_share:.1f}% of records, while repeat entities represent "
                    f"{repeat_count_share:.1f}%."
                )
            })

        if top_dimension_name:
            insights.append({
                "category": "Opportunity",
                "icon": "🎯",
                "severity": "positive" if top_dimension_share >= 20 else "neutral",
                "title": "Largest Segment Opportunity",
                "message": (
                    f"{top_dimension_name} is the largest visible group in the selected scope, representing approximately "
                    f"{top_dimension_share:.1f}% of measured volume. This may be useful for deeper opportunity review."
                )
            })

            if top_dimension_share >= 35:
                action_items.append(
                    f"Evaluate whether {top_dimension_name} concentration is strategic strength or a dependency risk."
                )

        if pareto_ratio <= 10:
            insights.append({
                "category": "Risk",
                "icon": "⚠️",
                "severity": "risk",
                "title": "Pareto Dependency Is Elevated",
                "message": (
                    f"Approximately {pareto_ratio:.1f}% of entities appear to drive 80% of selected {metric_label}, "
                    f"which suggests a concentrated dependency profile."
                )
            })
        elif pareto_ratio <= 25:
            insights.append({
                "category": "Risk",
                "icon": "⚠️",
                "severity": "warning",
                "title": "Pareto Distribution Is Moderately Concentrated",
                "message": (
                    f"Approximately {pareto_ratio:.1f}% of entities appear to drive 80% of selected {metric_label}."
                )
            })

        score = 100.0

        if growth_level == "Declining":
            score -= 20
        elif growth_level == "Flat":
            score -= 8

        if recurrence_available:
            if retention_level == "Watch":
                score -= 10
            elif retention_level == "At Risk":
                score -= 25

        if concentration_level == "Moderate":
            score -= 10
        elif concentration_level == "High":
            score -= 22

        if goal_status == "Projected Below Goal":
            score -= 15

        if confidence_label == "Low":
            score -= 8

        score = max(0.0, min(100.0, score))

        if score >= 85:
            health_label = "Excellent"
            health_status = "positive"
        elif score >= 70:
            health_label = "Healthy"
            health_status = "positive"
        elif score >= 55:
            health_label = "Watch"
            health_status = "warning"
        else:
            health_label = "At Risk"
            health_status = "risk"

        overview_parts = [
            f"The selected data is currently rated {health_label} with a health score of {score:.1f}/100."
        ]

        if projected_year_end > 0:
            overview_parts.append(
                f"The forecasted year-end position is approximately {projected_year_end:,.0f} in {metric_label}."
            )

        if recurrence_available:
            overview_parts.append(
                f"Repeat rate is {repeat_rate:.1f}% and concentration risk is classified as {concentration_level.lower()}."
            )
        else:
            overview_parts.append(
                f"Concentration risk is classified as {concentration_level.lower()}."
            )

        if goal_status != "No Goal Set":
            overview_parts.append(f"Goal status is currently {goal_status.lower()}.")

        executive_summary = " ".join(overview_parts)

        if not action_items:
            action_items.append(
                "Continue monitoring forecast, concentration, and mix as additional monthly data becomes available."
            )

        return {
            "portfolio_health_score": float(score),
            "portfolio_health_label": health_label,
            "portfolio_health_status": health_status,
            "executive_summary": executive_summary,
            "insights": insights,
            "recommended_actions": action_items[:5]
        }

    # ----------------------------------------------------------------------- #
    # Main pipeline
    # ----------------------------------------------------------------------- #

    def run_analysis(
        self,
        mapping: dict,
        dimension_filters: dict = None,
        projection_target: str = "value",
        primary_dimension: str = None,
        start_date: str = None,
        end_date: str = None,
        include_future_dates: bool = False,
        goal_value: float = 0,
        goals: list = None,
        forecast_horizon_months: int = 24,
        entity_view: str = "all"
    ) -> dict:
        """
        Run the full analysis pipeline on any tabular file: filter/scope the raw data per the
        request, compute KPIs, a repeat-rate/concentration profile, a seasonality-aware
        forecast, multi-goal pacing, and rule-based insights.

        `mapping` = {metric_column, timeline_column, entity_column, dimension_columns}.
        `dimension_filters` = {column_name: [allowed values]} — any configured dimension
        column can be filtered this way; an absent or empty list means "no filter" for that
        column.

        Raises ValueError if the required metric/timeline columns are missing or not present
        in the uploaded file (e.g. a mapping left over from a previously loaded file).
        """
        if not isinstance(mapping, dict):
            raise ValueError("A column mapping is required to run analysis.")

        metric_col = mapping.get("metric_column")
        time_col = mapping.get("timeline_column")
        entity_col = mapping.get("entity_column")
        dimension_cols = list(mapping.get("dimension_columns") or [])

        if not metric_col or not time_col:
            raise ValueError("A Metric column and a Timeline column are required fields.")

        if metric_col not in self.df.columns or time_col not in self.df.columns:
            raise ValueError(
                "The mapped Metric or Timeline column was not found in the uploaded file. "
                "This can happen if the mapping is left over from a different file — "
                "re-upload the file and confirm the schema mapping again."
            )

        # Optional mappings may reference a column that no longer exists (e.g. a stale
        # mapping or a saved view applied to a different file). Drop those silently rather
        # than raising, since the analysis can still run without them.
        entity_col = entity_col if entity_col in self.df.columns else None
        dimension_cols = [c for c in dimension_cols if c in self.df.columns]

        if primary_dimension and primary_dimension not in dimension_cols:
            primary_dimension = None
        if not primary_dimension and dimension_cols:
            primary_dimension = dimension_cols[0]

        cols_to_keep = list(dict.fromkeys(
            [metric_col, time_col] + dimension_cols + ([entity_col] if entity_col else [])
        ))

        working_df = self.df[cols_to_keep].copy()
        working_df[time_col] = pd.to_datetime(working_df[time_col], errors="coerce")
        working_df[metric_col] = pd.to_numeric(working_df[metric_col], errors="coerce").fillna(0)
        working_df = working_df.dropna(subset=[time_col])

        for col in dimension_cols:
            working_df[col] = working_df[col].apply(self._normalize_categorical_value)

        if entity_col:
            working_df[entity_col] = working_df[entity_col].fillna("Unknown")

        working_df = working_df[working_df[time_col] >= ANALYTICAL_BASELINE]

        future_records_removed = 0
        future_metric_amount = 0.0

        if not include_future_dates:
            today = pd.Timestamp(datetime.now().date())
            future_mask = working_df[time_col] > today
            future_records_removed = int(future_mask.sum())
            future_metric_amount = float(working_df.loc[future_mask, metric_col].sum())
            working_df = working_df[~future_mask]

        effective_start = None
        effective_end = None

        if start_date:
            start_dt = pd.to_datetime(start_date, errors="coerce")

            if pd.notna(start_dt):
                working_df = working_df[working_df[time_col] >= start_dt]
                effective_start = start_dt

        if end_date:
            end_dt = pd.to_datetime(end_date, errors="coerce")

            if pd.notna(end_dt):
                end_capped = end_dt + pd.Timedelta(hours=23, minutes=59, seconds=59)
                working_df = working_df[working_df[time_col] <= end_capped]
                effective_end = end_dt

        if effective_start is None and not working_df.empty:
            effective_start = working_df[time_col].min()

        if effective_end is None and not working_df.empty:
            effective_end = working_df[time_col].max()

        dimension_filters = dimension_filters or {}
        filters_applied = 0

        for col, allowed_values in dimension_filters.items():
            if col not in dimension_cols or not allowed_values:
                continue

            normalized_allowed = [self._normalize_categorical_value(v) for v in allowed_values]
            working_df = working_df[working_df[col].isin(normalized_allowed)]
            filters_applied += 1

        working_df = self._classify_entity_recurrence(working_df, entity_col, time_col)

        new_df = working_df[working_df["EntityStatus"] == "New"]
        repeat_df = working_df[working_df["EntityStatus"] == "Repeat"]
        recurrence_classified = bool(entity_col)

        entity_split = {
            "new_value": float(new_df[metric_col].sum()) if not new_df.empty else 0.0,
            "repeat_value": float(repeat_df[metric_col].sum()) if not repeat_df.empty else 0.0,
            "new_count": int(new_df[entity_col].nunique()) if entity_col and not new_df.empty else int(len(new_df)),
            "repeat_count": int(repeat_df[entity_col].nunique()) if entity_col and not repeat_df.empty else int(len(repeat_df)),
            "classification_method": "derived" if recurrence_classified else "unavailable"
        }

        if entity_view == "new":
            working_df = working_df[working_df["EntityStatus"] == "New"]
        elif entity_view == "repeat":
            working_df = working_df[working_df["EntityStatus"] == "Repeat"]

        effective_goals = goals if goals else (
            [{
                "id": "primary",
                "label": "Overall Goal",
                "period": "annual",
                "scope_type": "overall",
                "scope_column": None,
                "scope_value": None,
                "target_value": goal_value
            }] if goal_value and goal_value > 0 else []
        )

        metric_label = metric_col.replace("_", " ") if projection_target == "value" else (
            f"{(entity_col or 'record').replace('_', ' ')} count"
        )

        if working_df.empty:
            empty_goal_progress = self.compute_goals(
                working_df, metric_col, entity_col, time_col, projection_target, effective_goals, effective_end
            )

            return {
                "kpis": {
                    "total_metric_value": 0,
                    "unique_entity_count": 0,
                    "avg_entity_value": 0,
                    "repeat_rate": 0,
                    "hhi_index": 0,
                    "pareto_ratio": 0
                },
                "historical_timeline": {
                    "labels": [],
                    "values": [],
                    "rolling_avg": [],
                    "mom_growth": [],
                    "new_values": [],
                    "repeat_values": []
                },
                "dimension_distribution": {},
                "primary_dimension": primary_dimension,
                "seasonality": {},
                "projections": [],
                "forecast_outlook": self._empty_forecast_outlook(projection_target, metric_label),
                "long_range_forecast": self._empty_long_range_forecast(projection_target, forecast_horizon_months, metric_label),
                "ai_insights": self._empty_ai_insights(),
                "anomalies": [],
                "trend_anomalies": [],
                "goal_progress": empty_goal_progress,
                "entity_split": entity_split,
                "diagnostics": {
                    "future_records_removed": future_records_removed,
                    "future_metric_amount": future_metric_amount,
                    "include_future_dates": include_future_dates,
                    "filters_applied": filters_applied,
                    "recurrence_classified": recurrence_classified,
                    "entity_view": entity_view
                }
            }

        total_metric_value = float(working_df[metric_col].sum())
        unique_entity_count = int(working_df[entity_col].nunique()) if entity_col else int(len(working_df))
        avg_entity_value = float(total_metric_value / unique_entity_count) if unique_entity_count > 0 else 0.0

        hhi_index = 0.0

        if primary_dimension and total_metric_value > 0:
            shares = working_df.groupby(primary_dimension)[metric_col].sum()
            hhi_index = float(sum([(v / total_metric_value * 100) ** 2 for v in shares]))

        working_df["Year"] = working_df[time_col].dt.year
        repeat_rate = 100.0
        years_present = sorted(working_df["Year"].unique())

        if len(years_present) >= 2 and entity_col:
            prev_year_entities = set(working_df[working_df["Year"] == years_present[-2]][entity_col].unique())
            curr_year_entities = set(working_df[working_df["Year"] == years_present[-1]][entity_col].unique())

            if prev_year_entities:
                retained = prev_year_entities.intersection(curr_year_entities)
                repeat_rate = float(len(retained) / len(prev_year_entities) * 100)

        pareto_ratio = 20.0

        if entity_col and total_metric_value > 0:
            entity_sums = working_df.groupby(entity_col)[metric_col].sum().sort_values(ascending=False)
            cumulative_sum = entity_sums.cumsum()
            cutoff = total_metric_value * 0.80
            top_entities_count = len(cumulative_sum[cumulative_sum <= cutoff]) + 1
            pareto_ratio = float((top_entities_count / len(entity_sums)) * 100) if len(entity_sums) > 0 else 20.0

        working_df["YearMonth"] = working_df[time_col].dt.to_period("M")
        monthly_groups = working_df.groupby("YearMonth")

        monthly_df = pd.DataFrame({
            "value": monthly_groups[metric_col].sum(),
            "count": monthly_groups[entity_col].nunique() if entity_col else monthly_groups.size()
        }).reset_index()

        monthly_df["YearMonthStr"] = monthly_df["YearMonth"].astype(str)

        target_series = "value" if projection_target == "value" else "count"

        monthly_df["RollingAvg"] = monthly_df[target_series].rolling(window=3, min_periods=1).mean()
        monthly_df["MoM_Growth"] = (
            monthly_df[target_series]
            .pct_change()
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0) * 100
        )

        new_monthly = working_df[working_df["EntityStatus"] == "New"].groupby("YearMonth") if not new_df.empty else None
        repeat_monthly = working_df[working_df["EntityStatus"] == "Repeat"].groupby("YearMonth") if not repeat_df.empty else None

        if projection_target == "value":
            new_series = new_monthly[metric_col].sum() if new_monthly is not None else pd.Series(dtype=float)
            repeat_series = repeat_monthly[metric_col].sum() if repeat_monthly is not None else pd.Series(dtype=float)
        else:
            new_series = (new_monthly[entity_col].nunique() if entity_col else new_monthly.size()) if new_monthly is not None else pd.Series(dtype=float)
            repeat_series = (repeat_monthly[entity_col].nunique() if entity_col else repeat_monthly.size()) if repeat_monthly is not None else pd.Series(dtype=float)

        new_values = []
        repeat_values = []

        for ym in monthly_df["YearMonth"]:
            new_values.append(float(new_series.get(ym, 0)))
            repeat_values.append(float(repeat_series.get(ym, 0)))

        dimension_data = {}

        if primary_dimension:
            dim_metric_col = metric_col if projection_target == "value" else (entity_col if entity_col else metric_col)
            dim_agg = "sum" if projection_target == "value" else "nunique"

            dim_summary = (
                working_df
                .groupby(primary_dimension)[dim_metric_col]
                .agg(dim_agg)
                .sort_values(ascending=False)
                .head(20)
            )

            dimension_data = {str(k): float(v) for k, v in dim_summary.items()}

        working_df["MonthName"] = working_df[time_col].dt.strftime("%B")
        season_metric_col = metric_col if projection_target == "value" else (entity_col if entity_col else metric_col)
        season_agg = "sum" if projection_target == "value" else "nunique"

        season_summary = working_df.groupby("MonthName")[season_metric_col].agg(season_agg)
        seasonality = {k: float(v) for k, v in season_summary.to_dict().items()}

        projections = []

        if len(monthly_df) > 1:
            X = np.arange(len(monthly_df)).reshape(-1, 1)
            y = monthly_df[target_series].values

            model = LinearRegression().fit(X, y)
            future_X = np.arange(len(monthly_df), len(monthly_df) + 12).reshape(-1, 1)
            future_predictions = model.predict(future_X)

            last_date = working_df[time_col].max()

            for i, pred in enumerate(future_predictions):
                next_month = (last_date + pd.DateOffset(months=i + 1)).strftime("%Y-%m")

                projections.append({
                    "period": next_month,
                    "projected_value": max(0.0, float(pred))
                })

        forecast_outlook = self._compute_forecast_outlook(
            monthly_df=monthly_df,
            target_series=target_series,
            projection_target=projection_target,
            metric_label=metric_label
        )

        long_range_forecast = self._compute_long_range_forecast(
            monthly_df=monthly_df,
            target_series=target_series,
            projection_target=projection_target,
            horizon_months=forecast_horizon_months,
            metric_label=metric_label
        )

        goal_progress = self.compute_goals(
            working_df, metric_col, entity_col, time_col, projection_target, effective_goals, effective_end
        )

        primary_goal = goal_progress[0] if goal_progress else None

        anomalies = []

        if entity_col:
            top_entities = working_df.groupby(entity_col)[metric_col].sum().sort_values(ascending=False).head(10)

            for ent_id, ent_vol in top_entities.items():
                if total_metric_value > 0 and (ent_vol / total_metric_value) > 0.03:
                    anomalies.append({
                        "identifier": str(ent_id),
                        "value": float(ent_vol),
                        "reason": (
                            f"High Concentration Outlier "
                            f"({round(ent_vol / total_metric_value * 100, 1)}% of total selected scope)"
                        )
                    })

        trend_anomalies = []
        avg_monthly_scope_total = float(monthly_df[target_series].mean()) if not monthly_df.empty else 0.0

        for col in dimension_cols:
            trend_anomalies.extend(self._detect_trend_anomalies(
                working_df, time_col, metric_col, entity_col, target_series, col, avg_monthly_scope_total
            ))

        trend_anomalies = sorted(trend_anomalies, key=lambda a: abs(a["deviation_pct"]), reverse=True)[:10]

        ai_insights = self._generate_ai_insights(
            total_metric_value=total_metric_value,
            unique_entity_count=unique_entity_count,
            avg_entity_value=avg_entity_value,
            repeat_rate=repeat_rate,
            hhi_index=hhi_index,
            pareto_ratio=pareto_ratio,
            entity_split=entity_split,
            forecast_outlook=forecast_outlook,
            primary_goal=primary_goal,
            dimension_data=dimension_data,
            anomalies=anomalies,
            projection_target=projection_target,
            metric_label=metric_label
        )

        return {
            "kpis": {
                "total_metric_value": total_metric_value,
                "unique_entity_count": unique_entity_count,
                "avg_entity_value": avg_entity_value,
                "repeat_rate": repeat_rate,
                "hhi_index": hhi_index,
                "pareto_ratio": pareto_ratio
            },
            "historical_timeline": {
                "labels": monthly_df["YearMonthStr"].tolist(),
                "values": monthly_df[target_series].map(float).tolist(),
                "rolling_avg": monthly_df["RollingAvg"].map(float).tolist(),
                "mom_growth": monthly_df["MoM_Growth"].map(float).tolist(),
                "new_values": new_values,
                "repeat_values": repeat_values
            },
            "dimension_distribution": dimension_data,
            "primary_dimension": primary_dimension,
            "seasonality": seasonality,
            "projections": projections,
            "forecast_outlook": forecast_outlook,
            "ai_insights": ai_insights,
            "anomalies": anomalies,
            "trend_anomalies": trend_anomalies,
            "long_range_forecast": long_range_forecast,
            "goal_progress": goal_progress,
            "entity_split": entity_split,
            "diagnostics": {
                "future_records_removed": future_records_removed,
                "future_metric_amount": future_metric_amount,
                "include_future_dates": include_future_dates,
                "filters_applied": filters_applied,
                "recurrence_classified": recurrence_classified,
                "entity_view": entity_view
            }
        }

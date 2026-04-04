def mcnemar(
    csv_a: str,
    csv_b: str,
    qid_col: str = "qid",
    flag_col: str = "llm_flag",
    label_a: str = "A",
    label_b: str = "B",
    alpha: float = 0.05,
):
    import pandas as pd
    from statsmodels.stats.contingency_tables import mcnemar

    def _accuracy(x: pd.Series) -> float:
        return 100.0 * x.mean()

    def _delta(clean_acc: float, pert_acc: float) -> float:
        if clean_acc == 0:
            return float("nan")
        return 100.0 * (clean_acc - pert_acc) / clean_acc

    # Read files
    df_a = pd.read_csv(csv_a)
    df_b = pd.read_csv(csv_b)

    # Validate columns
    for name, df in [(label_a, df_a), (label_b, df_b)]:
        if qid_col not in df.columns:
            raise ValueError(f"{name} missing required column: {qid_col}")
        if flag_col not in df.columns:
            raise ValueError(f"{name} missing required column: {flag_col}")

    # Keep only qid + flag, remove duplicate qids
    a = (
        df_a[[qid_col, flag_col]]
        .drop_duplicates(subset=[qid_col])
        .rename(columns={flag_col: "flag_a"})
    )
    b = (
        df_b[[qid_col, flag_col]]
        .drop_duplicates(subset=[qid_col])
        .rename(columns={flag_col: "flag_b"})
    )

    # Inner join on qid
    merged = a.merge(b, on=qid_col, how="inner")

    # Convert to numeric and drop invalid missing rows
    merged["flag_a"] = pd.to_numeric(merged["flag_a"], errors="coerce")
    merged["flag_b"] = pd.to_numeric(merged["flag_b"], errors="coerce")
    merged = merged.dropna(subset=["flag_a", "flag_b"])

    merged["flag_a"] = merged["flag_a"].astype(int)
    merged["flag_b"] = merged["flag_b"].astype(int)

    # Check binary
    invalid = (~merged["flag_a"].isin([0, 1])) | (~merged["flag_b"].isin([0, 1]))
    if invalid.any():
        bad_rows = merged.loc[invalid, [qid_col, "flag_a", "flag_b"]].head(10)
        raise ValueError(
            f"Found non-binary values in {flag_col}. First examples:\n"
            f"{bad_rows.to_string(index=False)}"
        )

    # Accuracy
    acc_a = _accuracy(merged["flag_a"])
    acc_b = _accuracy(merged["flag_b"])
    delta_abs = acc_b - acc_a
    delta_rel = _delta(acc_a, acc_b)  # relative decrease wrt A

    # Contingency table
    ct = pd.crosstab(merged["flag_a"], merged["flag_b"])
    ct = ct.reindex(index=[0, 1], columns=[0, 1], fill_value=0)

    b01 = int(ct.loc[0, 1])  # A=0, B=1  -> help
    b10 = int(ct.loc[1, 0])  # A=1, B=0  -> hurt

    # McNemar
    res_exact = mcnemar(ct, exact=True)
    res_asymp = mcnemar(ct, exact=False, correction=True)

    significance_exact = "significant" if res_exact.pvalue < alpha else "not significant"
    significance_asymp = "significant" if res_asymp.pvalue < alpha else "not significant"

    report = {
        "n_paired": int(len(merged)),
        "accuracy": {
            label_a: float(acc_a),
            label_b: float(acc_b),
        },
        "delta": {
            "absolute_diff_percentage_points": float(delta_abs),
            "relative_decrease_percent_wrt_" + label_a: float(delta_rel),
        },
        "contingency_table_a_rows_b_cols": ct,
        "discordant_01_help": b01,
        "discordant_10_hurt": b10,
        "exact": {
            "p_value": float(res_exact.pvalue),
            "significance": significance_exact,
        },
        "asymptotic_cc": {
            "statistic": float(res_asymp.statistic),
            "p_value": float(res_asymp.pvalue),
            "significance": significance_asymp,
        },
        "notes": (
            f"H0: marginal proportions are equal between {label_a} and {label_b} "
            f"for paired binary outcomes."
        ),
    }

    # Print report
    print("=== McNemar Test on llm_flag (paired by qid) ===")
    print(f"Compared files: {label_a} vs {label_b}")
    print(f"Paired questions used: {report['n_paired']}")

    print("\n--- Accuracy ---")
    print(f"{label_a} accuracy: {acc_a:.2f}%")
    print(f"{label_b} accuracy: {acc_b:.2f}%")
    print(f"Absolute difference ({label_b} - {label_a}): {delta_abs:.2f} percentage points")
    print(f"Relative decrease wrt {label_a}: {delta_rel:.2f}%")

    print("\n--- Contingency table (rows={}, cols={}) ---".format(label_a, label_b))
    print(ct)

    print("\n--- Discordant pairs ---")
    print(f"{label_a}=0, {label_b}=1  (Help): {b01}")
    print(f"{label_a}=1, {label_b}=0  (Hurt): {b10}")

    print("\n--- McNemar test ---")
    print(f"Exact p-value: {res_exact.pvalue:.6g}  => {significance_exact}")
    print(
        f"Asymptotic (with continuity correction): "
        f"chi2={res_asymp.statistic:.6g}, p={res_asymp.pvalue:.6g}  => {significance_asymp}"
    )

    return report
 

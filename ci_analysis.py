import numpy as np
import scipy.stats as st
from statsmodels.stats.proportion import proportion_confint
from analyze_gee import load_trials, run_gee


def t_ci(x):
    x = np.asarray(x, dtype=float)
    m = x.mean()
    lo, hi = st.t.interval(0.95, len(x) - 1, loc=m, scale=st.sem(x))
    return f"{m:.2f}  [{lo:.2f}, {hi:.2f}]  (n={len(x)})"


def run_ci(df):
    df = df.dropna(subset=["confidence"]).copy()

    print("=== Mean confidence per condition, 95% t-interval ===")
    for c in ["A", "B", "C", "D"]:
        print(c, t_ci(df[df["condition"] == c]["confidence"]))

    p = df.pivot_table(index="participant_id", columns="condition", values="confidence")
    clean = p[["A", "B"]].mean(axis=1)
    error = p[["C", "D"]].mean(axis=1)
    print("\n=== Pooled, participant-level ===")
    print("clean (A,B):", t_ci(clean))
    print("error (C,D):", t_ci(error))
    print("CCG (clean - error):", t_ci(clean - error))

    print("\n=== Wilson 95% intervals (check against the paper) ===")
    err = df[df["injected_error"] == True]
    for label, sub in [("ABR overall", err),
                       ("ABR C", err[err.condition == "C"]),
                       ("ABR D", err[err.condition == "D"])]:
        k, n = int(sub["biased"].sum()), len(sub)
        lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
        print(f"{label}: {k}/{n} = {100*k/n:.1f}%  [{100*lo:.1f}, {100*hi:.1f}]")
    for label, sub in [("Detection overall", err),
                       ("Detection C", err[err.condition == "C"]),
                       ("Detection D", err[err.condition == "D"])]:
        k, n = int(sub["noticed_error"].sum()), len(sub)
        lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
        print(f"{label}: {k}/{n} = {100*k/n:.1f}%  [{100*lo:.1f}, {100*hi:.1f}]")


if __name__ == "__main__":
    df = load_trials()
    df["condition"] = df["condition"].str[0]   # A_baseline -> A, etc.

    run_ci(df)

    result = run_gee(df)
    params = result.params
    ci = result.conf_int()
    key = "C(condition)[T.D]"
    print("\nOR (D vs C):", round(np.exp(params[key]), 3),
          "95% CI:", np.round(np.exp(ci.loc[key].values), 3))


# from analyze_gee import load_trials
# df = load_trials()
# print(df["condition"].unique())
# print(df.groupby("condition").size())
# print(df[["condition", "injected_error", "confidence"]].head(8))
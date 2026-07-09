"""Tactical de-risking backtest — the stress classifier finally *acts*.

classify.py predicts whether a >5% drawdown starts within a month. On its own
that's a ROC curve; here it becomes an allocation decision, walk-forward:

  at each monthly rebalance date t
    1. retrain the logistic stress model on everything known by t
       (training labels need a full forward window, so the last usable
        training row ends STRESS_FWD_DAYS before t — build_dataset enforces it)
    2. read P(stress) for day t itself
    3. gate the book: P > STRESS_PROB_GATE -> min-variance weights (defensive),
       otherwise max-Sharpe weights (aggressive), both solved on a trailing
       window ending at t
    4. hold those weights until the next rebalance, paying transaction costs
       on the weight change

Nothing at step t sees a single day past t — the honest walk-forward loop.
Benchmarks: the equal-weight book (what the rest of Sentinel analyses) and an
always-max-Sharpe strategy (same machinery, gate removed) so the classifier's
contribution is isolated from the optimizer's.

Expectation management, stated up front: the classifier's edge is mostly the
volatility channel (vol clusters), so this is close to a vol-gated allocation.
If the gate adds nothing over always-max-Sharpe, the backtest SAYS so — a
negative result reported honestly beats a leaked positive one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.classify import STRESS_FWD_DAYS, build_dataset, build_features, logit_pipeline
from src.models.optimize import max_sharpe, min_variance
from src.models.risk import (TRADING_DAYS, annualized_return, annualized_vol,
                             calmar_ratio, max_drawdown, sharpe_ratio)

# why 21: rebalance monthly — matches the label horizon (the classifier predicts
# the coming month) and keeps turnover/costs realistic for a tactical overlay.
REBALANCE_DAYS = 21

# why 756: ~3 trading years of history before the first trade. Less and the
# first logistic fits are noise; more and the out-of-sample stretch gets short.
MIN_TRAIN_DAYS = 756

# why 0.5: class_weight="balanced" recenters the classifier so 0.5 is the
# natural "more likely stressed than not" line — no tuned magic threshold.
STRESS_PROB_GATE = 0.5

# why 504: ~2 trading years for the optimizer's moments — long enough for a
# stable covariance, short enough to adapt to the current regime.
OPT_LOOKBACK_DAYS = 504

# why 5bps: a realistic all-in cost for liquid US large-caps; charged on the
# sum of absolute weight changes at each rebalance, so switching regimes hurts.
TRANSACTION_COST_BPS = 5


def _stress_probability(history: pd.DataFrame) -> float:
    """P(stress) for the LAST day of `history`, trained only on `history`.

    Falls back to the base rate when the training window contains a single
    class (e.g. an all-calm early sample) — predict_proba needs two classes.
    """
    X, y = build_dataset(history)
    if len(X) < 60 or y.nunique() < 2:
        return float(y.mean()) if len(y) else 0.0
    model = logit_pipeline().fit(X, y)
    x_now = build_features(history).dropna().iloc[[-1]]
    return float(model.predict_proba(x_now)[0, 1])


def walk_forward_backtest(returns: pd.DataFrame,
                          rebalance_days: int = REBALANCE_DAYS,
                          min_train_days: int = MIN_TRAIN_DAYS,
                          gate: float = STRESS_PROB_GATE,
                          lookback: int = OPT_LOOKBACK_DAYS,
                          cost_bps: float = TRANSACTION_COST_BPS,
                          aggressive_book: str = "max_sharpe") -> dict:
    """Run the tactical strategy and its benchmarks over the same dates.

    aggressive_book — what the strategy holds in calm regimes:
      "max_sharpe" (default): the walk-forward tangency portfolio — concentrated
        and estimation-error-fragile, which is exactly where a risk overlay can
        earn its keep.
      "equal_weight": Sentinel's actual 1/N book, for the de-risking variant.

    Returns daily return series for the three books (tactical / ungated
    aggressive / equal-weight), the defensive-regime timeline, a stats table,
    and the rebalance log (date, P(stress), regime).
    """
    n = len(returns)
    eq_w = pd.Series(1.0 / returns.shape[1], index=returns.columns)

    tact_parts: list[pd.Series] = []
    aggr_parts: list[pd.Series] = []
    regime_parts: list[pd.Series] = []
    log_rows: list[dict] = []
    prev_tact_w = prev_aggr_w = None

    for t in range(min_train_days, n - 1, rebalance_days):
        history = returns.iloc[:t + 1]                      # data through day t
        trailing = returns.iloc[max(0, t + 1 - lookback):t + 1]

        p_stress = _stress_probability(history)
        defensive = p_stress > gate

        # the aggressive book is also carried ungated so the benchmark is the
        # identical machinery with the gate removed
        w_aggr = (eq_w if aggressive_book == "equal_weight"
                  else max_sharpe(trailing))
        w_tact = min_variance(trailing) if defensive else w_aggr

        block = returns.iloc[t + 1:min(t + 1 + rebalance_days, n)]
        tact_ret = block @ w_tact.reindex(block.columns)
        aggr_ret = block @ w_aggr.reindex(block.columns)

        # transaction costs on the weight change, charged on the first day held
        if prev_tact_w is not None:
            tact_ret.iloc[0] -= float(np.abs(w_tact - prev_tact_w).sum()) * cost_bps / 1e4
            aggr_ret.iloc[0] -= float(np.abs(w_aggr - prev_aggr_w).sum()) * cost_bps / 1e4
        prev_tact_w, prev_aggr_w = w_tact, w_aggr

        tact_parts.append(tact_ret)
        aggr_parts.append(aggr_ret)
        regime_parts.append(pd.Series(defensive, index=block.index))
        log_rows.append({"date": returns.index[t], "p_stress": round(p_stress, 4),
                         "defensive": bool(defensive)})

    tactical = pd.concat(tact_parts)
    aggressive = pd.concat(aggr_parts)
    equal = (returns @ eq_w).loc[tactical.index]
    regime = pd.concat(regime_parts)

    # when the aggressive book IS equal-weight the ungated benchmark and the
    # 1/N benchmark coincide — the dict key collapses them into one column
    ungated = ("equal_weight" if aggressive_book == "equal_weight"
               else "always_max_sharpe")
    daily = pd.DataFrame({"tactical": tactical,
                          ungated: aggressive,
                          "equal_weight": equal})
    stats = pd.DataFrame({
        "ann_return": annualized_return(daily),
        "ann_vol": annualized_vol(daily),
        "sharpe": sharpe_ratio(daily),
        "max_drawdown": daily.apply(max_drawdown),
        "calmar": calmar_ratio(daily),
    })

    return {
        "daily": daily,
        "equity": (1 + daily).cumprod(),
        "regime": regime,
        "stats": stats,
        "rebalance_log": pd.DataFrame(log_rows).set_index("date"),
        "pct_defensive": round(float(regime.mean()), 4),
        "n_rebalances": len(log_rows),
        "gate": gate,
        "cost_bps": cost_bps,
        "test_start": str(tactical.index[0].date()),
        "test_days": int(len(tactical)),
    }


def compare_overlays(returns: pd.DataFrame, **kwargs) -> dict:
    """Both overlay variants side by side — the full, two-sided finding.

    On the concentrated max-Sharpe book the gate is expected to cut drawdowns;
    on the diversified 1/N book it is expected to add little or even cost
    return (1/N is famously hard to beat — DeMiguel, Garlappi & Uppal 2009).
    Reporting both is the point: it shows WHERE an ML risk overlay earns its
    keep, not just that one configuration looked good.
    """
    ms = walk_forward_backtest(returns, aggressive_book="max_sharpe", **kwargs)
    eq = walk_forward_backtest(returns, aggressive_book="equal_weight", **kwargs)
    daily = pd.concat([
        ms["daily"][["tactical", "always_max_sharpe"]]
            .rename(columns={"tactical": "tactical_max_sharpe"}),
        eq["daily"].rename(columns={"tactical": "tactical_equal_weight"}),
    ], axis=1)
    stats = pd.DataFrame({
        "ann_return": annualized_return(daily),
        "ann_vol": annualized_vol(daily),
        "sharpe": sharpe_ratio(daily),
        "max_drawdown": daily.apply(max_drawdown),
        "calmar": calmar_ratio(daily),
    })
    return {"daily": daily, "equity": (1 + daily).cumprod(), "stats": stats,
            "regime": ms["regime"], "rebalance_log": ms["rebalance_log"],
            "pct_defensive": ms["pct_defensive"],
            "n_rebalances": ms["n_rebalances"], "gate": ms["gate"],
            "cost_bps": ms["cost_bps"], "test_start": ms["test_start"],
            "test_days": ms["test_days"]}


if __name__ == "__main__":
    from src.warehouse.duck import ensure_loaded, returns_wide

    ensure_loaded()
    out = compare_overlays(returns_wide())
    print(f"Walk-forward tactical backtest — {out['test_days']} out-of-sample days "
          f"from {out['test_start']}, {out['n_rebalances']} rebalances, "
          f"defensive {out['pct_defensive']:.0%} of the time "
          f"(gate {out['gate']}, costs {out['cost_bps']}bps)\n")
    pd.set_option("display.width", 200)
    print(out["stats"].round(3))

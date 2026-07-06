// Static showcase: a real dashboard screenshot in a terminal-window card
// (links to the live demo) + a real memo excerpt in a terminal card.

import Image from "next/image";
import { LINKS } from "../app/config";

export default function Showcase() {
  return (
    <div className="grid gap-6 lg:grid-cols-2">
      {/* real dashboard screenshot */}
      <a
        href={LINKS.demo}
        target="_blank"
        rel="noreferrer"
        className="group flex flex-col overflow-hidden rounded-xl border border-line bg-card transition-colors hover:border-mint/40"
      >
        <div className="flex items-center gap-2 border-b border-line px-4 py-2.5">
          <span className="h-2.5 w-2.5 rounded-full bg-line" aria-hidden />
          <span className="h-2.5 w-2.5 rounded-full bg-line" aria-hidden />
          <span className="h-2.5 w-2.5 rounded-full bg-mint/70" aria-hidden />
          <span className="ml-3 font-mono text-xs text-muted">
            sentinel — risk terminal
          </span>
          <span className="ml-auto font-mono text-[10px] text-mint opacity-0 transition-opacity group-hover:opacity-100">
            open live →
          </span>
        </div>
        <Image
          src="/dashboard.png"
          alt="Sentinel Streamlit dashboard — portfolio KPI cards, growth of $1 chart, and rolling volatility"
          width={1600}
          height={1000}
          className="min-h-0 w-full flex-1 object-cover object-left-top"
        />
      </a>

      {/* real memo excerpt */}
      <div className="overflow-hidden rounded-xl border border-line bg-card">
        <div className="flex items-center gap-2 border-b border-line px-4 py-2.5">
          <span className="font-mono text-xs text-mint">$</span>
          <span className="font-mono text-xs text-muted">
            curl localhost:8000/memo — auto-generated
          </span>
        </div>
        <pre className="overflow-x-auto p-5 font-mono text-xs leading-relaxed text-ink/90">
          {`# Sentinel Risk Memo — 2026-07-02

## 2. Market Risk & Factor Exposure
Sharpe 1.01, beta 0.92 vs SPY. Fama-French
alpha +7.5% (t=3.7), R² 0.90 — survives all
six factors. NVDA is 10% of capital, 19% of
portfolio risk.

## 4. Fundamental & Forensic Screens
AAPL ROE 152% — buyback-shrunk equity, not
operating strength. NVDA trips Beneish (M
−1.18): a hypergrowth false positive, not
manipulation.

## 5. Stress Tests & Allocation
2008-style crash: VaR95 1.7% → 4.6%. A
max-Sharpe reweight lifts Sharpe 1.01 → 1.18.

## 6. Recommendation
The binding constraint is the systemic
scenario: sector diversification does not
survive correlated drawdowns...`}
        </pre>
      </div>
    </div>
  );
}

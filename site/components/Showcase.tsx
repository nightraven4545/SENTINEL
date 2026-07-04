// Static showcase: a stylized dashboard vignette + a real memo excerpt in a
// terminal card. Swap the vignette for a real screenshot/GIF once the
// dashboard is deployed (drop it in public/ and replace the left card).

const KPIS = [
  { label: "PORTFOLIO VOL", value: "18.7%" },
  { label: "VAR 95 (DAILY)", value: "1.69%" },
  { label: "MAX DRAWDOWN", value: "−30.0%" },
  { label: "ANOMALIES", value: "14" },
];

// Deterministic pseudo-sparkline (no Math.random — keeps SSR/CSR identical)
const SPARK = Array.from({ length: 60 }, (_, i) => {
  const y = 30 - 12 * Math.sin(i / 6) - 6 * Math.sin(i / 2.3) - i * 0.15;
  return `${i * 8},${y + 24}`;
}).join(" ");

export default function Showcase() {
  return (
    <div className="grid gap-6 lg:grid-cols-2">
      {/* dashboard vignette */}
      <div className="overflow-hidden rounded-xl border border-line bg-card">
        <div className="flex items-center gap-2 border-b border-line px-4 py-2.5">
          <span className="h-2.5 w-2.5 rounded-full bg-line" aria-hidden />
          <span className="h-2.5 w-2.5 rounded-full bg-line" aria-hidden />
          <span className="h-2.5 w-2.5 rounded-full bg-mint/70" aria-hidden />
          <span className="ml-3 font-mono text-xs text-muted">
            sentinel — risk terminal
          </span>
        </div>
        <div className="p-5">
          <div className="grid grid-cols-4 gap-3">
            {KPIS.map((k) => (
              <div key={k.label} className="rounded-lg border border-line bg-bg p-3">
                <div className="font-mono text-[9px] tracking-wider text-muted">
                  {k.label}
                </div>
                <div className="mt-1 font-mono text-lg font-semibold text-ink">
                  {k.value}
                </div>
              </div>
            ))}
          </div>
          <svg
            viewBox="0 0 480 90"
            className="mt-5 w-full"
            role="img"
            aria-label="Portfolio return sparkline with anomaly markers"
          >
            <polyline
              points={SPARK}
              fill="none"
              stroke="var(--color-mint)"
              strokeWidth="1.6"
            />
            {[112, 128, 144, 384].map((x) => (
              <circle key={x} cx={x} cy={58} r="3.5" fill="var(--color-mint)" opacity="0.85" />
            ))}
          </svg>
          <div className="mt-2 font-mono text-[10px] text-muted">
            anomalous days flagged by both models · 2018 → today
          </div>
        </div>
      </div>

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

## 3. Anomalies
Both detectors (IsolationForest + autoencoder)
currently agree on 14 anomalous days. Most
recent: 2025-04-09 (tariff shock).

## 4. Stress Test Results
| scenario                | VaR95 | max DD |
|-------------------------|-------|--------|
| baseline                | 1.69% | -30.0% |
| rate_shock              | 2.26% | -38.8% |
| market_crash_2008_style | 4.56% | -97.2% |

## 5. Recommendation
The binding constraint is the systemic
scenario: diversification across sectors
does not survive correlated drawdowns...`}
        </pre>
      </div>
    </div>
  );
}

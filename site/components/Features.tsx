"use client";

import { motion } from "framer-motion";

// The engine reads a portfolio through two lenses — markets (CFA) and the
// filings behind them (CA) — then an agent narrates both. Each card names a
// real result so the breadth reads as work, not a feature list.
type Feature = { title: string; blurb: string; icon: React.ReactNode };

const MARKET: Feature[] = [
  {
    title: "Risk & VaR",
    blurb:
      "Historical, Gaussian and Cornish-Fisher VaR, Basel expected shortfall, Sharpe/Sortino/Calmar and a Kupiec backtest — the 95% VaR passes out-of-sample (p=0.58).",
    icon: (
      <path
        d="M12 3l7 3v5c0 4.6-3 7.7-7 9-4-1.3-7-4.4-7-9V6z"
        strokeLinejoin="round"
      />
    ),
  },
  {
    title: "Factor Model",
    blurb:
      "A Fama-French 6-factor regression on excess returns: R² 0.90, market beta 0.94, and a factor-adjusted alpha of +7.5% (t=3.7) that survives all six factors.",
    icon: (
      <>
        <path d="M4 8h8M17 8h3M4 16h3M12 16h8" strokeLinecap="round" />
        <circle cx="14.5" cy="8" r="2.2" />
        <circle cx="9.5" cy="16" r="2.2" />
      </>
    ),
  },
  {
    title: "Portfolio Optimization",
    blurb:
      "A Markowitz efficient frontier via long-only SLSQP: max-Sharpe lifts the equal-weight book from 1.01 to 1.18; min-variance cuts vol from 18.7% to 15.5%.",
    icon: (
      <>
        <path d="M3 20c6 .5 8-12 18-15" strokeLinecap="round" />
        <circle cx="15" cy="8" r="2.1" />
      </>
    ),
  },
  {
    title: "Anomaly Detection",
    blurb:
      "IsolationForest + a PyTorch autoencoder independently rediscovered the COVID crash and the 2025 tariff shock — from raw returns alone.",
    icon: (
      <path
        d="M2 12h4l2-7 4 14 3-10 2 3h5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    ),
  },
  {
    title: "Stress Testing",
    blurb:
      "Named macro scenarios replayed over history: a 2008-style crash takes daily VaR from 1.7% to 4.6% and the drawdown to −97%.",
    icon: (
      <path
        d="M4 20V10m6 10V4m6 16v-8m4 8H2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    ),
  },
  {
    title: "Network Risk",
    blurb:
      "A correlation network that recovers sectors with zero labels and ranks systemic importance by eigenvector centrality.",
    icon: (
      <>
        <circle cx="6" cy="6" r="2.4" />
        <circle cx="18" cy="8" r="2.4" />
        <circle cx="12" cy="18" r="2.4" />
        <path d="M8 7l7.6.8M7.2 8l3.6 8M16.8 10.2l-3.6 6" strokeLinecap="round" />
      </>
    ),
  },
];

const FUNDAMENTAL: Feature[] = [
  {
    title: "Fundamentals & DuPont",
    blurb:
      "SEC EDGAR XBRL → liquidity, solvency and profitability ratios with a 3-step DuPont bridge: Apple's 152% ROE is buyback-shrunk equity; Walmart's 21% is 2.5× asset turnover.",
    icon: (
      <path
        d="M4 5h16v14H4zM4 10h16M10 10v9"
        strokeLinejoin="round"
      />
    ),
  },
  {
    title: "Forensic Screens",
    blurb:
      "Altman Z, Piotroski F, Beneish M, Sloan accruals and Benford's Law — distress and earnings-manipulation screens run straight off the filings, with bank/utility exclusions handled honestly.",
    icon: (
      <>
        <circle cx="10" cy="10" r="6" />
        <path d="M14.5 14.5L20 20" strokeLinecap="round" />
      </>
    ),
  },
];

const MEMO: Feature = {
  title: "AI Risk Memo",
  blurb:
    "An agent with tool access to every model above answers questions and writes the analyst memo — market risk, factor exposure, forensic flags, stress and allocation, then a recommendation.",
  icon: (
    <path
      d="M6 3h9l4 4v14H6zM14 3v5h5M9 12h7M9 16h7"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  ),
};

function Card({ f, i }: { f: Feature; i: number }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-60px" }}
      transition={{ duration: 0.5, delay: i * 0.08, ease: "easeOut" }}
      className="rounded-xl border border-line bg-card p-6 transition hover:-translate-y-1 hover:border-mint/50"
    >
      <svg
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.6"
        className="h-8 w-8 text-mint"
        aria-hidden
      >
        {f.icon}
      </svg>
      <h3 className="mt-4 text-base font-semibold">{f.title}</h3>
      <p className="mt-2 text-sm leading-relaxed text-muted">{f.blurb}</p>
    </motion.div>
  );
}

function LensLabel({ tag, name }: { tag: string; name: string }) {
  return (
    <div className="mb-5 flex items-baseline gap-3">
      <span className="rounded border border-mint/30 px-2 py-0.5 font-mono text-[10px] tracking-[0.2em] text-mint">
        {tag}
      </span>
      <span className="font-mono text-xs tracking-[0.2em] text-muted">
        {name}
      </span>
    </div>
  );
}

export default function Features() {
  return (
    <div className="space-y-14">
      <div>
        <LensLabel tag="CFA" name="MARKET-RISK LENS" />
        <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
          {MARKET.map((f, i) => (
            <Card key={f.title} f={f} i={i} />
          ))}
        </div>
      </div>

      <div>
        <LensLabel tag="CA" name="FUNDAMENTAL & FORENSIC LENS" />
        <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
          {FUNDAMENTAL.map((f, i) => (
            <Card key={f.title} f={f} i={i} />
          ))}
        </div>
      </div>

      <div>
        <LensLabel tag="AGENT" name="READS BOTH LENSES" />
        <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
          <Card f={MEMO} i={0} />
        </div>
      </div>
    </div>
  );
}

"use client";

import { motion } from "framer-motion";
import { LINKS } from "../app/config";

const TICKER =
  "VOL 18.7%  ·  VAR95 1.69%  ·  MAXDD −30.0%  ·  ANOMALIES 14  ·  " +
  "AAPL +0.4  MSFT −0.2  NVDA +1.8  JPM +0.1  XOM −0.6  JNJ +0.2  " +
  "PG −0.1  CAT +0.3  NEE −0.4  WMT +0.5  ·  ";

export default function Hero() {
  return (
    <section className="relative flex min-h-[92vh] flex-col items-center justify-center overflow-hidden px-6">
      <div className="grid-bg absolute inset-0" aria-hidden />
      <div
        className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,transparent_0%,var(--color-bg)_78%)]"
        aria-hidden
      />

      <motion.div
        initial={{ opacity: 0, y: 28 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, ease: "easeOut" }}
        className="relative z-10 max-w-3xl text-center"
      >
        <p className="mb-6 font-mono text-xs tracking-[0.35em] text-mint">
          AGENTIC RISK ENGINE
        </p>
        <h1 className="text-6xl font-bold tracking-[0.18em] sm:text-7xl">
          SENTINEL<span className="text-mint">▮</span>
        </h1>
        <p className="mt-6 text-xl text-muted">
          A junior quant risk analyst, automated.
        </p>
        <p className="mx-auto mt-4 max-w-xl text-sm leading-relaxed text-muted">
          Ingests market data, detects anomalies with ML, stress-tests the
          portfolio, maps systemic risk as a network — then an AI agent reads
          it all and writes the risk memo.
        </p>

        <div className="mt-10 flex flex-wrap items-center justify-center gap-4">
          <a
            href={LINKS.demo}
            className="rounded-md bg-mint px-7 py-3 text-sm font-semibold text-bg transition hover:-translate-y-0.5 hover:shadow-[0_8px_30px_rgba(0,229,160,0.25)]"
          >
            Launch Live Demo
          </a>
          <a
            href={LINKS.github}
            className="rounded-md border border-line bg-card px-7 py-3 text-sm font-semibold text-ink transition hover:-translate-y-0.5 hover:border-mint/50"
          >
            View on GitHub
          </a>
        </div>
      </motion.div>

      <div className="absolute bottom-0 left-0 right-0 overflow-hidden border-t border-line py-3">
        <div className="ticker whitespace-nowrap font-mono text-xs text-muted/70">
          {TICKER.repeat(3)}
          {TICKER.repeat(3)}
        </div>
      </div>
    </section>
  );
}

"use client";

import { motion } from "framer-motion";

const FEATURES = [
  {
    title: "Anomaly Detection",
    blurb:
      "IsolationForest + a PyTorch autoencoder independently rediscovered the COVID crash and the 2025 tariff shock — from raw returns alone.",
    icon: (
      <path d="M2 12h4l2-7 4 14 3-10 2 3h5" strokeLinecap="round" strokeLinejoin="round" />
    ),
  },
  {
    title: "Stress Testing",
    blurb:
      "Named macro scenarios replayed over history: a 2008-style crash takes daily VaR from 1.7% to 4.6% and the drawdown to −97%.",
    icon: (
      <path d="M4 20V10m6 10V4m6 16v-8m4 8H2" strokeLinecap="round" strokeLinejoin="round" />
    ),
  },
  {
    title: "Graph Risk",
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
  {
    title: "AI Risk Memo",
    blurb:
      "An agent with tool access to the whole engine answers questions and writes the analyst memo — situation, risks, recommendation.",
    icon: (
      <path
        d="M6 3h9l4 4v14H6zM14 3v5h5M9 12h7M9 16h7"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    ),
  },
];

export default function Features() {
  return (
    <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
      {FEATURES.map((f, i) => (
        <motion.div
          key={f.title}
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-60px" }}
          transition={{ duration: 0.5, delay: i * 0.1, ease: "easeOut" }}
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
      ))}
    </div>
  );
}

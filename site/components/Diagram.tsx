"use client";

import { motion } from "framer-motion";

const NODES = [
  { name: "Data", sub: "yfinance · FRED" },
  { name: "Warehouse", sub: "DuckDB" },
  { name: "Models", sub: "sklearn · PyTorch · NetworkX" },
  { name: "Agent", sub: "Claude + tool use" },
  { name: "API", sub: "FastAPI · Docker" },
  { name: "Dashboard", sub: "Streamlit" },
];

export default function Diagram() {
  return (
    <div className="flex flex-wrap items-center justify-center gap-y-6">
      {NODES.map((node, i) => (
        <div key={node.name} className="flex items-center">
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-60px" }}
            transition={{ duration: 0.45, delay: i * 0.14, ease: "easeOut" }}
            className="rounded-lg border border-line bg-card px-5 py-3 text-center transition hover:-translate-y-1 hover:border-mint/60"
          >
            <div className="text-sm font-semibold">{node.name}</div>
            <div className="mt-1 font-mono text-[10px] text-muted">{node.sub}</div>
          </motion.div>
          {i < NODES.length - 1 && (
            <motion.span
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.14 + 0.25 }}
              className="mx-3 font-mono text-mint"
              aria-hidden
            >
              →
            </motion.span>
          )}
        </div>
      ))}
    </div>
  );
}

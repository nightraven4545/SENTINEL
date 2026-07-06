import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });
const jbmono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-jbmono" });

export const metadata: Metadata = {
  title: "Sentinel — a junior quant risk analyst, automated",
  description:
    "Open-source agentic quant risk engine with two lenses: market risk (VaR, " +
    "stress testing, Fama-French factor model, Markowitz optimization, ML " +
    "anomaly detection, correlation network) and forensic accounting on SEC " +
    "filings (Altman Z, Beneish M, Piotroski F, DuPont) — plus an LLM agent " +
    "that writes the analyst risk memo.",
  keywords: [
    "quantitative finance",
    "risk management",
    "Value at Risk",
    "stress testing",
    "anomaly detection",
    "machine learning",
    "PyTorch autoencoder",
    "correlation network",
    "Fama-French factor model",
    "factor investing",
    "Markowitz optimization",
    "efficient frontier",
    "forensic accounting",
    "Altman Z-score",
    "Beneish M-score",
    "Piotroski F-score",
    "DuPont analysis",
    "fundamental analysis",
    "SEC EDGAR",
    "LLM agent",
    "AI risk analyst",
    "portfolio risk",
  ],
  openGraph: {
    title: "Sentinel — a junior quant risk analyst, automated",
    description:
      "Agentic financial risk engine, two lenses: market risk (VaR, factor " +
      "model, optimization, ML anomalies, stress) and forensic accounting on " +
      "SEC filings — with an AI-written risk memo.",
    type: "website",
    siteName: "Sentinel",
  },
  twitter: {
    card: "summary_large_image",
    title: "Sentinel — a junior quant risk analyst, automated",
    description:
      "Agentic quant risk engine: market-risk + forensic-accounting lenses and an AI-written risk memo.",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${inter.variable} ${jbmono.variable}`}>
      <body className="font-sans antialiased">{children}</body>
    </html>
  );
}

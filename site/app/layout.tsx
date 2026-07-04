import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });
const jbmono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-jbmono" });

export const metadata: Metadata = {
  title: "Sentinel — a junior quant risk analyst, automated",
  description:
    "Open-source agentic quant risk engine: portfolio VaR and stress testing, " +
    "ML anomaly detection (PyTorch autoencoder + IsolationForest), correlation " +
    "network analysis, and an LLM agent that writes the analyst risk memo.",
  keywords: [
    "quantitative finance",
    "risk management",
    "Value at Risk",
    "stress testing",
    "anomaly detection",
    "machine learning",
    "PyTorch autoencoder",
    "correlation network",
    "LLM agent",
    "AI risk analyst",
    "portfolio risk",
  ],
  openGraph: {
    title: "Sentinel — a junior quant risk analyst, automated",
    description:
      "Agentic financial risk engine: VaR, ML anomaly detection, network risk, " +
      "stress testing, and an AI-written risk memo.",
    type: "website",
    siteName: "Sentinel",
  },
  twitter: {
    card: "summary_large_image",
    title: "Sentinel — a junior quant risk analyst, automated",
    description:
      "Agentic quant risk engine with ML anomaly detection and an AI-written risk memo.",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${inter.variable} ${jbmono.variable}`}>
      <body className="font-sans antialiased">{children}</body>
    </html>
  );
}

import FadeIn from "../components/FadeIn";
import Hero from "../components/Hero";
import Diagram from "../components/Diagram";
import Features from "../components/Features";
import Showcase from "../components/Showcase";
import { AUTHOR, LINKS } from "./config";

const STACK = [
  "Python",
  "pandas",
  "PyTorch",
  "scikit-learn",
  "NetworkX",
  "DuckDB",
  "FastAPI",
  "Docker",
  "Anthropic API",
  "Streamlit",
  "Next.js",
];

function Section({
  kicker,
  title,
  children,
}: {
  kicker: string;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="mx-auto max-w-5xl px-6 py-24">
      <FadeIn>
        <p className="font-mono text-xs tracking-[0.3em] text-mint">{kicker}</p>
        <h2 className="mt-3 text-3xl font-bold tracking-tight">{title}</h2>
      </FadeIn>
      <div className="mt-10">{children}</div>
    </section>
  );
}

export default function Page() {
  return (
    <main>
      <Hero />

      <Section kicker="THE PROBLEM" title="Risk reporting is slow, manual, and backwards-looking.">
        <FadeIn delay={0.1}>
          <p className="max-w-2xl leading-relaxed text-muted">
            Every risk desk runs the same loop: pull data, recompute metrics,
            eyeball charts, write the memo — hours of analyst time producing a
            report about yesterday. Sentinel automates the loop end to end:
            metrics, ML anomaly detection, network structure, stress scenarios,
            and the written memo itself, regenerated on demand in minutes.
          </p>
        </FadeIn>
      </Section>

      <Section kicker="HOW IT WORKS" title="One pipeline, from tick data to a written recommendation.">
        <Diagram />
      </Section>

      <Section kicker="WHAT IT DOES" title="Four analytical layers, one decision-oriented story.">
        <Features />
      </Section>

      <Section kicker="SHOWCASE" title="Real outputs, not mockups.">
        <FadeIn delay={0.1}>
          <Showcase />
        </FadeIn>
        <FadeIn delay={0.2} className="mt-8 text-center">
          <a
            href={LINKS.demo}
            className="inline-block rounded-md bg-mint px-7 py-3 text-sm font-semibold text-bg transition hover:-translate-y-0.5 hover:shadow-[0_8px_30px_rgba(0,229,160,0.25)]"
          >
            Explore the Live Dashboard
          </a>
        </FadeIn>
      </Section>

      <Section kicker="TECH STACK" title="Boring tools, deliberately.">
        <FadeIn delay={0.1}>
          <ul className="flex flex-wrap gap-3">
            {STACK.map((tech) => (
              <li
                key={tech}
                className="rounded-full border border-line bg-card px-4 py-1.5 font-mono text-xs text-muted transition hover:border-mint/50 hover:text-ink"
              >
                {tech}
              </li>
            ))}
          </ul>
        </FadeIn>
      </Section>

      <footer className="border-t border-line">
        <div className="mx-auto flex max-w-5xl flex-wrap items-center justify-between gap-4 px-6 py-10">
          <p className="text-sm text-muted">
            Built by <span className="text-ink">{AUTHOR}</span>
          </p>
          <nav className="flex gap-6 font-mono text-xs">
            <a href={LINKS.github} className="text-muted transition hover:text-mint">
              GitHub
            </a>
            <a href={LINKS.linkedin} className="text-muted transition hover:text-mint">
              LinkedIn
            </a>
            <a href={LINKS.demo} className="text-muted transition hover:text-mint">
              Live Demo
            </a>
          </nav>
        </div>
      </footer>
    </main>
  );
}

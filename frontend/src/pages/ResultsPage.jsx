import { useLocation, useNavigate } from 'react-router-dom'
import { ArrowLeft, Info } from 'lucide-react'
import AppHeader from '../components/AppHeader.jsx'
import ProbabilityBar from '../components/ProbabilityBar.jsx'
import ProbabilityChart from '../components/ProbabilityChart.jsx'
import { clamp01, toWholePercent } from '../utils/format.js'

/**
 * Results page — renders the real response returned by the backend
 * (passed via router state after an analysis). Nothing here is
 * fabricated: every value comes from `location.state.result`.
 */
export default function ResultsPage() {
  const navigate = useNavigate()
  const location = useLocation()
  const result = location.state?.result ?? null
  const source = location.state?.source ?? 'Text input'

  if (!result) {
    // Deep link / refresh without analysis data — show guidance, never
    // a fabricated result.
    return (
      <div className="min-h-screen bg-ivory font-sans text-ink antialiased">
        <AppHeader />
        <main className="mx-auto flex max-w-xl flex-col items-center px-5 pt-24 text-center">
          <span className="eyebrow text-burgundy">No report</span>
          <h1 className="mt-5 font-display text-3xl font-semibold tracking-[-0.02em]">
            Nothing to review yet
          </h1>
          <div className="mt-5 h-px w-12 bg-burgundy" aria-hidden="true" />
          <p className="mt-5 text-sm leading-6 text-ink-soft">
            Results appear here after you run an analysis on the Analyze page.
          </p>
          <button
            type="button"
            onClick={() => navigate('/')}
            className="group mt-8 inline-flex items-center gap-2.5 rounded-lg bg-burgundy px-7 py-3.5 text-[13px] font-semibold uppercase tracking-[0.16em] text-ivory transition hover:bg-burgundy-deep focus:outline-none focus:ring-4 focus:ring-burgundy-line"
          >
            <ArrowLeft
              className="h-4 w-4 transition group-hover:-translate-x-1"
              aria-hidden="true"
            />
            Go to Analyze
          </button>
        </main>
      </div>
    )
  }

  const isCredible = result.prediction === 'true_side'
  const verdictColor = isCredible ? 'text-forest' : 'text-burgundy'

  return (
    <div className="min-h-screen bg-ivory font-sans text-ink antialiased">
      <AppHeader />

      <main className="mx-auto w-full max-w-5xl px-5 pb-20 sm:px-8">
        {/* ---------------- Report header ---------------- */}
        <section className="pt-14 text-center sm:pt-16">
          <span className="eyebrow text-burgundy">Analysis report</span>
          <h1 className="mx-auto mt-5 max-w-2xl font-display text-4xl font-semibold leading-[1.05] tracking-[-0.03em] sm:text-5xl">
            Credibility assessment
          </h1>
          <div
            className="mx-auto mt-6 h-px w-12 bg-burgundy"
            aria-hidden="true"
          />
          <p className="mt-6 text-sm text-ink-soft">
            Source: <span className="text-ink">{source}</span>
            <span className="mx-2 text-rule-strong" aria-hidden="true">
              |
            </span>
            Model: <span className="text-ink">{result.model}</span>
          </p>
        </section>

        {/* ---------------- Verdict + confidence ---------------- */}
        <div className="mt-10 grid gap-5 lg:grid-cols-3">
          {/* Prediction result card */}
          <article className="rounded-xl border border-rule bg-paper p-6 shadow-card sm:p-8 lg:col-span-2">
            <div className="flex items-center justify-between gap-4">
              <span className="eyebrow text-ink-faint">Prediction result</span>
              <span
                className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.12em] ${
                  isCredible
                    ? 'border-forest/25 bg-forest/10 text-forest'
                    : 'border-burgundy-line bg-burgundy-tint text-burgundy'
                }`}
              >
                <span
                  className={`h-1.5 w-1.5 rounded-full ${
                    isCredible ? 'bg-forest' : 'bg-burgundy'
                  }`}
                  aria-hidden="true"
                />
                {isCredible ? 'Leans credible' : 'Leans not credible'}
              </span>
            </div>

            <p
              className={`mt-6 font-display text-4xl font-semibold leading-none tracking-[-0.03em] sm:text-5xl ${verdictColor}`}
            >
              {result.credibility_category}
            </p>

            <p className="mt-5 max-w-xl text-sm leading-6 text-ink-soft">
              The model assigned most of its probability mass to the{' '}
              <span className="font-semibold text-ink">
                {result.prediction}
              </span>{' '}
              class, which maps to the category above. Weigh the distribution
              below before drawing a conclusion.
            </p>

            <div className="mt-7 flex flex-wrap items-center gap-x-4 gap-y-2 border-t border-rule pt-5">
              <span className="eyebrow text-ink-faint">Model label</span>
              <code className="rounded-[3px] border border-rule bg-ivory px-2 py-1 font-mono text-xs text-ink">
                {result.prediction}
              </code>
            </div>
          </article>

          {/* Confidence card */}
          <article className="flex flex-col rounded-xl border border-rule bg-paper p-6 shadow-card sm:p-8">
            <span className="eyebrow text-ink-faint">Model confidence</span>
            <p className="mt-5 font-display text-6xl font-semibold leading-none tracking-[-0.04em] tabular-nums">
              {toWholePercent(result.confidence)}
              <span className="text-3xl text-ink-faint">%</span>
            </p>

            <div className="mt-auto pt-8">
              <div className="h-1.5 w-full overflow-hidden rounded-full bg-ivory-deep">
                <div
                  className="h-full rounded-full bg-burgundy transition-[width] duration-700 ease-out"
                  style={{ width: `${clamp01(result.confidence) * 100}%` }}
                />
              </div>
              <p className="mt-3 text-xs leading-5 text-ink-faint">
                Probability the model assigned to{' '}
                <span className="text-ink-soft">{result.prediction}</span>.
              </p>
            </div>
          </article>
        </div>
        {/* ---------------- Probabilities + graph ---------------- */}
        <div className="mt-5 grid gap-5 lg:grid-cols-2">
          {/* Class probability bars */}
          <article className="rounded-xl border border-rule bg-paper p-6 shadow-card sm:p-8">
            <span className="eyebrow text-ink-faint">Class probabilities</span>
            <div className="mt-7 space-y-7">
              <ProbabilityBar
                label="False side"
                value={result.probabilities?.false_side}
                accent={!isCredible}
                caption="Probability the claim belongs to the false class."
              />
              <ProbabilityBar
                label="True side"
                value={result.probabilities?.true_side}
                accent={isCredible}
                caption="Probability the claim belongs to the true class."
              />
            </div>
          </article>

          {/* Minimal graph */}
          <article className="rounded-xl border border-rule bg-paper p-6 shadow-card sm:p-8">
            <span className="eyebrow text-ink-faint">
              Probability distribution
            </span>
            <ProbabilityChart
              falseSide={result.probabilities?.false_side}
              trueSide={result.probabilities?.true_side}
              predicted={result.prediction}
            />
          </article>
        </div>
        {/* ---------------- Analyzed text ---------------- */}
        <article className="mt-5 rounded-xl border border-rule bg-paper p-6 shadow-card sm:p-8">
          <span className="eyebrow text-ink-faint">
            {source.startsWith('File:') ? 'Extracted text' : 'Analyzed text'}
          </span>
          <blockquote className="mt-5 border-l-2 border-burgundy pl-5 font-display text-lg italic leading-8 text-ink">
            {result.text}
          </blockquote>
        </article>

        {/* ---------------- Disclaimer + actions ---------------- */}
        <div className="mt-7 flex flex-col gap-5 border-t border-rule pt-6 sm:flex-row sm:items-start sm:justify-between">
          <p className="flex max-w-xl items-start gap-2 text-xs leading-5 text-ink-faint">
            <Info className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden="true" />
            This output is a model-based estimate from a fine-tuned language
            model, not a definitive fact-check.
          </p>

          <button
            type="button"
            onClick={() => navigate('/')}
            className="group inline-flex shrink-0 items-center justify-center gap-2.5 rounded-lg border border-rule-strong bg-paper px-6 py-3 text-[13px] font-semibold uppercase tracking-[0.14em] text-ink transition hover:border-burgundy hover:text-burgundy focus:outline-none focus:ring-4 focus:ring-burgundy-tint"
          >
            <ArrowLeft
              className="h-4 w-4 transition group-hover:-translate-x-1"
              aria-hidden="true"
            />
            Analyze another
          </button>
        </div>
      </main>
    </div>
  )
}

import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  AlertCircle,
  ArrowRight,
  FileText,
  Loader2,
  ScanSearch,
  Type,
  X,
} from 'lucide-react'
import { isHFMode } from '../api/hf.js'
import { predictFile, predictText } from '../api/predictions.js'
import AppHeader from '../components/AppHeader.jsx'
import FileDropZone from '../components/FileDropZone.jsx'

const HOW_IT_WORKS = [
  {
    step: '01',
    title: 'Provide input',
    text: 'Type a claim or upload a document or screenshot.',
  },
  {
    step: '02',
    title: 'Automatic analysis',
    text: 'Text is extracted and evaluated by a fine-tuned language model.',
  },
  {
    step: '03',
    title: 'Review the result',
    text: 'A credibility label with a confidence score is presented.',
  },
]

/**
 * Analyze page — sends the user's input to the backend
 * (POST /api/predict or POST /api/predict/file) and navigates to the
 * Results page with the real API response. Predictions are never
 * fabricated on the client.
 */
export default function AnalyzePage() {
  const navigate = useNavigate()
  const [inputType, setInputType] = useState('text') // 'text' | 'file'
  const [claimText, setClaimText] = useState('')
  const [selectedFile, setSelectedFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const hasInput =
    inputType === 'text'
      ? claimText.trim().length > 0
      : selectedFile !== null

  async function handleSubmit(event) {
    event.preventDefault()
    if (!hasInput || loading) return

    setLoading(true)
    setError(null)
    try {
      const result =
        inputType === 'text'
          ? await predictText(claimText.trim())
          : await predictFile(selectedFile)

      navigate('/results', {
        state: {
          result,
          source:
            inputType === 'text'
              ? 'Text input'
              : `File: ${selectedFile.name}`,
        },
      })
    } catch (apiError) {
      setError(
        apiError.friendlyMessage ||
          'Something went wrong while contacting the backend.'
      )
    } finally {
      setLoading(false)
    }
  }

  const typeIndicatorLabel =
    inputType === 'text'
      ? `Text input · ${claimText.trim().length} characters`
      : selectedFile
        ? `File · ${selectedFile.name}`
        : 'No file selected'

  return (
    <div className="min-h-screen bg-ivory font-sans text-ink antialiased">
      <AppHeader />

      <main className="mx-auto w-full max-w-5xl px-5 pb-20 sm:px-8">
        {/* ---------------- Hero ---------------- */}
        <section className="pt-14 text-center sm:pt-20">
          <span className="eyebrow text-burgundy">Credibility analysis</span>
          <h1 className="mx-auto mt-5 max-w-3xl font-display text-4xl font-semibold leading-[1.05] tracking-[-0.03em] sm:text-6xl">
            Read between the headlines.
          </h1>
          <div
            className="mx-auto mt-6 h-px w-12 bg-burgundy"
            aria-hidden="true"
          />
          <p className="mx-auto mt-6 max-w-2xl text-[15px] leading-7 text-ink-soft sm:text-base">
            Submit a claim, headline or post. Our fine-tuned language model
            reads it and returns a credibility verdict with an honest
            confidence score — no guesswork, no black box.
          </p>
        </section>
        {/* ---------------- Analyzer card ---------------- */}
        <form
          onSubmit={handleSubmit}
          aria-busy={loading}
          className="mt-12 rounded-xl border border-rule bg-paper shadow-card"
        >
          {/* Card header: title + segmented control */}
          <div className="flex flex-col gap-4 border-b border-rule px-5 py-5 sm:flex-row sm:items-center sm:justify-between sm:px-8">
            <div>
              <span className="eyebrow text-ink-faint">Analysis input</span>
              <h2 className="mt-2 font-display text-xl font-semibold tracking-[-0.01em]">
                What should we check?
              </h2>
            </div>

            <div
              className="inline-flex self-start rounded-lg border border-rule bg-ivory p-1"
              role="group"
              aria-label="Input type"
            >
              <button
                type="button"
                onClick={() => {
                  setInputType('text')
                  setError(null)
                }}
                aria-pressed={inputType === 'text'}
                disabled={loading}
                className={`inline-flex items-center gap-2 rounded-md px-4 py-2 text-xs font-semibold uppercase tracking-[0.12em] transition disabled:cursor-not-allowed disabled:opacity-60 ${
                  inputType === 'text'
                    ? 'bg-paper text-burgundy shadow-sm ring-1 ring-rule'
                    : 'text-ink-faint hover:text-ink'
                }`}
              >
                <Type className="h-3.5 w-3.5" aria-hidden="true" />
                Text
              </button>
              <button
                type="button"
                onClick={() => {
                  setInputType('file')
                  setError(null)
                }}
                aria-pressed={inputType === 'file'}
                disabled={loading}
                className={`inline-flex items-center gap-2 rounded-md px-4 py-2 text-xs font-semibold uppercase tracking-[0.12em] transition disabled:cursor-not-allowed disabled:opacity-60 ${
                  inputType === 'file'
                    ? 'bg-paper text-burgundy shadow-sm ring-1 ring-rule'
                    : 'text-ink-faint hover:text-ink'
                }`}
              >
                <FileText className="h-3.5 w-3.5" aria-hidden="true" />
                File
              </button>
            </div>
          </div>
          {/* Input area */}
          <div className="px-5 py-6 sm:px-8 sm:py-7">
            {inputType === 'text' ? (
              <div>
                <div className="flex items-baseline justify-between gap-2">
                  <label
                    htmlFor="claim-input"
                    className="eyebrow text-ink-faint"
                  >
                    Claim / statement
                  </label>
                  <span className="text-xs tabular-nums text-ink-faint">
                    {claimText.trim().length} characters
                  </span>
                </div>
                <textarea
                  id="claim-input"
                  rows={7}
                  value={claimText}
                  disabled={loading}
                  onChange={(event) => {
                    setClaimText(event.target.value)
                    if (error) setError(null)
                  }}
                  placeholder="Enter the statement you want to evaluate — a news headline, political claim, or social media post…"
                  className="mt-3 w-full resize-y rounded-lg border border-rule-strong bg-ivory px-4 py-3.5 text-[15px] leading-7 text-ink placeholder:text-ink-faint transition focus:border-burgundy focus:outline-none focus:ring-4 focus:ring-burgundy-tint disabled:opacity-60"
                />
                <p className="mt-3 text-xs leading-5 text-ink-faint">
                  One claim per analysis gives the clearest result.
                </p>
              </div>
            ) : (
              <div>
                <span className="eyebrow text-ink-faint">
                  Document or screenshot
                </span>
                <div className="mt-3">
                  <FileDropZone
                    file={selectedFile}
                    onFileChange={setSelectedFile}
                  />
                </div>
                <p className="mt-3 text-xs leading-5 text-ink-faint">
                  {isHFMode()
                    ? 'Text is extracted in your browser before analysis.'
                    : 'Text inside images is read automatically (OCR).'}
                </p>
              </div>
            )}
            {/* API error banner */}
            {error && (
              <div
                role="alert"
                className="mt-6 flex items-start gap-3 rounded-lg border border-burgundy-line bg-burgundy-tint px-4 py-3"
              >
                <AlertCircle
                  className="mt-0.5 h-4 w-4 shrink-0 text-burgundy"
                  aria-hidden="true"
                />
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-semibold text-burgundy-deep">
                    Analysis failed
                  </p>
                  <p className="mt-0.5 break-words text-xs leading-5 text-burgundy">
                    {error}
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => setError(null)}
                  className="rounded p-1 text-burgundy/60 transition hover:bg-paper hover:text-burgundy-deep"
                  aria-label="Dismiss error"
                >
                  <X className="h-4 w-4" aria-hidden="true" />
                </button>
              </div>
            )}

            {/* Input status + Analyze button */}
            <div className="mt-7 flex flex-col gap-4 border-t border-rule pt-6 sm:flex-row sm:items-center sm:justify-between">
              <span
                className={`inline-flex max-w-full items-center gap-2 self-start rounded-full border px-3 py-1.5 text-xs ${
                  hasInput
                    ? 'border-burgundy-line bg-burgundy-tint text-burgundy'
                    : 'border-rule bg-ivory text-ink-faint'
                }`}
              >
                <span
                  className={`h-1.5 w-1.5 shrink-0 rounded-full ${
                    hasInput ? 'bg-burgundy' : 'bg-rule-strong'
                  }`}
                  aria-hidden="true"
                />
                <span className="truncate">{typeIndicatorLabel}</span>
              </span>

              <button
                type="submit"
                disabled={!hasInput || loading}
                className="group inline-flex w-full items-center justify-center gap-2.5 rounded-lg bg-burgundy px-8 py-4 text-[13px] font-semibold uppercase tracking-[0.16em] text-ivory shadow-[0_14px_28px_-18px_rgba(91,21,33,0.9)] transition hover:bg-burgundy-deep focus:outline-none focus:ring-4 focus:ring-burgundy-line disabled:cursor-not-allowed disabled:bg-rule-strong disabled:text-ivory disabled:shadow-none sm:w-auto"
              >
                {loading ? (
                  <>
                    <Loader2
                      className="h-4 w-4 animate-spin"
                      aria-hidden="true"
                    />
                    Analyzing…
                  </>
                ) : (
                  <>
                    <ScanSearch className="h-4 w-4" aria-hidden="true" />
                    Analyze credibility
                    <ArrowRight
                      className="h-4 w-4 transition group-hover:translate-x-1"
                      aria-hidden="true"
                    />
                  </>
                )}
              </button>
            </div>
          </div>
        </form>
        {/* ---------------- How it works ---------------- */}
        <section className="mt-16">
          <div className="flex items-center gap-4">
            <span className="eyebrow text-ink-faint">How it works</span>
            <span className="h-px flex-1 bg-rule" aria-hidden="true" />
          </div>

          <ol className="mt-7 grid gap-8 sm:grid-cols-3 sm:gap-0 sm:divide-x sm:divide-rule">
            {HOW_IT_WORKS.map((item) => (
              <li key={item.step} className="sm:px-7 sm:first:pl-0 sm:last:pr-0">
                <span className="font-display text-2xl font-semibold text-burgundy">
                  {item.step}
                </span>
                <h3 className="mt-3 font-display text-base font-semibold tracking-[-0.01em]">
                  {item.title}
                </h3>
                <p className="mt-1.5 text-sm leading-6 text-ink-soft">
                  {item.text}
                </p>
              </li>
            ))}
          </ol>
        </section>
      </main>

      <footer className="border-t border-rule">
        <div className="mx-auto flex max-w-6xl flex-col items-start justify-between gap-3 px-5 py-8 sm:flex-row sm:items-center sm:px-8">
          <p className="font-display text-sm text-ink-soft">
            CrediLens AI — Information credibility detection project
          </p>
          <p className="eyebrow text-ink-faint">
            Text analysis by a fine-tuned RoBERTa model
          </p>
        </div>
      </footer>
    </div>
  )
}

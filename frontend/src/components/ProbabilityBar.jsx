import { clamp01, toPercent } from '../utils/format.js'

/**
 * Thin, editorial probability bar for one model class.
 *
 * Props:
 *   label:     display label (e.g. "False side")
 *   value:     raw probability from the API (0.0 - 1.0)
 *   accent:    true → burgundy fill (the class the model leaned toward)
 *   caption:   optional small note shown under the bar
 */
export default function ProbabilityBar({ label, value, accent = false, caption }) {
  const ratio = clamp01(value)

  return (
    <div>
      <div className="flex items-baseline justify-between gap-3">
        <span className="eyebrow text-ink-faint">{label}</span>
        <span className="font-display text-lg font-semibold tabular-nums text-ink">
          {toPercent(value)}
        </span>
      </div>

      {/* Track */}
      <div className="mt-2.5 h-1.5 w-full overflow-hidden rounded-full bg-ivory-deep">
        <div
          className={`h-full rounded-full transition-[width] duration-700 ease-out ${
            accent ? 'bg-burgundy' : 'bg-rule-strong'
          }`}
          style={{ width: `${ratio * 100}%` }}
        />
      </div>

      {caption ? (
        <p className="mt-2 text-xs leading-5 text-ink-faint">{caption}</p>
      ) : null}
    </div>
  )
}

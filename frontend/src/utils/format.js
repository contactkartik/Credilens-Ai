/**
 * Small formatting helpers shared by the results UI.
 *
 * Every value passed in here comes straight from the backend response —
 * these helpers only format, they never invent numbers.
 */

/**
 * Format a 0..1 probability from the API as a percentage string.
 * Returns an em dash for missing/invalid values instead of a fake number.
 */
export function toPercent(value, digits = 1) {
  const number = Number(value)
  if (!Number.isFinite(number)) return '—'
  return `${(number * 100).toFixed(digits)}%`
}

/** Whole-number percentage (used for the large confidence display). */
export function toWholePercent(value) {
  const number = Number(value)
  if (!Number.isFinite(number)) return '—'
  return (number * 100).toFixed(0)
}

/** Clamp any API value into the 0..1 range (safe for bar widths). */
export function clamp01(value) {
  const number = Number(value)
  if (!Number.isFinite(number)) return 0
  return Math.min(1, Math.max(0, number))
}

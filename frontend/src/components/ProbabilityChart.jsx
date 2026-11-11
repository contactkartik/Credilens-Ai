import { clamp01, toPercent } from '../utils/format.js'

/**
 * Minimal column chart of the class probabilities — drawn with plain SVG,
 * no chart library. It is intentionally sparse (hairline grid, two columns)
 * so it reads like a graphic in a print magazine rather than a dashboard.
 *
 * All values come from the API response; nothing is invented here.
 *
 * Props:
 *   falseSide: probability for class `false_side` (0.0 - 1.0)
 *   trueSide:  probability for class `true_side`  (0.0 - 1.0)
 *   predicted: raw model label ('true_side' | 'false_side')
 */
export default function ProbabilityChart({ falseSide, trueSide, predicted }) {
  const falseRatio = clamp01(falseSide)
  const trueRatio = clamp01(trueSide)

  // Geometry (viewBox units)
  const PLOT_LEFT = 44
  const PLOT_RIGHT = 332
  const PLOT_TOP = 30 // y of the 100% gridline
  const BASELINE = 134 // y of the 0% gridline
  const CENTER = (PLOT_LEFT + PLOT_RIGHT) / 2
  const BAR_WIDTH = 62
  const OFFSET = 56 // distance of each column from the plot centre

  const plotHeight = BASELINE - PLOT_TOP
  const falseHeight = falseRatio * plotHeight
  const trueHeight = trueRatio * plotHeight

  const falseX = CENTER - OFFSET - BAR_WIDTH / 2
  const trueX = CENTER + OFFSET - BAR_WIDTH / 2

  const gridlines = [
    { ratio: 1, label: '100' },
    { ratio: 0.5, label: '50' },
    { ratio: 0, label: '0' },
  ]

  return (
    <figure className="mt-5">
      <svg
        role="img"
        aria-label={`Probability distribution: false side ${toPercent(falseSide)}, true side ${toPercent(trueSide)}`}
        viewBox="0 0 340 174"
        className="h-auto w-full"
        preserveAspectRatio="xMidYMid meet"
      >
        {/* Hairline gridlines + axis labels */}
        {gridlines.map((line) => {
          const y = BASELINE - line.ratio * plotHeight
          return (
            <g key={line.label}>
              <line
                x1={PLOT_LEFT}
                x2={PLOT_RIGHT}
                y1={y}
                y2={y}
                stroke="#e7e0d3"
                strokeWidth="1"
                strokeDasharray={line.ratio === 0 ? undefined : '2 5'}
              />
              <text
                x={PLOT_LEFT - 10}
                y={y + 3.5}
                textAnchor="end"
                className="fill-ink-faint text-[9px] font-medium"
              >
                {line.label}
              </text>
            </g>
          )
        })}

        {/* Columns — the predicted class is filled burgundy */}
        <rect
          x={falseX}
          y={BASELINE - falseHeight}
          width={BAR_WIDTH}
          height={falseHeight}
          rx="2"
          fill={predicted === 'false_side' ? '#7a1e2b' : '#d5cbbb'}
        />
        <rect
          x={trueX}
          y={BASELINE - trueHeight}
          width={BAR_WIDTH}
          height={trueHeight}
          rx="2"
          fill={predicted === 'true_side' ? '#7a1e2b' : '#d5cbbb'}
        />

        {/* Value labels above each column */}
        <text
          x={falseX + BAR_WIDTH / 2}
          y={BASELINE - falseHeight - 9}
          textAnchor="middle"
          className="fill-ink font-display text-[12px] font-semibold"
        >
          {toPercent(falseSide)}
        </text>
        <text
          x={trueX + BAR_WIDTH / 2}
          y={BASELINE - trueHeight - 9}
          textAnchor="middle"
          className="fill-ink font-display text-[12px] font-semibold"
        >
          {toPercent(trueSide)}
        </text>

        {/* Category labels under the baseline */}
        <text
          x={falseX + BAR_WIDTH / 2}
          y={BASELINE + 18}
          textAnchor="middle"
          letterSpacing="1.4"
          className="fill-ink-faint text-[9px] font-semibold uppercase"
        >
          False side
        </text>
        <text
          x={trueX + BAR_WIDTH / 2}
          y={BASELINE + 18}
          textAnchor="middle"
          letterSpacing="1.4"
          className="fill-ink-faint text-[9px] font-semibold uppercase"
        >
          True side
        </text>
      </svg>

      <figcaption className="mt-4 border-t border-rule pt-3 text-xs leading-5 text-ink-faint">
        Column height = probability assigned to each class. The burgundy column
        is the class the model selected.
      </figcaption>
    </figure>
  )
}

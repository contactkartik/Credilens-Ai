import { Link } from 'react-router-dom'
import { ScanSearch } from 'lucide-react'

/**
 * Editorial masthead for CrediLens AI.
 *
 * Sticky, hairline-ruled header with a burgundy monogram, serif wordmark
 * and small-caps meta on the right — modelled on a newspaper masthead
 * rather than a typical app navbar.
 */
export default function AppHeader() {
  return (
    <header className="sticky top-0 z-30 border-b border-rule bg-ivory/90 backdrop-blur-md">
      <div className="mx-auto flex h-16 w-full max-w-6xl items-center justify-between gap-4 px-5 sm:px-8">
        {/* Brand */}
        <Link
          to="/"
          className="group flex items-center gap-3 focus:outline-none focus-visible:ring-2 focus-visible:ring-burgundy-line"
        >
          <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-[3px] bg-burgundy font-display text-lg font-semibold text-ivory transition group-hover:bg-burgundy-deep">
            C
          </span>
          <span className="flex flex-col">
            <span className="font-display text-[19px] font-semibold leading-none tracking-[-0.01em] text-ink">
              CrediLens
            </span>
            <span className="eyebrow mt-1.5 text-[9px] text-ink-faint">
              AI credibility desk
            </span>
          </span>
        </Link>

        {/* Masthead meta */}
        <div className="flex items-center gap-3">
          <span className="hidden h-4 w-px bg-rule-strong sm:block" />
          <span className="eyebrow hidden items-center gap-2 text-ink-faint sm:flex">
            <span className="h-1.5 w-1.5 rounded-full bg-burgundy" aria-hidden="true" />
            Explainable NLP
          </span>
          <span
            className="inline-flex h-8 w-8 items-center justify-center rounded-[3px] border border-rule bg-paper text-burgundy"
            title="Fine-tuned RoBERTa model"
          >
            <ScanSearch className="h-4 w-4" aria-hidden="true" />
          </span>
        </div>
      </div>
    </header>
  )
}

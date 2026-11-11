import { useRef, useState } from 'react'
import { FileText, Upload, X } from 'lucide-react'

const ACCEPTED_FORMATS = '.txt,.pdf,.docx,.png,.jpg,.jpeg'

/**
 * File upload card (drag & drop) styled as part of the editorial system:
 * ivory inset, hairline dashed border, burgundy affordances.
 *
 * UI only — no upload request is made here.
 *
 * Props:
 *   file:         currently selected File (or null)
 *   onFileChange: callback when a file is chosen/dropped
 */
export default function FileDropZone({ file, onFileChange }) {
  const inputRef = useRef(null)
  const [isDragging, setIsDragging] = useState(false)

  function handleFiles(fileList) {
    const next = fileList && fileList[0]
    if (next) onFileChange(next)
  }

  function handleDrop(event) {
    event.preventDefault()
    setIsDragging(false)
    handleFiles(event.dataTransfer.files)
  }

  function handleKeyDown(event) {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      inputRef.current?.click()
    }
  }

  if (file) {
    return (
      <div className="flex items-center justify-between gap-3 rounded-lg border border-rule bg-ivory px-4 py-3.5">
        <div className="flex min-w-0 items-center gap-3">
          <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-[3px] border border-rule bg-paper text-burgundy">
            <FileText className="h-4.5 w-4.5" aria-hidden="true" />
          </span>
          <div className="min-w-0">
            <p className="truncate text-sm font-medium text-ink">{file.name}</p>
            <p className="eyebrow mt-1.5 text-[10px] text-ink-faint">
              {(file.size / 1024).toFixed(1)} KB &middot; ready to analyze
            </p>
          </div>
        </div>
        <button
          type="button"
          onClick={() => onFileChange(null)}
          className="rounded-[3px] p-1.5 text-ink-faint transition hover:bg-paper hover:text-burgundy focus:outline-none focus-visible:ring-2 focus-visible:ring-burgundy-line"
          aria-label="Remove selected file"
        >
          <X className="h-4 w-4" aria-hidden="true" />
        </button>
      </div>
    )
  }

  return (
    <div
      role="button"
      tabIndex={0}
      aria-label="Upload a document or screenshot"
      onClick={() => inputRef.current?.click()}
      onKeyDown={handleKeyDown}
      onDragOver={(event) => {
        event.preventDefault()
        setIsDragging(true)
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      className={`flex cursor-pointer flex-col items-center justify-center rounded-lg border border-dashed px-6 py-10 text-center transition focus:outline-none focus-visible:ring-2 focus-visible:ring-burgundy-line ${
        isDragging
          ? 'border-burgundy bg-burgundy-tint'
          : 'border-rule-strong bg-ivory hover:border-burgundy hover:bg-burgundy-tint/50'
      }`}
    >
      <span className="flex h-11 w-11 items-center justify-center rounded-full border border-rule bg-paper text-burgundy">
        <Upload className="h-5 w-5" aria-hidden="true" />
      </span>
      <p className="mt-4 text-sm text-ink-soft">
        Drag &amp; drop a file here, or{' '}
        <span className="font-medium text-burgundy underline decoration-burgundy-line underline-offset-4">
          browse
        </span>
      </p>
      <p className="eyebrow mt-2.5 text-[10px] text-ink-faint">
        TXT &middot; PDF &middot; DOCX &middot; PNG / JPG (screenshots)
      </p>

      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED_FORMATS}
        className="hidden"
        onChange={(event) => {
          handleFiles(event.target.files)
          event.target.value = ''
        }}
      />
    </div>
  )
}

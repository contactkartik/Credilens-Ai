/**
 * Client-side text extraction for the hosted (HF Inference) deployment,
 * where there is no FastAPI backend to do it.
 *
 * Supported: .txt, .pdf, .docx
 * Images are rejected — OCR needs the backend's PaddleOCR pipeline.
 */

const MAX_CHARS = 20000 // keep payloads reasonable for the inference API

export const SUPPORTED_EXTENSIONS = ['.txt', '.pdf', '.docx']

export function fileExtension(file) {
  const name = file?.name || ''
  const dot = name.lastIndexOf('.')
  return dot === -1 ? '' : name.slice(dot).toLowerCase()
}

function truncate(text) {
  const trimmed = (text || '').trim()
  return trimmed.length > MAX_CHARS ? trimmed.slice(0, MAX_CHARS) : trimmed
}

async function extractTxt(file) {
  return truncate(await file.text())
}

async function extractPdf(file) {
  const pdfjsLib = await import('pdfjs-dist')
  pdfjsLib.GlobalWorkerOptions.workerSrc = new URL(
    'pdfjs-dist/build/pdf.worker.min.mjs',
    import.meta.url
  ).toString()

  const data = await file.arrayBuffer()
  const doc = await pdfjsLib.getDocument({ data }).promise

  let text = ''
  for (let pageNum = 1; pageNum <= doc.numPages; pageNum++) {
    const page = await doc.getPage(pageNum)
    const content = await page.getTextContent()
    text += content.items.map((item) => item.str).join(' ') + '\n'
  }
  return truncate(text)
}

async function extractDocx(file) {
  const mammoth = await import('mammoth')
  const arrayBuffer = await file.arrayBuffer()
  const result = await mammoth.extractRawText({ arrayBuffer })
  return truncate(result.value)
}

/**
 * Extract plain text from an uploaded file in the browser.
 *
 * Returns { text } on success.
 * Throws Error with a user-friendly .friendlyMessage on failure.
 */
export async function extractTextFromFile(file) {
  const ext = fileExtension(file)

  if (!SUPPORTED_EXTENSIONS.includes(ext)) {
    const err = new Error(`Unsupported file type '${ext || 'unknown'}'`)
    err.friendlyMessage =
      'Hosted demo supports TXT, PDF and DOCX files. ' +
      'Image OCR requires the full backend deployment.'
    throw err
  }

  let text
  try {
    if (ext === '.txt') text = await extractTxt(file)
    else if (ext === '.pdf') text = await extractPdf(file)
    else text = await extractDocx(file)
  } catch (exc) {
    const err = new Error(`Extraction failed: ${exc.message}`)
    err.friendlyMessage = `Could not extract text from '${file.name}'.`
    throw err
  }

  if (!text) {
    const err = new Error('No text extracted')
    err.friendlyMessage = `No readable text found in '${file.name}'.`
    throw err
  }
  return { text }
}

import apiClient from './client.js'
import { extractTextFromFile } from './fileText.js'
import { isHFMode, predictTextHF } from './hf.js'

/**
 * Text credibility prediction.
 *
 * Backend mode (default): POST /api/predict with { "text": "..." }
 * HF mode (VITE_HF_MODEL_URL set): Hugging Face Inference API.
 *
 * Returns the real API response, e.g.
 * { text, prediction, credibility_category, confidence, probabilities, model }
 */
export async function predictText(text) {
  if (isHFMode()) {
    return predictTextHF(text)
  }
  const response = await apiClient.post('/predict', { text })
  return response.data
}

/**
 * File credibility prediction.
 *
 * Backend mode: POST /api/predict/file (multipart), server extracts text
 * (incl. PaddleOCR for images).
 * HF mode: text is extracted in the browser (TXT/PDF/DOCX; no image OCR)
 * and sent to the inference API.
 *
 * Returns the same response shape as predictText.
 */
export async function predictFile(file) {
  if (isHFMode()) {
    const { text } = await extractTextFromFile(file)
    return predictTextHF(text)
  }

  const formData = new FormData()
  // Field name must match the backend signature:
  //   async def predict_credibility_from_file(file: UploadFile = File(...))
  formData.append('file', file)
  const response = await apiClient.post('/predict/file', formData, {
    // Explicitly drop any inherited Content-Type (e.g. 'application/json')
    // so axios keeps the FormData untouched and the browser can set
    // 'multipart/form-data' with its own boundary. Without this, axios
    // would JSON-stringify the FormData and the backend would return
    // 422 "Field required" for the `file` field.
    headers: { 'Content-Type': undefined },
  })
  return response.data
}

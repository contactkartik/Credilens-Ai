import apiClient from './client.js'

/**
 * Text credibility prediction — POST /api/predict
 * Body: { "text": "..." }
 * Returns the real API response, e.g.
 * { text, prediction, credibility_category, confidence, probabilities, model }
 */
export async function predictText(text) {
  const response = await apiClient.post('/predict', { text })
  return response.data
}

/**
 * File credibility prediction — POST /api/predict/file
 * multipart/form-data with a single `file` field.
 * Returns the same response shape as predictText (text = extracted text).
 */
export async function predictFile(file) {
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
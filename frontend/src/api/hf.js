/**
 * Hugging Face Inference API client.
 *
 * Used only when VITE_HF_MODEL_URL is set (hosted/static deployment).
 * When unset, the app keeps calling the FastAPI backend as before.
 *
 * Env vars (set in Vercel project settings or .env.production):
 *   VITE_HF_MODEL_URL - e.g. https://api-inference.huggingface.co/models/<user>/credilens-roberta
 *   VITE_HF_TOKEN     - Hugging Face read access token
 */

const HF_MODEL_URL = import.meta.env.VITE_HF_MODEL_URL
const HF_TOKEN = import.meta.env.VITE_HF_TOKEN

export function isHFMode() {
  return Boolean(HF_MODEL_URL)
}

const CREDIBILITY_CATEGORIES = {
  true_side: 'Credible',
  false_side: 'Not Credible',
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

async function callInference(text, attempt = 0) {
  const response = await fetch(HF_MODEL_URL, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${HF_TOKEN}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ inputs: text }),
  })

  // 503 = model is cold-loading on HF's side; retry a few times
  if (response.status === 503 && attempt < 8) {
    await sleep(5000)
    return callInference(text, attempt + 1)
  }

  const data = await response.json().catch(() => null)

  if (!response.ok) {
    const message =
      data?.error ||
      (Array.isArray(data) ? data[0]?.error : null) ||
      `Inference request failed (HTTP ${response.status})`
    const err = new Error(message)
    err.friendlyMessage = message
    throw err
  }

  return data
}

/**
 * Predict credibility via HF Inference API.
 * Returns the same response shape as the FastAPI /api/predict endpoint.
 */
export async function predictTextHF(text) {
  const data = await callInference(text)

  // Response: [{ label: 'true_side'|'false_side', score: 0..1 }, ...]
  const scores = Array.isArray(data) ? data : []
  if (scores.length === 0) {
    const err = new Error('Empty response from inference API')
    err.friendlyMessage = 'The model returned no result. Please try again.'
    throw err
  }

  const probabilities = {}
  for (const item of scores) {
    probabilities[item.label] = item.score
  }

  const prediction = scores.reduce((best, item) =>
    item.score > best.score ? item : best
  )

  return {
    text,
    prediction: prediction.label,
    credibility_category: CREDIBILITY_CATEGORIES[prediction.label] || 'Unknown',
    confidence: prediction.score,
    probabilities,
    model: 'roberta-base (fine-tuned) — HF Inference',
  }
}

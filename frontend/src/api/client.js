import axios from 'axios'

/**
 * Shared Axios client for all CrediLens AI API calls.
 *
 * During development the Vite dev server proxies `/api` requests to the
 * FastAPI backend (see `vite.config.js`), so no absolute URL is needed.
 * In production, set `VITE_API_BASE_URL` (e.g. `https://host/api`).
 */
const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '/api',
  timeout: 60000, // OCR + model inference can take a while
  // IMPORTANT: do NOT set a default 'Content-Type' header here.
  // - JSON requests ({ text }) already get 'application/json' automatically
  //   from axios' transformRequest, so no default is needed.
  // - File uploads send FormData and MUST go out as
  //   'multipart/form-data; boundary=...' set by the browser. Forcing
  //   'application/json' makes axios serialize the FormData into a JSON
  //   string (the binary file part is dropped), so the backend receives
  //   no multipart fields and answers 422 "Field required" for `file`.
})

/**
 * Normalize backend errors so UI code can display `message` directly.
 * The FastAPI errors use `detail` for HTTP errors (string or list).
 */
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response) {
      const detail = error.response.data?.detail
      const message =
        typeof detail === 'string'
          ? detail
          : Array.isArray(detail)
            ? detail.map((d) => d.msg).join('; ')
            : `Request failed with status ${error.response.status}`
      error.friendlyMessage = message
    } else {
      error.friendlyMessage = 'Network error: backend is unreachable'
    }
    return Promise.reject(error)
  }
)

export default apiClient
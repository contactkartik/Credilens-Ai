import { Route, Routes } from 'react-router-dom'
import AnalyzePage from './pages/AnalyzePage.jsx'
import ResultsPage from './pages/ResultsPage.jsx'

function App() {
  return (
    <Routes>
      <Route path="/" element={<AnalyzePage />} />
      <Route path="/analyze" element={<AnalyzePage />} />
      <Route path="/results" element={<ResultsPage />} />
      {/* Stage 9: evidence/SHAP/history routes can be added here */}
      <Route path="*" element={<AnalyzePage />} />
    </Routes>
  )
}

export default App

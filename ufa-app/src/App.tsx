import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import './App.css'
import Home from './pages/Home'
import GameDetail from './pages/GameDetail'
import Prediction from './pages/Prediction'
import Embeddings from './pages/Embeddings'
import Turnovers from './pages/Turnovers'
import PullPlays from './pages/PullPlays'
import EPV from './pages/EPV'
import ZoneStrategy from './pages/ZoneStrategy'
import CompletionPredictor from './pages/CompletionPredictor'
import LineSynergy from './pages/LineSynergy'
import TeamPage from './pages/TeamPage'

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/game/:gameId" element={<GameDetail />} />
        <Route path="/predict" element={<Prediction />} />
        <Route path="/embeddings" element={<Embeddings />} />
        <Route path="/turnovers" element={<Turnovers />} />
        <Route path="/pull-plays" element={<PullPlays />} />
        <Route path="/epv" element={<EPV />} />
        <Route path="/zone-strategy" element={<ZoneStrategy />} />
        <Route path="/completion" element={<CompletionPredictor />} />
        <Route path="/line-synergy" element={<LineSynergy />} />
        <Route path="/team/:teamId" element={<TeamPage />} />
      </Routes>
    </Router>
  )
}

export default App

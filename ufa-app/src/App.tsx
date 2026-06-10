import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import './App.css'
import { Layout } from './components/Layout'
import Home from './pages/Home'
import GameDetail from './pages/GameDetail'
import Prediction from './pages/Prediction'
import Embeddings from './pages/Embeddings'
import Turnovers from './pages/Turnovers'
import EPV from './pages/EPV'
import LineSynergy from './pages/LineSynergy'
import TeamPage from './pages/TeamPage'
import PlayerPage from './pages/PlayerPage'
import PlayerSearch from './pages/PlayerSearch'

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/game/:gameId" element={<GameDetail />} />
          <Route path="/predict" element={<Prediction />} />
          <Route path="/embeddings" element={<Embeddings />} />
          <Route path="/turnovers" element={<Turnovers />} />
          <Route path="/epv" element={<EPV />} />
          <Route path="/line-synergy" element={<LineSynergy />} />
          <Route path="/team/:teamId" element={<TeamPage />} />
          <Route path="/player/:playerId" element={<PlayerPage />} />
          <Route path="/player-search" element={<PlayerSearch />} />
        </Routes>
      </Layout>
    </Router>
  )
}

export default App

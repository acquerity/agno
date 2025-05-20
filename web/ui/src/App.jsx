import React, { useState } from 'react';
import Login from './Login';
import ApiKeys from './ApiKeys';
import Chat from './Chat';
import GraphMemory from './GraphMemory';
import Tools from './Tools';

function App() {
  const [token, setToken] = useState(null);
  const [view, setView] = useState('chat');

  if (!token) {
    return <Login setToken={setToken} />;
  }

  return (
    <div className="app-root" style={{ display: 'flex', height: '100vh', background: '#181c20', color: '#fff' }}>
      <nav style={{ width: 220, background: '#23272e', padding: 24, display: 'flex', flexDirection: 'column', gap: 16 }}>
        <h2 style={{ color: '#7fffd4' }}>Agno Neurite UI</h2>
        <button onClick={() => setView('chat')}>Chat</button>
        <button onClick={() => setView('graph')}>Memory Graph</button>
        <button onClick={() => setView('tools')}>Tools</button>
        <button onClick={() => setView('apikeys')}>API Keys</button>
        <button onClick={() => setToken(null)} style={{ marginTop: 'auto', color: '#ff6b6b' }}>Logout</button>
      </nav>
      <main style={{ flex: 1, padding: 32, overflow: 'auto' }}>
        {view === 'chat' && <Chat token={token} />}
        {view === 'graph' && <GraphMemory token={token} />}
        {view === 'tools' && <Tools token={token} />}
        {view === 'apikeys' && <ApiKeys token={token} />}
      </main>
    </div>
  );
}

export default App; 
import React, { useState } from 'react';
import axios from 'axios';

function Login({ setToken }) {
  const [username, setUsername] = useState('devuser');
  const [password, setPassword] = useState('SuperSecureDevPassword123!');
  const [error, setError] = useState('');

  const handleLogin = async (e) => {
    e.preventDefault();
    setError('');
    try {
      const res = await axios.post('/auth/login', { username, password });
      setToken(res.data.access_token);
    } catch (err) {
      setError('Login failed.');
    }
  };

  return (
    <div style={{ maxWidth: 320, margin: '120px auto', background: '#23272e', padding: 32, borderRadius: 12 }}>
      <h2 style={{ color: '#7fffd4' }}>Login</h2>
      <form onSubmit={handleLogin}>
        <div style={{ marginBottom: 16 }}>
          <label>Username</label>
          <input value={username} onChange={e => setUsername(e.target.value)} style={{ width: '100%' }} />
        </div>
        <div style={{ marginBottom: 16 }}>
          <label>Password</label>
          <input type="password" value={password} onChange={e => setPassword(e.target.value)} style={{ width: '100%' }} />
        </div>
        <button type="submit" style={{ width: '100%', background: '#7fffd4', color: '#23272e', padding: 10, border: 'none', borderRadius: 6 }}>Login</button>
        {error && <div style={{ color: '#ff6b6b', marginTop: 12 }}>{error}</div>}
      </form>
    </div>
  );
}

export default Login; 
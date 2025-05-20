import React, { useEffect, useState } from 'react';
import axios from 'axios';

function ApiKeys({ token }) {
  const [keys, setKeys] = useState({ claude: '', grok: '', openai: '', google: '' });
  const [status, setStatus] = useState('');

  useEffect(() => {
    axios.get('/user/api_keys', { headers: { Authorization: 'Bearer ' + token } })
      .then(res => setKeys(res.data))
      .catch(() => setStatus('Failed to load API keys.'));
  }, [token]);

  const handleChange = (e) => {
    setKeys({ ...keys, [e.target.name]: e.target.value });
  };

  const handleSave = async (e) => {
    e.preventDefault();
    setStatus('');
    try {
      await axios.post('/user/api_keys/update', keys, { headers: { Authorization: 'Bearer ' + token } });
      setStatus('Saved!');
    } catch {
      setStatus('Failed to save.');
    }
  };

  return (
    <div style={{ maxWidth: 480, margin: '0 auto', background: '#23272e', padding: 32, borderRadius: 12 }}>
      <h2 style={{ color: '#7fffd4' }}>API Key Management</h2>
      <form onSubmit={handleSave}>
        {['claude', 'grok', 'openai', 'google'].map((k) => (
          <div key={k} style={{ marginBottom: 16 }}>
            <label style={{ textTransform: 'capitalize' }}>{k} API Key</label>
            <input name={k} value={keys[k] || ''} onChange={handleChange} style={{ width: '100%' }} />
          </div>
        ))}
        <button type="submit" style={{ width: '100%', background: '#7fffd4', color: '#23272e', padding: 10, border: 'none', borderRadius: 6 }}>Save</button>
        {status && <div style={{ marginTop: 12, color: status === 'Saved!' ? '#7fffd4' : '#ff6b6b' }}>{status}</div>}
      </form>
    </div>
  );
}

export default ApiKeys; 
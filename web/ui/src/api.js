import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

export const api = axios.create({
  baseURL: API_BASE_URL,
});

// Usage example:
// api.get('/agent/chat/history', { headers: { Authorization: 'Bearer ' + token } }) 
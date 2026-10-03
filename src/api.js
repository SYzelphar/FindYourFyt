import axios from 'axios';

// Vite proxies /api and /images to the Flask server (see vite.config.js).
const client = axios.create({ baseURL: '/api', timeout: 10000 });

const SESSION_KEY = 'fyf_session_id';
const DEPARTMENT_KEY = 'fyf_department';

// "Shopping for" choices -> H&M index groups understood by the backend.
export const DEPARTMENTS = {
    all: { label: 'All', groups: null },
    women: { label: 'Women', groups: ['Ladieswear', 'Divided'] },
    men: { label: 'Men', groups: ['Menswear'] },
};

const read = (key) => {
    try { return sessionStorage.getItem(key); } catch { return null; }
};
const write = (key, value) => {
    try { sessionStorage.setItem(key, value); } catch { /* storage unavailable */ }
};

export const getStoredSessionId = () => read(SESSION_KEY);
export const storeSessionId = (id) => write(SESSION_KEY, id);
export const getStoredDepartment = () => (DEPARTMENTS[read(DEPARTMENT_KEY)] ? read(DEPARTMENT_KEY) : 'all');
export const storeDepartment = (key) => write(DEPARTMENT_KEY, key);

export const isSessionNotFound = (error) =>
    error?.response?.data?.error?.code === 'session_not_found';

// Human-readable reason for a failed request.
export const describeError = (error) => {
    if (error?.response?.data?.error?.message) return error.response.data.error.message;
    if (error?.code === 'ECONNABORTED') return 'The recommendation server timed out.';
    // No response, or a bare 5xx from the Vite dev proxy when Flask is down.
    if (!error?.response || error.response.status >= 500) return 'Cannot reach the recommendation server.';
    return `Server error (${error.response.status}).`;
};

export async function createSession(department = 'all') {
    const groups = DEPARTMENTS[department]?.groups;
    const { data } = await client.post('/session', groups ? { departments: groups } : {});
    return data;
}

export async function fetchRecommendations(sessionId) {
    const { data } = await client.get('/recommendations', { params: { session_id: sessionId } });
    return data;
}

export async function sendSwipe(sessionId, productId, direction) {
    const { data } = await client.post('/swipe', { session_id: sessionId, product_id: productId, direction });
    return data;
}

export async function addSteer(sessionId, text) {
    const { data } = await client.post('/steer', { session_id: sessionId, text });
    return data;
}

export async function removeSteer(sessionId, text) {
    const { data } = await client.delete('/steer', { data: { session_id: sessionId, text } });
    return data;
}

export async function fetchProfile(sessionId) {
    const { data } = await client.get(`/session/${sessionId}/profile`);
    return data;
}

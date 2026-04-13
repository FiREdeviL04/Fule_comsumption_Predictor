import axios from "axios";

const apiBaseUrl = import.meta.env.DEV
  ? "http://localhost:5000"
  : import.meta.env.VITE_API_BASE_URL || "/_/backend";

const api = axios.create({
  baseURL: apiBaseUrl,
  timeout: 15000,
});

export const predictFuel = async (payload) => {
  const { data } = await api.post("/predict", payload);
  return data;
};

export const getInsights = async () => {
  const { data } = await api.get("/insights");
  return data;
};

export const getHistory = async (limit = 100) => {
  const { data } = await api.get(`/history?limit=${limit}`);
  return data;
};

export const exportHistoryCsvUrl = `${apiBaseUrl}/history/export`;

export default api;

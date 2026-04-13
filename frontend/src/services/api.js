import axios from "axios";

const api = axios.create({
  baseURL: "http://localhost:5000",
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

export const exportHistoryCsvUrl = "http://localhost:5000/history/export";

export default api;

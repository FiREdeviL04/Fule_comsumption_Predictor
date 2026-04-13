import { useEffect, useMemo, useState } from "react";
import { Link, Route, Routes } from "react-router-dom";
import ThemeToggle from "./components/ThemeToggle";
import HomePage from "./pages/HomePage";
import InsightsPage from "./pages/InsightsPage";
import { getHistory, getInsights, predictFuel } from "./services/api";

const samples = [
  { engine_size: 1.6, cylinders: 4, emissions: 186 },
  { engine_size: 2.4, cylinders: 4, emissions: 212 },
  { engine_size: 3.0, cylinders: 6, emissions: 263 },
];

function App() {
  const [theme, setTheme] = useState("dark");
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [insights, setInsights] = useState(null);
  const [history, setHistory] = useState([]);
  const [apiError, setApiError] = useState("");

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  useEffect(() => {
    const loadInitial = async () => {
      try {
        const [insightData, historyData] = await Promise.all([getInsights(), getHistory(50)]);
        setInsights(insightData);
        setHistory((historyData.history ?? []).slice().reverse());
      } catch (error) {
        setApiError(error?.response?.data?.error || "Failed to load API data.");
      }
    };

    loadInitial();
  }, []);

  const sampleCycle = useMemo(() => ({ index: 0 }), []);

  const handleSample = () => {
    const item = samples[sampleCycle.index % samples.length];
    sampleCycle.index += 1;
    return item;
  };

  const handlePredict = async (payload) => {
    setLoading(true);
    setApiError("");
    try {
      const result = await predictFuel(payload);
      setPrediction(result);
      const historyData = await getHistory(50);
      setHistory((historyData.history ?? []).slice().reverse());

      if (insights == null) {
        const insightData = await getInsights();
        setInsights(insightData);
      }
    } catch (error) {
      setApiError(error?.response?.data?.error || "Prediction request failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <h1>Car Fuel Prediction System</h1>
          <p>Flask + React application with tuned Random Forest models.</p>
        </div>
        <ThemeToggle
          theme={theme}
          onToggle={() => setTheme((prev) => (prev === "dark" ? "light" : "dark"))}
        />
      </header>

      <nav className="nav-row">
        <Link to="/">Home</Link>
        <Link to="/insights">Model Insights</Link>
      </nav>

      <Routes>
        <Route
          path="/"
          element={
            <HomePage
              onPredict={handlePredict}
              onSample={handleSample}
              loading={loading}
              prediction={prediction}
              insights={insights}
              history={history}
              apiError={apiError}
            />
          }
        />
        <Route path="/insights" element={<InsightsPage insights={insights} />} />
      </Routes>
    </main>
  );
}

export default App;

import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

function ChartsPanel({ prediction, insights }) {
  const predictionData = prediction
    ? [
        { name: "Fuel", value: prediction.fuel },
        { name: "HWY", value: prediction.hwy },
        { name: "COMB", value: prediction.comb },
      ]
    : [];

  const featureImportanceSource = insights?.feature_importance?.comb ?? {};
  const importanceData = Object.entries(featureImportanceSource).map(([name, value]) => ({
    name,
    value,
  }));

  return (
    <section className="panel chart-panel">
      <h2>Graph Section</h2>
      <div className="chart-grid">
        <div className="chart-box">
          <h3>Prediction Bar Chart</h3>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={predictionData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="value" fill="#0ea5e9" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-box">
          <h3>Feature Importance</h3>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={importanceData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="value" fill="#22c55e" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </section>
  );
}

export default ChartsPanel;

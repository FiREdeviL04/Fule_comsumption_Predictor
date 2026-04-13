import InputForm from "../components/InputForm";
import PredictionCards from "../components/PredictionCards";
import ChartsPanel from "../components/ChartsPanel";
import HistoryTable from "../components/HistoryTable";

function HomePage({
  onPredict,
  onSample,
  loading,
  prediction,
  insights,
  history,
  apiError,
}) {
  return (
    <div className="page-grid">
      <InputForm onPredict={onPredict} onSample={onSample} loading={loading} />
      {apiError && <p className="error panel">{apiError}</p>}
      <PredictionCards prediction={prediction} />
      <ChartsPanel prediction={prediction} insights={insights} />
      <HistoryTable rows={history} />
    </div>
  );
}

export default HomePage;

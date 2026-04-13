function InsightsPage({ insights }) {
  const performance = insights?.performance ?? {};

  return (
    <section className="panel">
      <h2>Model Insights</h2>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th>R2</th>
              <th>MAE</th>
              <th>CV Score</th>
              <th>Best Params</th>
            </tr>
          </thead>
          <tbody>
            {Object.keys(performance).length === 0 ? (
              <tr>
                <td colSpan={5}>No insights available.</td>
              </tr>
            ) : (
              Object.entries(performance).map(([name, metrics]) => (
                <tr key={name}>
                  <td>{name}</td>
                  <td>{Number(metrics.r2).toFixed(4)}</td>
                  <td>{Number(metrics.mae).toFixed(4)}</td>
                  <td>{Number(metrics.cv_score).toFixed(4)}</td>
                  <td>{JSON.stringify(metrics.best_params ?? {})}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export default InsightsPage;

import { exportHistoryCsvUrl } from "../services/api";

function HistoryTable({ rows }) {
  return (
    <section className="panel">
      <div className="panel-header">
        <h2>Prediction History</h2>
        <a className="button-link" href={exportHistoryCsvUrl}>Export CSV</a>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Time</th>
              <th>Engine</th>
              <th>Cylinders</th>
              <th>Emissions</th>
              <th>Fuel</th>
              <th>HWY</th>
              <th>COMB</th>
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 ? (
              <tr>
                <td colSpan={7}>No predictions yet.</td>
              </tr>
            ) : (
              rows.map((row, index) => (
                <tr key={`${row.timestamp}-${index}`}>
                  <td>{row.timestamp}</td>
                  <td>{row.engine_size}</td>
                  <td>{row.cylinders}</td>
                  <td>{row.emissions}</td>
                  <td>{row.fuel}</td>
                  <td>{row.hwy}</td>
                  <td>{row.comb}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export default HistoryTable;

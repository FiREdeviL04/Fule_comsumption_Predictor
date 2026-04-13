import { useState } from "react";

const initialState = {
  engine_size: "",
  cylinders: "",
  emissions: "",
};

function InputForm({ onPredict, onSample, loading }) {
  const [form, setForm] = useState(initialState);
  const [error, setError] = useState("");

  const onChange = (event) => {
    const { name, value } = event.target;
    setForm((prev) => ({ ...prev, [name]: value }));
  };

  const validate = () => {
    if (!form.engine_size || !form.cylinders || !form.emissions) {
      return "All fields are required.";
    }

    const engine = Number(form.engine_size);
    const cylinders = Number(form.cylinders);
    const emissions = Number(form.emissions);

    if (Number.isNaN(engine) || Number.isNaN(cylinders) || Number.isNaN(emissions)) {
      return "Inputs must be numeric values.";
    }

    if (engine <= 0 || cylinders <= 0 || emissions <= 0) {
      return "Inputs must be greater than zero.";
    }

    return "";
  };

  const handlePredict = () => {
    const message = validate();
    if (message) {
      setError(message);
      return;
    }

    setError("");
    onPredict({
      engine_size: Number(form.engine_size),
      cylinders: Number(form.cylinders),
      emissions: Number(form.emissions),
    });
  };

  const handleReset = () => {
    setForm(initialState);
    setError("");
  };

  const handleSample = () => {
    const sample = onSample();
    setForm({
      engine_size: String(sample.engine_size),
      cylinders: String(sample.cylinders),
      emissions: String(sample.emissions),
    });
    setError("");
  };

  return (
    <section className="panel">
      <h2>Input Form</h2>
      <div className="form-grid">
        <label>
          Engine Size
          <input
            name="engine_size"
            type="number"
            step="0.1"
            title="Engine displacement size in liters"
            value={form.engine_size}
            onChange={onChange}
          />
        </label>
        <label>
          Cylinders
          <input
            name="cylinders"
            type="number"
            title="Number of engine cylinders"
            value={form.cylinders}
            onChange={onChange}
          />
        </label>
        <label>
          Emissions
          <input
            name="emissions"
            type="number"
            step="0.1"
            title="Vehicle emissions value from test data"
            value={form.emissions}
            onChange={onChange}
          />
        </label>
      </div>

      <div className="button-row">
        <button onClick={handlePredict} disabled={loading}>Predict</button>
        <button className="secondary" onClick={handleReset} disabled={loading}>Reset</button>
        <button className="secondary" onClick={handleSample} disabled={loading}>Load Sample Data</button>
      </div>

      {loading && <p className="loading">Predicting...</p>}
      {error && <p className="error">{error}</p>}
    </section>
  );
}

export default InputForm;

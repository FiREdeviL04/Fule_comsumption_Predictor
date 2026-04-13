function PredictionCards({ prediction }) {
  const cards = [
    { label: "Fuel Consumption", value: prediction?.fuel },
    { label: "Highway (L/100km)", value: prediction?.hwy },
    { label: "Combined (L/100km)", value: prediction?.comb },
  ];

  return (
    <section className="panel">
      <h2>Prediction Results</h2>
      <div className="card-grid">
        {cards.map((card) => (
          <article className="result-card" key={card.label}>
            <p>{card.label}</p>
            <strong>{card.value == null ? "--" : `${card.value.toFixed(2)} L/100km`}</strong>
          </article>
        ))}
      </div>
    </section>
  );
}

export default PredictionCards;

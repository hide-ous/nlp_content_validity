import React, { useState, useEffect } from "react";

const API_BASE = "http://localhost:8002/api";

function App() {
  const [examples, setExamples] = useState([]);
  const [selectedExample, setSelectedExample] = useState(null);

  const [targetDef, setTargetDef] = useState("");
  const [adversaries, setAdversaries] = useState([""]);
  const [items, setItems] = useState([""]);
  const [itemScores, setItemScores] = useState([]);
  const [predictedScore, setPredictedScore] = useState(null);
  const [trueScore, setTrueScore] = useState(null);
  const [edited, setEdited] = useState(false);

  // Load example IDs
  useEffect(() => {
    fetch(`${API_BASE}/examples`)
      .then((res) => res.json())
      .then(setExamples);
  }, []);

  const loadExample = (id) => {
    fetch(`${API_BASE}/examples/${id}`)
      .then((res) => res.json())
      .then((data) => {
        setSelectedExample(id);
        setTargetDef(data.target_def);
        setAdversaries(data.adversaries);
        setItems(data.items);
        setTrueScore(data.true_score);
        setPredictedScore(null);
        setItemScores([]);
        setEdited(false);
      });
  };

  const handlePredict = async () => {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        target_def: targetDef,
        adversaries,
        items,
      }),
    });
    const data = await res.json();
    setPredictedScore(data.predicted_score);
    setItemScores(data.item_scores);
    setEdited(false);
  };

  const handleItemChange = (index, value) => {
    const newItems = [...items];
    newItems[index] = value;
    setItems(newItems);
    setEdited(true);
  };

  const addItem = () => {
    setItems([...items, ""]);
    setItemScores([]);
    setEdited(true);
  };

  const deleteItem = (index) => {
    const newItems = items.filter((_, i) => i !== index);
    setItems(newItems);
    setItemScores([]);
    setEdited(true);
  };

  return (
    <div style={{ padding: "1em", maxWidth: "800px", margin: "auto" }}>
      <h2>Scale Validity Predictor</h2>

      <label>Load Example:</label>
      <select onChange={(e) => loadExample(e.target.value)} value={selectedExample || ""}>
        <option value="">-- Select Example --</option>
        {examples.map((id) => (
          <option key={id} value={id}>{id}</option>
        ))}
      </select>

      <div style={{ marginTop: "1em" }}>
        <label>Focal Construct Definition:</label>
        <textarea
          value={targetDef}
          onChange={(e) => {
            setTargetDef(e.target.value);
            setEdited(true);
          }}
          rows={3}
          style={{ width: "100%" }}
        />
      </div>

      {adversaries.map((adv, idx) => (
        <div key={idx}>
          <label>Adversary {idx + 1} Definition:</label>
          <textarea
            value={adv}
            onChange={(e) => {
              const updated = [...adversaries];
              updated[idx] = e.target.value;
              setAdversaries(updated);
              setEdited(true);
            }}
            rows={2}
            style={{ width: "100%" }}
          />
        </div>
      ))}

      <h4>Scale Items</h4>
      {items.map((item, idx) => (
        <div key={idx} style={{ display: "flex", marginBottom: "0.5em" }}>
          <input
            type="text"
            value={item}
            onChange={(e) => handleItemChange(idx, e.target.value)}
            style={{ flex: 1, marginRight: "0.5em" }}
          />
          <button onClick={() => deleteItem(idx)}>Delete</button>
          {itemScores.length > 0 && (
            <span style={{ marginLeft: "1em" }}>
              Score: {itemScores[idx]?.toFixed(2)}
            </span>
          )}
        </div>
      ))}
      <button onClick={addItem}>+ Add Item</button>

      <div style={{ marginTop: "1em" }}>
        <button onClick={handlePredict}>Predict Validity</button>
      </div>

      {edited && (
        <p style={{ color: "orange" }}>
          Inputs have been edited since last prediction.
        </p>
      )}

      {predictedScore !== null && (
        <div style={{ marginTop: "1em" }}>
          <h4>Predicted Validity Score: {predictedScore.toFixed(2)}</h4>
          {trueScore !== null && (
            <p>Ground Truth Score: {trueScore.toFixed(2)}</p>
          )}
        </div>
      )}
    </div>
  );
}

export default App;

import React, { useEffect, useState } from "react";

const API_BASE = "http://localhost:8002/api";

function App() {
  const [exampleIds, setExampleIds] = useState([]);
  const [selectedId, setSelectedId] = useState("");

  const [targetDef, setTargetDef] = useState("");
  const [adversaries, setAdversaries] = useState(["", ""]);
  const [items, setItems] = useState([""]);

  const [itemScores, setItemScores] = useState([]);
  const [aggregatedScore, setAggregatedScore] = useState(null);
  const [edited, setEdited] = useState(false);
  const [models, setModels] = useState([]);
  const [modelName, setModelName] = useState("");

  // Fetch example IDs and available models on load
  useEffect(() => {
    fetch(`${API_BASE}/examples`)
      .then(res => res.json())
      .then(setExampleIds);

    fetch(`${API_BASE}/models`)
      .then(res => res.json())
      .then(data => {
        setModels(data);
        if (data.length > 0) setModelName(data[0]);
      });
  }, []);

  const loadExample = (id) => {
    fetch(`${API_BASE}/example/${encodeURIComponent(id)}`)
      .then(res => res.json())
      .then(data => {
        setSelectedId(id);
        setTargetDef(data.target_def);
        setAdversaries(data.adversaries);
        setItems(data.items);
        setItemScores([]);
        setAggregatedScore(null);
        setEdited(false);
      });
  };

  const predict = () => {
    fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model_name: modelName,
        target_def: targetDef,
        adversaries,
        items,
      }),
    })
      .then(res => res.json())
      .then(data => {
        setItemScores(data.item_scores);
        setAggregatedScore(data.aggregated_score);
        setEdited(false);
      });
  };

  const handleItemChange = (idx, value) => {
    const updated = [...items];
    updated[idx] = value;
    setItems(updated);
    setEdited(true);
  };

  const deleteItem = (idx) => {
    const updated = items.filter((_, i) => i !== idx);
    setItems(updated);
    setEdited(true);
  };

  const addItem = () => {
    setItems([...items, ""]);
    setEdited(true);
  };

  return (
    <div style={{ maxWidth: 800, margin: "auto", padding: "2rem" }}>
      <h2>Scale Validity Predictor</h2>

      {/* Section 1: Model and Example Selection */}
      <div style={{ marginBottom: "1.5rem" }}>
        <label><strong>Select Model:</strong></label><br />
        <select value={modelName} onChange={e => setModelName(e.target.value)}>
          {models.map((m) => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>

        <br /><br />
        <label><strong>Load Example:</strong></label><br />
        <select value={selectedId} onChange={e => loadExample(e.target.value)}>
          <option value="">-- Choose an example --</option>
          {exampleIds.map(id => (
            <option key={id} value={id}>{id}</option>
          ))}
        </select>
      </div>

      {/* Section 2: Definitions */}
      <div style={{ marginBottom: "1.5rem" }}>
        <label><strong>Focal Definition</strong></label>
        <textarea value={targetDef} onChange={e => { setTargetDef(e.target.value); setEdited(true); }}
                  rows={3} style={{ width: "100%" }} />

        {adversaries.map((adv, idx) => (
          <div key={idx}>
            <label><strong>Adversary {idx + 1}</strong></label>
            <textarea
              value={adv}
              rows={2}
              style={{ width: "100%" }}
              onChange={(e) => {
                const updated = [...adversaries];
                updated[idx] = e.target.value;
                setAdversaries(updated);
                setEdited(true);
              }}
            />
          </div>
        ))}
      </div>

      {/* Section 3: Items */}
      <div style={{ marginBottom: "1.5rem" }}>
        <h4>Scale Items</h4>
        {items.map((item, idx) => (
          <div key={idx} style={{ display: "flex", alignItems: "center", marginBottom: "0.5rem" }}>
            <input
              type="text"
              value={item}
              onChange={e => handleItemChange(idx, e.target.value)}
              style={{ flex: 1 }}
            />
            <button onClick={() => deleteItem(idx)} style={{ marginLeft: "0.5rem" }}>Delete</button>
            {itemScores.length > 0 && (
              <span style={{ marginLeft: "1rem" }}>Score: {itemScores[idx].toFixed(3)}</span>
            )}
          </div>
        ))}
        <button onClick={addItem}>+ Add Item</button>
      </div>

      {/* Section 4: Prediction Buttons */}
      <div style={{ marginBottom: "1.5rem" }}>
        <button onClick={predict}>Run Prediction</button>
        {edited && (
          <p style={{ color: "orange", marginTop: "0.5rem" }}>
            Fields have been edited since the last prediction.
          </p>
        )}
      </div>

      {/* Section 5: Output */}
      {aggregatedScore !== null && (
        <div>
          <h4>Predicted Validity Score: {aggregatedScore.toFixed(3)}</h4>
        </div>
      )}
    </div>
  );
}

export default App;

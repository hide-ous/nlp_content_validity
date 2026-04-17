import React, { useEffect, useState } from "react";
const API_BASE = import.meta.env.VITE_API_URL;
// const API_BASE = "http://localhost:8002/api";

// #  - bug: when an item is added after a prediction, the interface crashes content.js:1 content script loaded
// # index-C34YdxC5.js:52 Uncaught TypeError: Cannot read properties of undefined (reading 'toFixed')
// #     at index-C34YdxC5.js:52:215
// #     at Array.map (<anonymous>)
// #     at Gh (index-C34YdxC5.js:51:222)
// #     at xf (index-C34YdxC5.js:48:34113)
// #     at lc (index-C34YdxC5.js:48:61996)
// #     at R0 (index-C34YdxC5.js:48:72494)
// #     at ud (index-C34YdxC5.js:48:106451)
// #     at xy (index-C34YdxC5.js:48:105533)
// #     at _c (index-C34YdxC5.js:48:105369)
// #     at F0 (index-C34YdxC5.js:48:102497)
// # (anonymous) @ index-C34YdxC5.js:52
// # Gh @ index-C34YdxC5.js:51
// # xf @ index-C34YdxC5.js:48
// # lc @ index-C34YdxC5.js:48
// # R0 @ index-C34YdxC5.js:48
// # ud @ index-C34YdxC5.js:48
// # xy @ index-C34YdxC5.js:48
// # _c @ index-C34YdxC5.js:48
// # F0 @ index-C34YdxC5.js:48
// # rd @ index-C34YdxC5.js:48
// # ee @ index-C34YdxC5.js:48
// # hd @ index-C34YdxC5.js:48
// # (anonymous) @ index-C34YdxC5.js:48

function App() {
  const [exampleIds, setExampleIds] = useState([]);
  const [selectedId, setSelectedId] = useState("");

  const [targetDef, setTargetDef] = useState("");
  const [adversaries, setAdversaries] = useState(["", ""]);
  const [items, setItems] = useState([""]);

  const [itemScores, setItemScores] = useState([]);
  const [aggregatedScore, setAggregatedScore] = useState(null);
  const [percentile, setPercentile] = useState(null);
  const [edited, setEdited] = useState(false);
  const [models, setModels] = useState([]);
  const [modelName, setModelName] = useState("");
  const [predictionMade, setPredictionMade] = useState(false);


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
    if (id === "__custom__") {
      setSelectedId("");
      setTargetDef("");
      setAdversaries(["", ""]);
      setItems([""]);
      setItemScores([]);
      setAggregatedScore(null);
      setPercentile(null);
      setEdited(false);
      return;
    }

    fetch(`${API_BASE}/example/${encodeURIComponent(id)}`)
      .then(res => res.json())
      .then(data => {
        setSelectedId(id);
        setTargetDef(""); setTimeout(() => setTargetDef(data.target_def), 0);
        setAdversaries(["", ""]); setTimeout(() => setAdversaries(data.adversaries), 0);
        setItems([""]); setTimeout(() => setItems(data.items), 0);
        setItemScores([]);
        setAggregatedScore(null);
        setPercentile(null);
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
        setPercentile(data.percentile_rank);
        setEdited(false);
        setPredictionMade(true); // ✅ track that prediction occurred
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
      <div style={{ marginBottom: "1.5rem" }}>
        <label><strong>Select Model:</strong></label><br />
        <select
          value={modelName}
            onChange={e => {
              setModelName(e.target.value);
              if (predictionMade) setEdited(true); // ✅ only show change warning after prediction
              setItemScores([]);
              setAggregatedScore(null);
              setPercentile(null);
            }}
        >
          {models.map((m) => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>

        <br /><br />
        <label><strong>Load Example:</strong></label><br />
        <select value={selectedId} onChange={e => loadExample(e.target.value)}>
          <option value="__custom__">Write your own</option>
          {exampleIds.map(id => (
            <option key={id} value={id}>{id}</option>
          ))}
        </select>
      </div>

      <div style={{ marginBottom: "1.5rem" }}>
        <label><strong>Focal Definition</strong></label>
        <textarea
          value={targetDef}
          onChange={e => { setTargetDef(e.target.value); setEdited(true); }}
          rows={Math.max(4, targetDef.split('\n').length)}
          style={{ width: "100%", resize: "vertical" }}
        />


        {adversaries.map((adv, idx) => (
          <div key={idx}>
            <label><strong>Adversary {idx + 1}</strong></label>
            <textarea
              value={adv}
              rows={Math.max(4, targetDef.split('\n').length)}
              style={{ width: "100%", resize: "vertical" }}
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

      <div style={{ marginBottom: "1.5rem" }}>
        <h4>Scale Items</h4>
        {items.map((item, idx) => (
          <div key={idx} style={{ display: "flex", alignItems: "center", marginBottom: "0.5rem" }}>
            <textarea
              value={item}
              onChange={e => handleItemChange(idx, e.target.value)}
              rows={Math.max(2, item.split('\n').length)}
              style={{ width: '100%', resize: 'vertical' }}
            />
            <button onClick={() => deleteItem(idx)} style={{ marginLeft: "0.5rem" }}>Delete</button>
            {itemScores.length > 0 && (
              <span style={{ marginLeft: "1rem" }}>Score: {itemScores[idx].toFixed(3)}</span>
            )}
          </div>
        ))}
        <button onClick={addItem}>+ Add Item</button>
      </div>

      {aggregatedScore !== null && (
        <div>
          <h4>Predicted Validity Score:</h4>
          {percentile !== null && (
            <div style={{ marginTop: "1rem" }}>
              <p>higher than {percentile.toFixed(1)}% of reference scales</p>
              <div style={{
                position: 'relative',
                height: '20px',
                borderRadius: '4px',
                background: 'linear-gradient(to right, #ff6b6b, #ffe066, #51cf66)',
                marginTop: '4px'
              }}>
                <div style={{
                  position: 'absolute',
                  left: `${percentile}%`,
                  top: '-8px',
                  width: '2px',
                  height: '36px',
                  backgroundColor: 'black'
                }} />
              </div>
            </div>
          )}
        </div>
      )}

      <div style={{ marginBottom: "1.5rem" }}>
        <button onClick={predict}>Run Prediction</button>
        {edited && predictionMade && (
          <p style={{ color: "orange", marginTop: "0.5rem" }}>
            Fields have been edited since the last prediction.
          </p>
        )}
      </div>


    </div>
  );
}

export default App;
import React, { useState, useEffect } from "react";

const API_BASE = "http://localhost:8002/api";

export default function ScaleValidityApp() {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [targetDef, setTargetDef] = useState("");
  const [adversaries, setAdversaries] = useState(["", ""]);
  const [items, setItems] = useState([""]);
  const [scores, setScores] = useState(null);
  const [edited, setEdited] = useState(false);

  useEffect(() => {
    fetch(`${API_BASE}/models`)
      .then((res) => res.json())
      .then(setModels);
  }, []);

  const handlePredict = async () => {
    const payload = {
      model_name: selectedModel,
      target_def: targetDef,
      adversaries: adversaries.filter(a => a.trim() !== ""),
      items: items.filter(i => i.trim() !== "")
    };

    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    setScores(data);
    setEdited(false);
  };

  const handleClear = () => {
    setSelectedModel("");
    setTargetDef("");
    setAdversaries(["", ""]);
    setItems([""]);
    setScores(null);
    setEdited(false);
  };

  const handleItemChange = (idx, val) => {
    const updated = [...items];
    updated[idx] = val;
    setItems(updated);
    setEdited(true);
  };

  const handleAdversaryChange = (idx, val) => {
    const updated = [...adversaries];
    updated[idx] = val;
    setAdversaries(updated);
    setEdited(true);
  };

  const handleAddItem = () => {
    setItems([...items, ""]);
    setEdited(true);
  };

  const handleDeleteItem = (idx) => {
    const updated = items.filter((_, i) => i !== idx);
    setItems(updated);
    setEdited(true);
  };

  return (
    <div className="max-w-2xl mx-auto p-4 space-y-6">
      <h1 className="text-xl font-bold">Scale Validity via Sentence Similarity</h1>

      <div className="space-y-2">
        <label className="font-semibold">1. Select Similarity Model</label>
        <select
          value={selectedModel}
          onChange={(e) => {
            setSelectedModel(e.target.value);
            setEdited(true);
          }}
          className="w-full border rounded p-2"
        >
          <option value="">-- Choose a model --</option>
          {models.map((model) => (
            <option key={model} value={model}>{model}</option>
          ))}
        </select>
      </div>

      <div className="space-y-2">
        <label className="font-semibold">2. Target Definition</label>
        <textarea
          className="w-full border rounded p-2"
          rows={3}
          value={targetDef}
          onChange={(e) => {
            setTargetDef(e.target.value);
            setEdited(true);
          }}
        />
        {adversaries.map((adv, idx) => (
          <div key={idx}>
            <label className="font-semibold">Adversary {idx + 1} Definition</label>
            <textarea
              className="w-full border rounded p-2"
              rows={2}
              value={adv}
              onChange={(e) => handleAdversaryChange(idx, e.target.value)}
            />
          </div>
        ))}
      </div>

      <div className="space-y-2">
        <label className="font-semibold">3. Scale Items</label>
        {items.map((item, idx) => (
          <div key={idx} className="flex items-center space-x-2 mb-1">
            <input
              value={item}
              onChange={(e) => handleItemChange(idx, e.target.value)}
              className="flex-grow border rounded p-2"
            />
            <button onClick={() => handleDeleteItem(idx)} className="text-red-500">✕</button>
            {scores?.item_scores?.[idx] != null && (
              <span className="text-sm text-gray-600">
                Score: {scores.item_scores[idx].toFixed(3)}
              </span>
            )}
          </div>
        ))}
        <button onClick={handleAddItem} className="text-blue-600">+ Add Item</button>
      </div>

      <div className="space-x-4">
        <button
          onClick={handlePredict}
          className="bg-blue-600 text-white px-4 py-2 rounded"
        >
          Predict
        </button>
        <button
          onClick={handleClear}
          className="bg-gray-300 px-4 py-2 rounded"
        >
          Clear
        </button>
        {edited && <span className="text-yellow-600">Inputs edited since last prediction</span>}
      </div>

      {scores && (
        <div className="mt-6">
          <p className="font-semibold">Model Used: {scores.model_used}</p>
          <p className="font-semibold">Aggregated Validity Score: {scores.aggregated_score.toFixed(3)}</p>
        </div>
      )}
    </div>
  );
}

import React, { useEffect, useMemo, useState } from "react";
import "./App.css";

const API_BASE = import.meta.env.VITE_API_URL;
const STEP_COUNT = 6;

function App() {
  const [step, setStep] = useState(0);
  const [exampleIds, setExampleIds] = useState([]);
  const [selectedId, setSelectedId] = useState("__custom__");

  const [targetDef, setTargetDef] = useState("");
  const [adversaries, setAdversaries] = useState(["", ""]);
  const [items, setItems] = useState([""]);

  const [models, setModels] = useState([]);
  const [modelName, setModelName] = useState("");

  const [itemScores, setItemScores] = useState([]);
  const [itemOrbiting1Scores, setItemOrbiting1Scores] = useState([]);
  const [itemOrbiting2Scores, setItemOrbiting2Scores] = useState([]);
  const [aggregatedScore, setAggregatedScore] = useState(null);
  const [percentile, setPercentile] = useState(null);
  const [referenceMean, setReferenceMean] = useState(null);
  const [referenceMedian, setReferenceMedian] = useState(null);
  const [edited, setEdited] = useState(false);
  const [predictionMade, setPredictionMade] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);

  useEffect(() => {
    fetch(`${API_BASE}/examples`).then((res) => res.json()).then(setExampleIds);
    fetch(`${API_BASE}/models`)
      .then((res) => res.json())
      .then((data) => {
        setModels(data);
        if (data.length > 0) setModelName(data[0]);
      });
  }, []);

  const resetPrediction = () => {
    setItemScores([]);
    setItemOrbiting1Scores([]);
    setItemOrbiting2Scores([]);
    setAggregatedScore(null);
    setPercentile(null);
    setReferenceMean(null);
    setReferenceMedian(null);
    setPredictionMade(false);
  };

  const markEdited = () => {
    setEdited(true);
    resetPrediction();
  };

  const loadExample = (id) => {
    setSelectedId(id);
    if (id === "__custom__") {
      setTargetDef("");
      setAdversaries(["", ""]);
      setItems([""]);
      setEdited(false);
      resetPrediction();
      return;
    }

    fetch(`${API_BASE}/example/${encodeURIComponent(id)}`)
      .then((res) => res.json())
      .then((data) => {
        setTargetDef(data.target_def);
        setAdversaries(data.adversaries);
        setItems(data.items);
        setEdited(false);
        resetPrediction();
      });
  };

  const predict = () => {
    setIsPredicting(true);
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
      .then((res) => res.json())
      .then((data) => {
        setItemScores(data.item_scores ?? []);
        setItemOrbiting1Scores(data.item_orbiting_1_scores ?? []);
        setItemOrbiting2Scores(data.item_orbiting_2_scores ?? []);
        setAggregatedScore(data.aggregated_score ?? null);
        setPercentile(data.percentile_rank ?? null);
        setReferenceMean(data.reference_mean ?? null);
        setReferenceMedian(data.reference_median ?? null);
        setEdited(false);
        setPredictionMade(true);
        setStep(STEP_COUNT - 1);
      })
      .finally(() => setIsPredicting(false));
  };

  const addItem = () => {
    setItems([...items, ""]);
    markEdited();
  };

  const deleteItem = (idx) => {
    const updated = items.filter((_, i) => i !== idx);
    setItems(updated.length > 0 ? updated : [""]);
    markEdited();
  };

  const setAdversary = (idx, value) => {
    const updated = [...adversaries];
    updated[idx] = value;
    setAdversaries(updated);
    markEdited();
  };

  const itemRows = useMemo(
    () =>
      items.map((item, idx) => ({
        item,
        focal: itemScores[idx],
        orbiting1: itemOrbiting1Scores[idx],
        orbiting2: itemOrbiting2Scores[idx],
      })),
    [items, itemScores, itemOrbiting1Scores, itemOrbiting2Scores]
  );

  return (
    <main className="app">
      <header className="app-header">
        <h1>Scale Validity Predictor</h1>
        <p className="step-caption">
          Step {step + 1} of {STEP_COUNT}
        </p>
      </header>

      <section className="card">
        {step === 0 && (
          <>
            <h2>How this tool works</h2>
            <p>
              This tool estimates scale validity by comparing item similarity to one focal definition
              versus two orbiting definitions. You can start from an existing scale or write your own,
              edit everything, choose a model, and then review both scale-level and item-level outputs.
            </p>
            <p>
              The final result shows your predicted scale score against the dataset reference
              distribution and shows each item&apos;s similarity to focal, orbiting 1, and orbiting 2.
            </p>
          </>
        )}

        {step === 1 && (
          <>
            <h2>Select scale</h2>
            <label htmlFor="scale-select">Available scales</label>
            <select id="scale-select" value={selectedId} onChange={(e) => loadExample(e.target.value)}>
              <option value="__custom__">Write your own</option>
              {exampleIds.map((id) => (
                <option key={id} value={id}>
                  {id}
                </option>
              ))}
            </select>
          </>
        )}

        {step === 2 && (
          <>
            <h2>Edit definitions</h2>
            <label htmlFor="focal-definition">Focal definition</label>
            <textarea
              id="focal-definition"
              value={targetDef}
              onChange={(e) => {
                setTargetDef(e.target.value);
                markEdited();
              }}
            />

            <label htmlFor="orbiting-1">Orbiting definition 1</label>
            <textarea
              id="orbiting-1"
              value={adversaries[0]}
              onChange={(e) => setAdversary(0, e.target.value)}
            />

            <label htmlFor="orbiting-2">Orbiting definition 2</label>
            <textarea
              id="orbiting-2"
              value={adversaries[1]}
              onChange={(e) => setAdversary(1, e.target.value)}
            />
          </>
        )}

        {step === 3 && (
          <>
            <h2>Edit items</h2>
            <div className="items-list">
              {items.map((item, idx) => (
                <div key={idx} className="item-row">
                  <textarea
                    value={item}
                    onChange={(e) => {
                      const updated = [...items];
                      updated[idx] = e.target.value;
                      setItems(updated);
                      markEdited();
                    }}
                  />
                  <button type="button" className="danger" onClick={() => deleteItem(idx)}>
                    Remove
                  </button>
                </div>
              ))}
            </div>
            <button type="button" onClick={addItem}>
              + Add item
            </button>
          </>
        )}

        {step === 4 && (
          <>
            <h2>Select model</h2>
            <label htmlFor="model-select">Available models</label>
            <select
              id="model-select"
              value={modelName}
              onChange={(e) => {
                setModelName(e.target.value);
                markEdited();
              }}
            >
              {models.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
            <button type="button" onClick={predict} disabled={isPredicting || !modelName}>
              {isPredicting ? "Predicting..." : "Run prediction"}
            </button>
          </>
        )}

        {step === 5 && (
          <>
            <h2>Prediction results</h2>
            {aggregatedScore !== null ? (
              <>
                <p>
                  Predicted scale score: <strong>{aggregatedScore.toFixed(3)}</strong>
                </p>
                {percentile !== null && (
                  <p>Compared with the dataset, this score is higher than {percentile.toFixed(1)}% of scales.</p>
                )}
                <div className="stats-grid">
                  <div className="stat-box">
                    <span>Reference mean</span>
                    <strong>{referenceMean !== null ? referenceMean.toFixed(3) : "N/A"}</strong>
                  </div>
                  <div className="stat-box">
                    <span>Reference median</span>
                    <strong>{referenceMedian !== null ? referenceMedian.toFixed(3) : "N/A"}</strong>
                  </div>
                </div>

                <h3>Item similarities</h3>
                <div className="table-wrapper">
                  <table>
                    <thead>
                      <tr>
                        <th>Item</th>
                        <th>Focal</th>
                        <th>Orbiting 1</th>
                        <th>Orbiting 2</th>
                      </tr>
                    </thead>
                    <tbody>
                      {itemRows.map((row, idx) => (
                        <tr key={idx}>
                          <td>{row.item}</td>
                          <td>{row.focal !== undefined ? row.focal.toFixed(3) : "N/A"}</td>
                          <td>{row.orbiting1 !== undefined ? row.orbiting1.toFixed(3) : "N/A"}</td>
                          <td>{row.orbiting2 !== undefined ? row.orbiting2.toFixed(3) : "N/A"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {edited && predictionMade && (
                  <p className="warning">Inputs changed since the last prediction. Run prediction again for fresh results.</p>
                )}
              </>
            ) : (
              <p>No prediction yet. Go back to model step and run a prediction.</p>
            )}
          </>
        )}
      </section>

      <footer className="nav-actions">
        <button type="button" onClick={() => setStep((s) => Math.max(0, s - 1))} disabled={step === 0}>
          Previous
        </button>
        <button
          type="button"
          onClick={() => setStep((s) => Math.min(STEP_COUNT - 1, s + 1))}
          disabled={step >= STEP_COUNT - 1}
        >
          Next
        </button>
      </footer>
    </main>
  );
}

export default App;
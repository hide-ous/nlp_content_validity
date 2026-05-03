import React, { useEffect, useMemo, useState } from "react";
import "./App.css";

const API_BASE = import.meta.env.VITE_API_URL;
const STEP_COUNT = 6;
const MODEL_PRIORITY = ["fasttext-wmd", "sentence-t5-base"];

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
  const [interpretationText, setInterpretationText] = useState("");
  const [direction, setDirection] = useState("higher_better");
  const [edited, setEdited] = useState(false);
  const [predictionMade, setPredictionMade] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);

  useEffect(() => {
    fetch(`${API_BASE}/examples`).then((res) => res.json()).then(setExampleIds);
    fetch(`${API_BASE}/models`)
      .then((res) => res.json())
      .then((data) => {
        const ordered = [...data].sort((a, b) => {
          const ai = MODEL_PRIORITY.indexOf(a);
          const bi = MODEL_PRIORITY.indexOf(b);
          const aw = ai === -1 ? Number.MAX_SAFE_INTEGER : ai;
          const bw = bi === -1 ? Number.MAX_SAFE_INTEGER : bi;
          if (aw !== bw) return aw - bw;
          return a.localeCompare(b);
        });
        setModels(ordered);
        if (ordered.length > 0) setModelName(ordered[0]);
      });
  }, []);

  const resetPrediction = () => {
    setItemScores([]);
    setItemOrbiting1Scores([]);
    setItemOrbiting2Scores([]);
    setAggregatedScore(null);
    setPercentile(null);
    setInterpretationText("");
    setDirection("higher_better");
    setPredictionMade(false);
  };

  const markEdited = () => {
    setEdited(true);
    resetPrediction();
  };

  const loadExample = (id, { advanceToDefinitions = false } = {}) => {
    setSelectedId(id);
    if (id === "__custom__") {
      setTargetDef("");
      setAdversaries(["", ""]);
      setItems([""]);
      setEdited(false);
      resetPrediction();
      if (advanceToDefinitions) setStep(2);
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
        if (advanceToDefinitions) setStep(2);
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
        setInterpretationText(data.interpretation_text ?? "");
        setDirection(data.direction ?? "higher_better");
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

  const directionSymbol = direction === "lower_better" ? "↓" : "↑";
  const directionLabel = direction === "lower_better" ? "Lower is better" : "Higher is better";
  const hasAllDefinitions =
    targetDef.trim().length > 0 &&
    adversaries[0].trim().length > 0 &&
    adversaries[1].trim().length > 0;
  const hasAtLeastOneItem = items.some((item) => item.trim().length > 0);

  const startFromScratch = () => {
    setSelectedId("__custom__");
    setTargetDef("");
    setAdversaries(["", ""]);
    setItems([""]);
    setEdited(false);
    resetPrediction();
    setStep(1);
  };

  const handleNext = () => {
    if (step === 1) {
      const idToLoad = selectedId && selectedId !== "" ? selectedId : "__custom__";
      loadExample(idToLoad, { advanceToDefinitions: true });
      return;
    }
    if (step === 2 && !hasAllDefinitions) return;
    if (step === 3 && !hasAtLeastOneItem) return;
    setStep((s) => Math.min(STEP_COUNT - 1, s + 1));
  };

  return (
    <main className="app">
      <header className="app-header">
        <h1>Scale Validity Predictor</h1>
        <p className="step-caption">
          Step {step + 1} of {STEP_COUNT}
        </p>
      </header>

      <section className="card">
        <div key={step} className="step-panel">
          {step === 0 && (
          <>
            <h2>Welcome to ALCoVa</h2>
            <p>
              <strong>Mettiamo un riferimento a un preprint?</strong>
            </p>
            <p>
              ALCoVa (A Lightweight Content Validator) is a free, open-access web tool that helps you
              assess the content validity of a scale: whether the items in your scale capture the meaning
              of the construct you intend to measure (definitional correspondence) and whether they capture
              it more strongly than related but distinct constructs (definitional distinctiveness).
            </p>
            <p>
              The tool uses NLP models to compare each item against three definitions: one focal definition
              (the construct you want to measure) and two orbiting definitions (related constructs from which
              your scale should be distinguishable).
            </p>
            <h3>What you can do here</h3>
            <ul>
              <li>Start with an existing published scale from the literature, or build your own from scratch.</li>
              <li>Edit definitions and items at any step, and rerun the analysis as often as you like.</li>
              <li>
                See a scale-level validity estimate (compared to a benchmark of 112 published scales) plus
                an item-by-item breakdown.
              </li>
            </ul>
            <p>
              ALCoVa is a diagnostic aid, not a substitute for human content validation. Use it to refine
              your items before collecting human ratings, not to replace them.
            </p>
          </>
          )}

          {step === 1 && (
          <>
            <h2>Step 2: Choose how to start</h2>
            <p>You can begin in one of two ways. Pick the option that matches your goal.</p>
            <div className="choice-layout">
              <button
                type="button"
                className="ghost cta-choice"
                onClick={() => loadExample("__custom__", { advanceToDefinitions: true })}
              >
                Option A - Start writing
              </button>
              <p>
                Recommended if you are developing a new measure. Enter your construct definitions and
                items directly, then use ALCoVa as a real-time diagnostic while you draft, revise, and
                refine your scale.
              </p>
              <div className="choice-divider" aria-hidden="true">
                <span>or</span>
              </div>
            </div>
            <label htmlFor="scale-select">Option B - Load a published scale</label>
            <select
              id="scale-select"
              value={selectedId === "__custom__" ? "" : selectedId}
              onChange={(e) => loadExample(e.target.value, { advanceToDefinitions: true })}
            >
              <option value="" disabled>
                Select a scale...
              </option>
              {exampleIds.map((id) => (
                <option key={id} value={id}>
                  {id}
                </option>
              ))}
            </select>
            <p>
              Recommended if you are exploring the tool, or want a reference point. Choose one of 112
              published scales from the Colquitt et al. (2019) benchmark, inspect performance, compare with
              your own work, or modify items to see how changes affect predicted scores.
            </p>
          </>
          )}

          {step === 2 && (
          <>
            <h2>Step 3: Define your constructs</h2>
            <p>
              Content validity assessment requires three definitions: the construct you want to measure,
              plus two related constructs you want your scale to be distinguishable from.
            </p>
            <p>
              Together they let the tool evaluate both correspondence (does the item capture the focal
              construct?) and distinctiveness (does it capture the focal construct more than related ones?).
            </p>
            <label htmlFor="focal-definition">Focal definition</label>
            <p>
              Provide a clear, complete definition. The text you enter here will be used as the reference
              for every item.
            </p>
            <textarea
              id="focal-definition"
              value={targetDef}
              onChange={(e) => {
                setTargetDef(e.target.value);
                markEdited();
              }}
            />

            <label htmlFor="orbiting-1">Orbiting definition 1</label>
            <p>
              Choose a construct that is conceptually close to the focal one but should not be confused
              with it. Following Colquitt et al. (2019), good orbiting constructs sit at the same stage of
              causal flow as the focal construct, do not stand in a part-whole relationship to it, and share
              the same referent (for example, person, team, or organization).
            </p>
            <textarea
              id="orbiting-1"
              value={adversaries[0]}
              onChange={(e) => setAdversary(0, e.target.value)}
            />

            <label htmlFor="orbiting-2">Orbiting definition 2</label>
            <p>
              Choose a different conceptually proximal construct. Two orbiting definitions provide a more
              robust distinctiveness assessment than one.
            </p>
            <textarea
              id="orbiting-2"
              value={adversaries[1]}
              onChange={(e) => setAdversary(1, e.target.value)}
            />
            <p>
              Tip: avoid orbiting constructs that are nearly synonymous with the focal one, or clearly
              unrelated. The most informative orbiting constructs are close calls that researchers might
              genuinely confuse.
            </p>
            {!hasAllDefinitions && (
              <p className="warning">Please fill focal and both orbiting definitions before continuing.</p>
            )}
          </>
          )}

          {step === 3 && (
          <>
            <h2>Step 4: Edit your items</h2>
            <p>
              These are the items evaluated against your focal and orbiting definitions. Each item should be
              a single statement that a respondent would rate (for example, agree/disagree or never/always).
            </p>
            <h3>You can:</h3>
            <ul>
              <li>Edit any item by clicking in the text box.</li>
              <li>Remove an item with the ✕ button.</li>
              <li>Add new items with + Add item.</li>
            </ul>
            <p>
              Tip for diagnostic use: try modifying a single word (for example, a verb or qualifier) and
              rerun the analysis. Comparing before and after scores can reveal which item features drive the
              predicted validity.
            </p>
            <p>
              Multi-item scales typically include 3-7 items. Single-item measures are also supported but may
              produce more compressed scores; see the paper for details.
            </p>
            <div className="items-list">
              {items.map((item, idx) => (
                <div key={idx} className="item-row">
                  <textarea
                    rows={2}
                    value={item}
                    onChange={(e) => {
                      const updated = [...items];
                      updated[idx] = e.target.value;
                      setItems(updated);
                      markEdited();
                    }}
                  />
                  <button type="button" className="danger" onClick={() => deleteItem(idx)}>
                    <span aria-hidden="true">✕</span>
                    <span className="sr-only">Remove item</span>
                  </button>
                </div>
              ))}
            </div>
            <button type="button" onClick={addItem}>
              + Add item
            </button>
            {!hasAtLeastOneItem && (
              <p className="warning">Please provide at least one non-empty item before continuing.</p>
            )}
          </>
          )}

          {step === 4 && (
          <>
            <h2>Step 5: Choose a model</h2>
            <p>
              ALCoVa offers two NLP models, each optimized for a different aspect of content validity.
              Pick the one that best matches what you want to evaluate.
            </p>
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
            <p>
              <strong>sentence-t5-base</strong> is recommended for definitional correspondence (HTC). It is
              a contextual sentence embedding model suited to fine-grained semantic alignment.
            </p>
            <p>
              <strong>fasttext-wmd</strong> is recommended for definitional distinctiveness (HTD) and for
              single-item measures. It offers transparent, word-level alignment via Word Mover&apos;s Distance.
            </p>
            <p>If you are unsure, run both models for complementary perspectives.</p>
          </>
          )}

          {step === 5 && (
          <>
            <h2>Step 6: Your results</h2>
            {aggregatedScore !== null ? (
              <>
                <p>
                  Model used: <strong>{modelName}</strong>
                </p>
                <p className="direction-chip" title={directionLabel}>
                  <span aria-hidden="true">{directionSymbol}</span> {directionLabel}
                </p>
                <p>
                  Predicted content validity: <strong>{aggregatedScore.toFixed(3)}</strong>
                </p>
                {percentile !== null && (
                  <>
                    <p>
                      Your scale ranks higher than <strong>{percentile.toFixed(1)}%</strong> of the 112
                      published scales in the benchmark.
                    </p>
                    <div className="meter">
                      <div className="meter-gradient" />
                      <div className="meter-marker" style={{ left: `${percentile}%` }}>
                        <span className="meter-label">Current scale</span>
                      </div>
                    </div>
                  </>
                )}
                {interpretationText && <p className="interpretation-text">{interpretationText}</p>}
                <p>
                  The predicted score reflects how well the tool expects your scale to perform under a human
                  content-validity rating procedure (HTC x HTD). It is a probabilistic estimate, not a
                  definitive judgment.
                </p>
                <p>
                  Interpret scores in relative terms. Diagnostic value comes from the gap between focal and
                  orbiting scores per item, and each item&apos;s relative position in your scale.
                </p>

                <h3>Item-level diagnostics</h3>
                <div className="table-wrapper">
                  <table>
                    <thead>
                      <tr>
                        <th>Item</th>
                        <th title={targetDef}>Focal</th>
                        <th title={adversaries[0] || "No definition provided"}>Orbiting 1</th>
                        <th title={adversaries[1] || "No definition provided"}>Orbiting 2</th>
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
                <h3>What to look for</h3>
                <ul>
                  <li>
                    Strong correspondence and distinctiveness: focal should be substantially higher than both
                    orbiting scores.
                  </li>
                  <li>
                    Weak correspondence: focal is low relative to other items; revise wording to align with
                    the focal definition.
                  </li>
                  <li>
                    Weak distinctiveness: focal is close to or lower than an orbiting score; clarify wording
                    to increase construct specificity.
                  </li>
                </ul>
                <h3>Next steps</h3>
                <ul>
                  <li>Iterate: click Previous to refine items or definitions and rerun the analysis.</li>
                  <li>Compare across models to check whether patterns converge.</li>
                  <li>Document: save or screenshot outputs for scale development reports.</li>
                  <li>Validate with humans: use ALCoVa to prioritize items for human review.</li>
                </ul>
                {edited && predictionMade && (
                  <p className="warning">Inputs changed since the last prediction. Run prediction again for fresh results.</p>
                )}
                <button type="button" className="ghost" onClick={startFromScratch}>
                  Start from scratch
                </button>
              </>
            ) : (
              <p>No prediction yet. Go back to model step and run a prediction.</p>
            )}
          </>
          )}
        </div>
      </section>

      <footer className="nav-actions">
        {step !== 0 && (
          <button type="button" onClick={() => setStep((s) => Math.max(0, s - 1))}>
            Previous
          </button>
        )}
        {step < STEP_COUNT - 1 && step !== 4 && (
          <button
            type="button"
            onClick={handleNext}
            disabled={(step === 2 && !hasAllDefinitions) || (step === 3 && !hasAtLeastOneItem)}
          >
            Next
          </button>
        )}
        {step === 4 && (
          <button
            type="button"
            onClick={predict}
            disabled={isPredicting || !modelName || !hasAllDefinitions || !hasAtLeastOneItem}
          >
            {isPredicting ? "Predicting..." : "Run prediction"}
          </button>
        )}
      </footer>
    </main>
  );
}

export default App;
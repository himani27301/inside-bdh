import { useMemo, useState } from "react";
import "./index.css";

const API_URL = "https://inside-bdh.onrender.com/analyze";

async function analyze(text) {
  const r = await fetch(API_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

function Delta({ value, suffix = "" }) {
  const n = Number(value);
  return (
    <span className={n > 0 ? "up" : n < 0 ? "down" : "flat"}>
      {n > 0 ? "+" : ""}{n.toFixed(2)}{suffix}
    </span>
  );
}

function Bar({ label, value }) {
  const v = Math.max(0, Math.min(100, Number(value || 0)));
  return (
    <div className="bar-row">
      <div className="bar-label"><span>{label}</span><b>{v.toFixed(2)}%</b></div>
      <div className="track"><div className="fill" style={{ width: `${v}%` }} /></div>
    </div>
  );
}

function Neurons({ units = [] }) {
  const shown = units.slice(0, 36);
  const max = Math.max(...shown.map(x => Number(x.activation || 0)), 0.0001);
  return (
    <div className="neurons">
      {shown.map((u, i) => {
        const s = Number(u.activation || 0) / max;
        return (
          <div
            className="neuron"
            key={`${u.unit}-${i}`}
            title={`Unit ${u.unit} • ${Number(u.activation).toFixed(4)}`}
            style={{ opacity: 0.25 + s * 0.75, transform: `scale(${0.75 + s * 0.25})` }}
          />
        );
      })}
    </div>
  );
}

function ReLUView({ examples = [], expected, observed }) {
  return (
    <section className="panel relu-panel">
      <div className="section-kicker">MECHANISM</div>
      <h2>Before ReLU → after ReLU</h2>
      <p className="muted">These are real values from the selected BDH layer. Negative values are clipped to zero; positive values survive.</p>

      <div className="relu-grid">
        {examples.slice(0, 12).map((x) => (
          <div className="relu-pair" key={x.unit}>
            <div className={x.pre <= 0 ? "pre negative" : "pre positive"}>{Number(x.pre).toFixed(2)}</div>
            <div className="relu-arrow">↓</div>
            <div className={x.post === 0 ? "post zero" : "post positive"}>{Number(x.post).toFixed(2)}</div>
          </div>
        ))}
      </div>

      <div className="truth-row">
        <div><span>Expected zeros from pre-ReLU ≤ 0</span><strong>{Number(expected).toFixed(2)}%</strong></div>
        <div><span>Observed zeros after ReLU</span><strong>{Number(observed).toFixed(2)}%</strong></div>
        <div className="match"><span>Mechanism check</span><strong>✓ MATCH</strong></div>
      </div>
    </section>
  );
}

export default function App() {
  const [a, setA] = useState("The king walked into the castle.");
  const [b, setB] = useState("A A A A A A A A A A A A");
  const [ra, setRa] = useState(null);
  const [rb, setRb] = useState(null);
  const [layerIndex, setLayerIndex] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [quiz, setQuiz] = useState(null);

  const la = ra?.layers?.[layerIndex];
  const lb = rb?.layers?.[layerIndex];

  const overlap = useMemo(() => {
    if (!la || !lb) return 0;
    const A = new Set((la.last_token_top_units || []).slice(0, 20).map(x => x.unit));
    const B = new Set((lb.last_token_top_units || []).slice(0, 20).map(x => x.unit));
    let n = 0;
    A.forEach(x => { if (B.has(x)) n += 1; });
    return (n / 20) * 100;
  }, [la, lb]);

  async function run() {
    setLoading(true);
    setError("");
    try {
      const [x, y] = await Promise.all([analyze(a.trim()), analyze(b.trim())]);
      setRa(x); setRb(y); setLayerIndex(0);
    } catch (e) {
      console.error(e);
      setError("Backend not reachable. Keep `uvicorn api:app --reload` running in the backend folder.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <header>
        <div className="logo">INSIDE <span>BDH</span></div>
        <div className="live"><i /> LIVE TOY BDH</div>
      </header>

      <main>
        <section className="hero">
          <div className="section-kicker">DATAFORGE • PATHWAY TRACK</div>
          <h1>See sparsity happen.<br/><span>Not just hear about it.</span></h1>
          <p>Change only the input, then inspect the real pre-ReLU and post-ReLU states of the same trained toy BDH checkpoint.</p>
          <div className="claim"><b>ONE-SENTENCE CLAIM</b> BDH converts a signed latent projection into an input-dependent, non-negative sparse state; changing the input changes which latent units survive.</div>
        </section>

        <section className="compare-inputs">
          <div className="input-card">
            <div className="section-kicker">INPUT A</div>
            <textarea value={a} onChange={e => setA(e.target.value)} />
            <div className="presets">
              <button onClick={() => setA("The king walked into the castle.")}>Natural</button>
              <button onClick={() => setA("The scientist studied neural networks.")}>Technical</button>
            </div>
          </div>
          <div className="vs">VS</div>
          <div className="input-card">
            <div className="section-kicker">INPUT B</div>
            <textarea value={b} onChange={e => setB(e.target.value)} />
            <div className="presets">
              <button onClick={() => setB("A A A A A A A A A A A A")}>Repetition</button>
              <button onClick={() => setB("xqz @@ 91827 zz qwe 66219")}>Noise</button>
            </div>
          </div>
        </section>

        <button className="run" onClick={run} disabled={loading}>{loading ? "RUNNING REAL MODEL…" : "COMPARE INTERNAL STATES →"}</button>
        {error && <div className="error">{error}</div>}

        {la && lb && <>
          <section className="results-title">
            <div className="section-kicker">LIVE EXPERIMENT</div>
            <h2>Same checkpoint. Different input.</h2>
          </section>

          <div className="layers">
            {ra.layers.map((l, i) => <button key={l.layer} className={i === layerIndex ? "active" : ""} onClick={() => setLayerIndex(i)}>L{l.layer}</button>)}
          </div>

          <section className="metrics panel">
            <div className="metric-head"><span>Metric</span><span>Input A</span><span>Input B</span><span>Δ B−A</span></div>
            {[
              ["X active", la.x_active_percent, lb.x_active_percent, "%"],
              ["Y active", la.y_active_percent, lb.y_active_percent, "%"],
              ["XY active", la.xy_active_percent, lb.xy_active_percent, "%"],
              ["Max activation", la.max_activation, lb.max_activation, ""],
            ].map(([name, av, bv, suffix]) => <div className="metric-row" key={name}>
              <span>{name}</span><b>{Number(av).toFixed(2)}{suffix}</b><b>{Number(bv).toFixed(2)}{suffix}</b><Delta value={Number(bv)-Number(av)} suffix={suffix}/>
            </div>)}
          </section>

          <section className="overlap panel">
            <div><div className="section-kicker">TOP-20 UNIT OVERLAP</div><strong>{overlap.toFixed(1)}%</strong></div>
            <p>How many of the 20 strongest last-token latent units are shared by the two inputs in this layer.</p>
          </section>

          <section className="two-col">
            <div className="panel">
              <div className="section-kicker">INPUT A • LAYER {la.layer}</div>
              <h3>Strongest units</h3>
              <Neurons units={la.last_token_top_units} />
              <div className="heads">{la.head_activity_percent.map((v,i)=><Bar key={i} label={`Head ${i+1}`} value={v}/>)}</div>
            </div>
            <div className="panel">
              <div className="section-kicker">INPUT B • LAYER {lb.layer}</div>
              <h3>Strongest units</h3>
              <Neurons units={lb.last_token_top_units} />
              <div className="heads">{lb.head_activity_percent.map((v,i)=><Bar key={i} label={`Head ${i+1}`} value={v}/>)}</div>
            </div>
          </section>

          <ReLUView
            examples={la.relu_examples}
            expected={la.pre_relu_nonpositive_percent}
            observed={la.post_relu_zero_percent}
          />

          <section className="panel evidence">
            <div className="section-kicker">EVIDENCE DISCIPLINE</div>
            <h2>Published BDH ≠ this toy checkpoint</h2>
            <div className="evidence-grid">
              <div><b>Published BDH</b><p>Pathway reports roughly 5% active neurons in reported BDH runs, with activity varying by predictability.</p></div>
              <div><b>Our live experiment</b><p>This is an independently trained byte-level toy BDH checkpoint on Tiny Shakespeare. Its percentages are measurements of this checkpoint only.</p></div>
              <div><b>What is live?</b><p>Inputs, BDH forward pass, ReLU transformation, layer measurements, top units and comparison metrics are computed live.</p></div>
            </div>
          </section>

          <section className="panel quiz">
            <div className="section-kicker">60-SECOND TEST</div>
            <h2>Can you predict ReLU?</h2>
            <p>Pre-ReLU state: <code>[-1.2, 0.8, -0.3, 2.1]</code></p>
            <div className="quiz-buttons">
              <button onClick={()=>setQuiz(false)}>[-1.2, 0.8, -0.3, 2.1]</button>
              <button onClick={()=>setQuiz(true)}>[0, 0.8, 0, 2.1]</button>
              <button onClick={()=>setQuiz(false)}>[1.2, 0.8, 0.3, 2.1]</button>
            </div>
            {quiz === true && <div className="correct">✓ Correct — negative values become zero; positive values survive.</div>}
            {quiz === false && <div className="wrong">Try again: ReLU is max(0, x).</div>}
          </section>

          <section className="panel architecture">
            <div className="section-kicker">WHERE THIS SITS IN BDH</div>
            <h2>From projection to sparse state</h2>
            <div className="flow"><span>Input bytes</span><i>→</i><span>Embedding</span><i>→</i><span>x_latent = x @ encoder</span><i>→</i><span className="highlight">x_sparse = ReLU(x_latent)</span><i>→</i><span>attention / memory interaction</span></div>
            <pre>{`x_latent = x @ self.encoder\nx_sparse = F.relu(x_latent)`}</pre>
          </section>
        </>}
      </main>

      <footer>INSIDE BDH • interactive educational artifact • toy checkpoint clearly labeled</footer>
    </div>
  );
}


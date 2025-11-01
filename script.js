/* script.js — mobile-first research prototype predictor
   - in-browser TF.js tiny model trained on synthetic data
   - mobile-optimized UI interactions
   - NOT a real betting predictor: research prototype only
*/

(() => {
  // --- DOM refs
  const RESULTS = [
    "1.00X - 1.50X","1.51X - 2.00X","2.01X - 3.00X","3.01X - 4.00X",
    "4.01X - 5.00X","5.01X - 6.00X","6.01X - 7.00X","7.01X - 8.00X",
    "8.01X - 10.00X","10.01X - 15.00X","15.01X - 20.00X","20.01X+"
  ];

  const sel1 = document.getElementById('result-1');
  const sel2 = document.getElementById('result-2');
  const sel3 = document.getElementById('result-3');
  const out = document.getElementById('out');
  const status = document.getElementById('status');
  const logEl = document.getElementById('log');
  const modelReadyLabel = document.getElementById('model-ready');
  const pointsEl = document.getElementById('points');
  const confFill = document.getElementById('conf-fill');

  // Chart
  const lossCtx = document.getElementById('loss-chart')?.getContext?.('2d');
  let lossChart = null;

  // State
  const STORAGE_KEY = 'oldstudio_proto_v1';
  const LOG_KEY = 'oldstudio_proto_log_v1';
  let MODEL = null, TRAINED = false;
  let lossHistory = [];
  let HISTORY = JSON.parse(localStorage.getItem(LOG_KEY) || '[]');
  let points = Number(localStorage.getItem('oldstudio_points') || 1000);
  pointsEl.innerText = points;

  // --- helpers
  function populateSelects(){
    [sel1, sel2, sel3].forEach(s => {
      s.innerHTML = '<option value="" disabled selected>SELECT</option>';
      RESULTS.forEach((r, i) => {
        const opt = document.createElement('option');
        opt.value = i; opt.textContent = r;
        s.appendChild(opt);
      });
    });
  }
  populateSelects();

  function saveInputs(a,b,c){ try{ localStorage.setItem(STORAGE_KEY, JSON.stringify([a,b,c])); }catch(e){} }
  function loadInputs(){ try{ const v=JSON.parse(localStorage.getItem(STORAGE_KEY)||'null'); if(Array.isArray(v)&&v.length===3) return v; }catch(e){} return null; }
  (function restoreInputs(){ const v = loadInputs(); if(v){ sel1.value=v[0]; sel2.value=v[1]; sel3.value=v[2]; } })();

  function appendLog(item){
    HISTORY.unshift(item);
    if(HISTORY.length>500) HISTORY.length = 500;
    localStorage.setItem(LOG_KEY, JSON.stringify(HISTORY));
    renderLog();
  }
  function renderLog(){
    logEl.innerHTML = HISTORY.map(i => `<div>${i.time} | ${i.input} → ${i.pred} | conf:${i.conf}% | stake:${i.stake} | ${i.result}</div>`).join('');
  }
  renderLog();

  function idxToNumeric(i){
    const mids=[1.25,1.75,2.5,3.5,4.5,5.5,6.5,7.5,9.0,12.5,17.5,22.0];
    return mids[i] ?? (1+i);
  }

  function generateSynthetic(n=1800){
    const X=[], y=[];
    for(let k=0;k<n;k++){
      const base = Math.floor(Math.random()*RESULTS.length);
      const spike = Math.random()<0.06;
      const trendUp = Math.random()<0.46;
      const prev = Math.max(0, Math.min(RESULTS.length-1, base + (trendUp? -1:1) + Math.floor((Math.random()-0.5)*2)));
      const second = Math.max(0, Math.min(RESULTS.length-1, prev + Math.floor((Math.random()-0.5)*3)));
      const latest = Math.max(0, Math.min(RESULTS.length-1, second + (spike?3:(trendUp?1:-1)) + Math.floor((Math.random()-0.5)*2)));
      const noise = (Math.random()-0.5)*0.28;
      X.push([idxToNumeric(prev), idxToNumeric(second), idxToNumeric(latest)]);
      y.push(idxToNumeric(latest) * (spike ? (1 + Math.random()*1.2) : (1+noise)));
    }
    return { X: tf.tensor2d(X), y: tf.tensor1d(y) };
  }

  function buildModel(){
    const model = tf.sequential();
    model.add(tf.layers.dense({units:20, activation:'relu', inputShape:[3]}));
    model.add(tf.layers.dense({units:14, activation:'relu'}));
    model.add(tf.layers.dense({units:1}));
    model.compile({ optimizer: tf.train.adam(0.008), loss: 'meanSquaredError' });
    return model;
  }

  async function trainModel(epochs=18, onEpoch){
    status.innerText = 'Preparing data...';
    const data = generateSynthetic(Math.max(1200, epochs*90));
    const model = buildModel();
    status.innerText = 'Training model...';
    lossHistory = [];
    await model.fit(data.X, data.y, {
      epochs, batchSize:64, shuffle:true,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          lossHistory.push(logs.loss);
          if(typeof onEpoch === 'function') onEpoch(epoch, logs.loss);
          if(lossChart){ lossChart.data.datasets[0].data = lossHistory.slice(); lossChart.update(); }
        }
      }
    });
    data.X.dispose(); data.y.dispose();
    if(MODEL){ try{ MODEL.dispose(); }catch(e){} }
    MODEL = model; TRAINED = true;
    modelReadyLabel.innerText = 'ready (prototype)';
    status.innerText = 'Model trained (prototype)';
    renderLossChart();
  }

  function numericToRange(val){
    let best=0, bd=Infinity;
    for(let i=0;i<RESULTS.length;i++){
      const d = Math.abs(val - idxToNumeric(i));
      if(d < bd){ bd = d; best = i; }
    }
    const conf = Math.max(30, Math.min(95, Math.round(95 - bd*8 + (Math.random()*6 - 3))));
    return { label: RESULTS[best], idx: best, conf, raw: Number(val.toFixed(2)) };
  }

  async function runPrediction(i1,i2,i3){
    saveInputs(i1,i2,i3);
    if(!TRAINED){ await trainModel(14, (ep,loss)=> status.innerText = `Training... epoch ${ep+1}`); }
    const input = tf.tensor2d([[idxToNumeric(i1), idxToNumeric(i2), idxToNumeric(i3)]]);
    const output = MODEL.predict(input);
    const val = (await output.data())[0];
    input.dispose(); output.dispose();
    return numericToRange(val);
  }

  function showResult(mapped){
    out.innerHTML = `
      <div style="text-align:center">
        <div class="title" style="font-size:1.15rem">${mapped.label}</div>
        <div class="muted" style="margin-top:8px">Predicted numeric: <strong>${mapped.raw}x</strong></div>
      </div>
    `;
    // update confidence bar
    if(confFill){
      confFill.style.width = mapped.conf + '%';
      confFill.className = 'conf-fill ' + (mapped.conf > 72 ? 'good' : (mapped.conf > 50 ? 'warn' : 'bad'));
    }
    document.getElementById('prediction-details').innerText = `Confidence: ${mapped.conf}% — research prototype.`;
  }

  function renderLossChart(){
    if(!lossCtx) return;
    if(lossChart) lossChart.destroy();
    lossChart = new Chart(lossCtx, {
      type: 'line',
      data: { labels: lossHistory.map((_,i)=>i+1), datasets: [{ label:'loss', data: lossHistory, borderWidth:2, pointRadius:0 }]},
      options: { plugins:{legend:{display:false}}, scales:{ x:{display:false}, y:{display:true} } }
    });
  }

  // --- events
  document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const a = parseInt(sel1.value), b = parseInt(sel2.value), c = parseInt(sel3.value);
    if([a,b,c].some(v => Number.isNaN(v))){ out.innerHTML = '<div class="muted" style="color:#ffb3b3">Please select all three ranges.</div>'; return; }
    status.innerText = 'Running prediction...';
    out.innerHTML = `<div class="muted">Analyzing...</div>`;
    try{
      const mapped = await runPrediction(a,b,c);
      showResult(mapped);
      const now = new Date().toLocaleString();
      appendLog({ time: now, input: `${RESULTS[a]}|${RESULTS[b]}|${RESULTS[c]}`, pred: mapped.label, conf: mapped.conf, stake: 0, result: 'N/A' });
      logMsg(`Predicted ${mapped.label} (conf ${mapped.conf}%)`);
    }catch(err){
      console.error(err);
      out.innerHTML = '<div class="muted" style="color:#ffb3b3">Prediction failed.</div>';
    }
  });

  document.getElementById('retrain').addEventListener('click', async () => {
    status.innerText = 'Retraining model...';
    await trainModel(22, (ep,loss) => status.innerText = `Retraining... epoch ${ep+1}`);
    logMsg('Model retrained');
  });

  document.getElementById('stake-sim').addEventListener('click', async () => {
    const stake = Math.max(1, Math.floor(Number(document.getElementById('stake-amount').value) || 0));
    if(stake <= 0){ alert('Enter stake amount >= 1'); return; }
    const a = parseInt(sel1.value), b = parseInt(sel2.value), c = parseInt(sel3.value);
    if([a,b,c].some(v => Number.isNaN(v))){ alert('Select all three last results first'); return; }
    const pred = await runPrediction(a,b,c);
    const winProb = Math.min(0.95, Math.max(0.05, pred.conf / 110));
    const win = Math.random() < winProb;
    const payout = Math.max(1.2, pred.raw / 2.5);
    let change = 0;
    if(win){ change = Math.round(stake * payout); points += change; } else { change = -stake; points += change; }
    localStorage.setItem('oldstudio_points', points);
    pointsEl.innerText = points;
    const now = new Date().toLocaleString();
    appendLog({ time: now, input: `${RESULTS[a]}|${RESULTS[b]}|${RESULTS[c]}`, pred: pred.label, conf: pred.conf, stake, result: (win ? `WIN +${change}` : `LOSS ${change}`) });
    showResult(pred);
    logMsg(`Stake ${stake} → ${win ? 'WIN' : 'LOSS'} (${change}). Points: ${points}`);
  });

  document.getElementById('reset').addEventListener('click', () => {
    localStorage.removeItem(STORAGE_KEY);
    localStorage.removeItem(LOG_KEY);
    localStorage.removeItem('oldstudio_points');
    HISTORY = []; renderLog(); populateSelects(); out.innerHTML = '<div class="placeholder">Reset complete.</div>';
    points = 1000; pointsEl.innerText = points;
    logMsg('Reset saved state');
  });

  document.getElementById('export-log').addEventListener('click', () => {
    const rows = [['time','input','prediction','confidence','stake','result']];
    HISTORY.slice().reverse().forEach(r => rows.push([r.time, r.input, r.pred, r.conf, r.stake, r.result]));
    const csv = rows.map(r => r.map(c => `"${String(c).replace(/"/g,'""')}"`).join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob); const a = document.createElement('a'); a.href = url; a.download = 'oldstudio_log.csv'; a.click(); URL.revokeObjectURL(url);
  });

  function logMsg(m){ const t = new Date().toLocaleTimeString(); logEl.insertAdjacentHTML('afterbegin', `<div>[${t}] ${m}</div>`); }
  function renderLog(){ logEl.innerHTML = HISTORY.map(i => `<div>${i.time} | ${i.input} → ${i.pred} | conf:${i.conf}% | stake:${i.stake} | ${i.result}</div>`).join(''); }

  // background pretrain
  (async function background(){
    try{ await new Promise(r => setTimeout(r, 600)); if(!TRAINED) await trainModel(12); logMsg('Background pretrain complete'); } catch(e){ console.warn(e); status.innerText = 'Model not trained. Use Retrain or Predict.'; }
  })();

})();

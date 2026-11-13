const $ = id => document.getElementById(id);
let THR = null, timer = null, seq = 0;

const now = new Date(Date.now() - new Date().getTimezoneOffset() * 60000);
$('when').value = now.toISOString().slice(0, 16);

async function predict() {
  const mode = $('mode').value, when = $('when').value;
  if (!when) return;
  const mine = ++seq;                                   // ignore out-of-order responses
  try {
    const r = await fetch('/api/predict', {method: 'POST', headers: {'Content-Type': 'application/json'},
                                           body: JSON.stringify({shipping_mode: mode, order_time: when})});
    if (!r.ok) throw new Error('API returned ' + r.status);
    const d = await r.json();
    if (mine === seq) show(d);
  } catch (e) {
    $('verdict').className = 'verdict err';
    $('verdict').textContent = 'Could not reach the prediction API: ' + e.message;
  }
}

function show(d) {
  THR = d.threshold;
  const p = d.late_probability, hour = d.order_hour;
  $('pct').textContent = (p * 100).toFixed(1) + ' %';
  $('fill').style.width = (p * 100) + '%';
  $('fill').style.background = d.alert ? 'var(--bad)' : 'var(--ok)';
  $('mark').style.left = (THR * 100) + '%';
  $('thrlabel').textContent = 'alert at ' + Math.round(THR * 100) + ' %';
  let why = d.alert ? 'Warn the customer early, consider a faster shipping mode, or escalate this order.'
                    : 'No alert needed. This order is expected to arrive on time.';
  if (d.shipping_mode === 'Same Day') why += hour < 12 ? ' Placed before the noon cut-off, so same-day delivery is expected.'
                                                       : ' Placed after the noon cut-off, so it will almost always miss the same-day deadline.';
  $('verdict').className = 'verdict ' + (d.alert ? 'bad' : 'ok');
  $('verdict').innerHTML = (d.alert ? 'ALERT: likely late' : 'Likely on time') + '<span>' + why + '</span>';
  $('rows').innerHTML = d.by_mode.map(m =>
    `<tr class="${m.shipping_mode === d.shipping_mode ? 'cur' : ''}"><td>${m.shipping_mode}</td><td>${(m.late_probability * 100).toFixed(1)} %</td><td>${m.alert ? 'Alert' : 'On time'}</td></tr>`).join('');
}

async function loadInfo() {
  try {
    const m = await (await fetch('/api/model-info')).json(), t = m.test_metrics;
    $('facts').innerHTML = [[t.roc_auc.toFixed(3), 'test ROC-AUC'], [(t.accuracy * 100).toFixed(1) + ' %', 'accuracy'],
                            [(t.precision * 100).toFixed(0) + ' % / ' + (t.recall * 100).toFixed(0) + ' %', 'precision / recall']]
      .map(([v, l]) => `<div><b>${v}</b><span>${l}</span></div>`).join('');
    $('about').innerHTML = `Evaluated on ${m.test_orders.toLocaleString()} later orders (from ${m.test_start.slice(0, 10)}), cancelled orders excluded. ` +
      `Model: smoothed lookup table on <b>shipping mode x order hour</b>; no other order-time column beat random noise, and RandomForest, LightGBM and XGBoost scored the same. ` +
      `Alert threshold ${m.threshold.toFixed(3)}. It estimates lateness for orders that eventually ship; Standard Class orders (~40 % late) cannot be separated further from order-time data.`;
  } catch (e) { /* info box is optional */ }
}

['input', 'change'].forEach(ev => { $('mode').addEventListener(ev, predict); $('when').addEventListener(ev, predict); });
loadInfo(); predict();

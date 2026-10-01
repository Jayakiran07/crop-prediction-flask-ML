const form = document.querySelector('#predictionForm');
const result = document.querySelector('#result');
const error = document.querySelector('#error');
const sample = { nitrogen: 90, phosphorus: 45, potassium: 40, temperature: 27, humidity: 82, ph: 6.2, rainfall: 220 };

async function loadHistory() {
  const items = await fetch('/api/history').then(r => r.json());
  document.querySelector('#historyList').innerHTML = items.length ? items.map(i => `<div class="history-row"><span>${i.crop}</span><small>${i.confidence.toFixed(1)}% match · ${new Date(i.created_at).toLocaleDateString()}</small></div>`).join('') : '<p class="muted">No predictions yet.</p>';
}
document.querySelector('#sampleBtn').addEventListener('click', () => Object.entries(sample).forEach(([key, value]) => form.elements[key].value = value));
form.addEventListener('submit', async event => {
  event.preventDefault(); error.textContent = ''; result.hidden = true;
  const button = form.querySelector('.submit'); button.disabled = true; button.querySelector('span').textContent = 'Analyzing field…';
  try {
    const payload = Object.fromEntries(new FormData(form));
    const response = await fetch('/api/predict', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) });
    const data = await response.json(); if (!response.ok) throw new Error(data.error);
    result.innerHTML = `<p class="eyebrow">BEST MATCH</p><div class="crop-name">${data.crop}</div><div class="confidence"><div><span style="width:${data.confidence}%"></span></div><b>${data.confidence}% confidence</b></div><p class="alternatives">Also consider: ${data.alternatives.map(a => `${a.crop} (${a.confidence}%)`).join(' · ')}</p>`;
    result.hidden = false; loadHistory();
  } catch (err) { error.textContent = err.message || 'Something went wrong. Please try again.'; }
  finally { button.disabled = false; button.querySelector('span').textContent = 'Find my crop'; }
});
loadHistory();

let importanceChartInstance = null;
const HISTORY_KEY = 'agriMindPredictionHistory';
const MAX_HISTORY_ITEMS = 20;
const FEATURE_IDS = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'];
let predictionHistory = loadHistory();

function updateVal(id) {
    const input = document.getElementById(id);
    document.getElementById(`${id}-val`).textContent = input.value;
}

function applyPreset(type) {
    const presets = {
        coastal: { N: 90, P: 45, K: 40, ph: 6.5, temperature: 25, humidity: 85, rainfall: 240 },
        arid: { N: 110, P: 45, K: 20, ph: 7.2, temperature: 26, humidity: 78, rainfall: 75 },
        highland: { N: 100, P: 25, K: 30, ph: 6.2, temperature: 24, humidity: 60, rainfall: 160 }
    };

    const preset = presets[type];
    if (!preset) {
        return;
    }

    Object.entries(preset).forEach(([key, value]) => {
        document.getElementById(key).value = value;
        updateVal(key);
    });
}

function loadHistory() {
    try {
        const saved = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
        if (!Array.isArray(saved)) {
            return [];
        }

        return saved
            .filter((item) =>
                item &&
                typeof item.crop === 'string' &&
                Number.isFinite(item.confidence)
            )
            .map((item) => ({
                crop: item.crop,
                category: typeof item.category === 'string' ? item.category : 'General',
                confidence: item.confidence,
                timestamp: typeof item.timestamp === 'string' ? item.timestamp : null,
                time: typeof item.time === 'string' ? item.time : null,
                inputs: item.inputs && typeof item.inputs === 'object' ? item.inputs : {},
                recommendations: Array.isArray(item.recommendations)
                    ? item.recommendations.filter((recommendation) =>
                        recommendation && typeof recommendation.name === 'string'
                    )
                    : [{
                        name: item.crop,
                        confidence: item.confidence,
                        category: item.category || 'General',
                        season: '',
                        water: ''
                    }]
            }))
            .slice(0, MAX_HISTORY_ITEMS);
    } catch (error) {
        console.warn('Could not read saved prediction history.', error);
        return [];
    }
}

function saveHistory(entry) {
    predictionHistory = [entry, ...predictionHistory].slice(0, MAX_HISTORY_ITEMS);
    try {
        localStorage.setItem(HISTORY_KEY, JSON.stringify(predictionHistory));
        renderHistory();
        return true;
    } catch (error) {
        console.error('Could not save prediction history.', error);
        renderHistory();
        return false;
    }
}

function clearHistory() {
    if (!predictionHistory.length || !window.confirm('Clear all saved predictions from this browser?')) {
        return;
    }

    try {
        localStorage.removeItem(HISTORY_KEY);
    } catch (error) {
        console.error('Could not clear saved prediction history.', error);
        showFeedback('Could not clear saved history from this browser.', 'error');
        return;
    }

    predictionHistory = [];
    renderHistory();
}

function renderHistory() {
    const listContainer = document.getElementById('history-list');
    const exportButton = document.getElementById('export-history');
    exportButton.disabled = predictionHistory.length === 0;
    listContainer.replaceChildren();

    if (!predictionHistory.length) {
        const emptyMessage = document.createElement('p');
        emptyMessage.className = 'subtle-text';
        emptyMessage.textContent = 'Your recent recommendations will appear here.';
        listContainer.appendChild(emptyMessage);
        return;
    }

    predictionHistory.forEach((item) => {
        const row = document.createElement('div');
        row.className = 'history-item';

        const crop = document.createElement('div');
        crop.className = 'history-crop';

        const name = document.createElement('strong');
        name.textContent = item.crop;
        crop.appendChild(name);

        const details = document.createElement('span');
        const savedTime = item.timestamp
            ? new Date(item.timestamp).toLocaleString()
            : (item.time || 'Saved in an earlier version');
        details.textContent = `${item.category} · ${savedTime}`;
        crop.appendChild(details);

        const score = document.createElement('div');
        score.className = 'history-score';
        score.textContent = `${item.confidence}%`;

        row.append(crop, score);
        listContainer.appendChild(row);
    });
}

function csvCell(value) {
    const text = value === null || value === undefined ? '' : String(value);
    return `"${text.replaceAll('"', '""')}"`;
}

function exportHistory() {
    if (!predictionHistory.length) {
        return;
    }

    const columns = [
        'Prediction time', 'Rank', 'Crop', 'Model score (%)', 'Category',
        'Season', 'Water requirement', ...FEATURE_IDS
    ];
    const rows = [columns];

    predictionHistory.forEach((entry) => {
        entry.recommendations.forEach((recommendation, index) => {
            rows.push([
                entry.timestamp || entry.time || '',
                index + 1,
                recommendation.name,
                recommendation.confidence,
                recommendation.category,
                recommendation.season,
                recommendation.water,
                ...FEATURE_IDS.map((feature) => entry.inputs[feature] ?? '')
            ]);
        });
    });

    const csv = rows.map((row) => row.map(csvCell).join(',')).join('\r\n');
    const blob = new Blob(['\uFEFF', csv], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `agrimind-predictions-${new Date().toISOString().slice(0, 10)}.csv`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function showFeedback(message, type = 'error') {
    const feedback = document.getElementById('prediction-error');
    feedback.textContent = message;
    feedback.classList.toggle('warning-message', type === 'warning');
    feedback.classList.toggle('error-message', type === 'error');
    feedback.classList.remove('hidden');
}

function clearFeedback() {
    const feedback = document.getElementById('prediction-error');
    feedback.textContent = '';
    feedback.classList.add('hidden');
}

function setSubmitting(isSubmitting) {
    const button = document.getElementById('predict-button');
    button.disabled = isSubmitting;
    button.setAttribute('aria-busy', String(isSubmitting));
    document.getElementById('predict-button-label').textContent =
        isSubmitting ? 'Checking suitability…' : 'Find Suitable Crops';
}

async function updateSystemStatus() {
    const statusIndicator = document.getElementById('status-indicator');
    const statusText = document.getElementById('status-text');

    try {
        const response = await fetch('/health');
        const data = await response.json();
        if (response.ok && data.status === 'ok') {
            statusIndicator.classList.remove('warning', 'error');
            statusText.textContent = 'Recommendation model ready';
            return;
        }

        statusIndicator.classList.add('warning');
        statusText.textContent = 'Model setup required';
    } catch (error) {
        statusIndicator.classList.add('error');
        statusText.textContent = 'Backend offline';
    }
}

async function loadModelInfo() {
    try {
        const response = await fetch('/model-info');
        const data = await response.json();
        if (!response.ok || !data.success) {
            throw new Error(data.error || 'Model information is unavailable.');
        }

        document.getElementById('model-algorithm').textContent = data.algorithm;
        document.getElementById('metric-crop-count').textContent = data.supported_crops;
        renderChart(data.feature_importance);

        if (data.evaluation) {
            document.getElementById('metric-accuracy').textContent =
                `${(data.evaluation.accuracy * 100).toFixed(1)}%`;
            document.getElementById('metric-f1').textContent =
                `${(data.evaluation.macro_f1 * 100).toFixed(1)}%`;
            document.getElementById('metric-test-size').textContent =
                data.evaluation.test_samples.toLocaleString();
            document.getElementById('model-note').textContent =
                `Evaluated ${new Date(data.evaluation.evaluated_at).toLocaleDateString()} on held-out data. ` +
                'Because the training data is generated, these scores should not be treated as field performance.';
        } else {
            document.getElementById('model-algorithm').textContent =
                `${data.algorithm} · train to evaluate`;
            document.getElementById('model-note').textContent =
                'Run python train_model.py to generate held-out evaluation metrics. Model scores are not calibrated probabilities.';
        }
    } catch (error) {
        console.error('Could not load model information.', error);
        document.getElementById('model-algorithm').textContent = 'Model info unavailable';
        document.getElementById('model-note').textContent =
            'Check that the backend is running and model files are available.';
    }
}

document.getElementById('prediction-form').addEventListener('submit', async function(event) {
    event.preventDefault();
    clearFeedback();

    const payload = Object.fromEntries(
        FEATURE_IDS.map((id) => [id, document.getElementById(id).value])
    );

    setSubmitting(true);
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(data.error || 'The prediction request failed.');
        }

        renderResults(data);
        const top = data.top_recommendations[0];
        const saved = saveHistory({
            crop: top.name,
            category: top.category,
            confidence: top.confidence,
            timestamp: new Date().toISOString(),
            inputs: payload,
            recommendations: data.top_recommendations
        });
        if (!saved) {
            showFeedback('Prediction succeeded, but browser storage is unavailable; this result cannot persist after you leave the page.', 'warning');
        }
    } catch (error) {
        console.error('Prediction request failed.', error);
        showFeedback(error.message || 'Could not reach the prediction service. Please try again.');
    } finally {
        setSubmitting(false);
    }
});

document.getElementById('clear-history').addEventListener('click', clearHistory);
document.getElementById('export-history').addEventListener('click', exportHistory);

function renderResults(data) {
    document.querySelector('.hero-placeholder').classList.add('hidden');
    document.getElementById('hero-content').classList.remove('hidden');

    const top = data.top_recommendations[0];
    document.getElementById('hero-crop-name').textContent = top.name;
    document.getElementById('hero-icon').textContent = top.icon;
    document.getElementById('hero-confidence').textContent = `${top.confidence}% model score`;
    document.getElementById('meta-category').textContent = top.category;
    document.getElementById('meta-season').textContent = top.season;
    document.getElementById('meta-water').textContent = top.water;
    document.getElementById('hero-crop-desc').textContent =
        'Ranked using the model’s learned patterns across the supplied soil and climate measurements.';

    const listContainer = document.getElementById('rankings-list');
    listContainer.replaceChildren();

    data.top_recommendations.forEach((item, index) => {
        const row = document.createElement('div');
        row.className = 'ranking-item';

        const cropInfo = document.createElement('div');
        cropInfo.className = 'crop-info';
        const icon = document.createElement('span');
        icon.textContent = item.icon;
        const name = document.createElement('span');
        name.textContent = `${index + 1}. ${item.name}`;
        cropInfo.append(icon, name);

        const scoreGroup = document.createElement('div');
        scoreGroup.className = 'rank-score-group';
        const score = document.createElement('span');
        score.className = 'rank-score';
        score.textContent = `${item.confidence}%`;
        const barBackground = document.createElement('div');
        barBackground.className = 'rank-bar-bg';
        barBackground.setAttribute('aria-label', `${item.confidence}% model score`);
        const bar = document.createElement('div');
        bar.className = 'rank-bar-fill';
        bar.style.width = `${Math.max(0, Math.min(100, item.confidence))}%`;
        barBackground.appendChild(bar);
        scoreGroup.append(score, barBackground);
        row.append(cropInfo, scoreGroup);
        listContainer.appendChild(row);
    });
}

function renderChart(importances) {
    const canvas = document.getElementById('importanceChart');
    if (!canvas || typeof Chart === 'undefined') {
        return;
    }

    if (importanceChartInstance) {
        importanceChartInstance.destroy();
    }

    importanceChartInstance = new Chart(canvas.getContext('2d'), {
        type: 'bar',
        data: {
            labels: Object.keys(importances).map((feature) => feature.toUpperCase()),
            datasets: [{
                label: 'Relative feature importance',
                data: Object.values(importances),
                backgroundColor: 'rgba(16, 185, 129, 0.6)',
                borderColor: '#10b981',
                borderWidth: 1.5,
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (context) => `Relative importance: ${(context.raw * 100).toFixed(1)}%`
                    }
                }
            },
            scales: {
                x: { ticks: { color: '#94a3b8' }, grid: { display: false } },
                y: {
                    min: 0,
                    max: 1,
                    ticks: {
                        color: '#94a3b8',
                        callback: (value) => `${(value * 100).toFixed(0)}%`
                    },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' }
                }
            }
        }
    });
}

updateSystemStatus();
loadModelInfo();
renderHistory();

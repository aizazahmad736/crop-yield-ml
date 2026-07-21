let importanceChartInstance = null;

function updateVal(id) {
    const val = document.getElementById(id).value;
    document.getElementById(`${id}-val`).innerText = val;
}

function applyPreset(type) {
    const presets = {
        coastal: { N: 90, P: 45, K: 40, ph: 6.5, temperature: 25, humidity: 85, rainfall: 240 },
        arid: { N: 110, P: 45, K: 20, ph: 7.2, temperature: 26, humidity: 78, rainfall: 75 },
        highland: { N: 100, P: 25, K: 30, ph: 6.2, temperature: 24, humidity: 60, rainfall: 160 }
    };

    if (presets[type]) {
        const p = presets[type];
        for (const key in p) {
            document.getElementById(key).value = p[key];
            updateVal(key);
        }
    }
}

document.getElementById('prediction-form').addEventListener('submit', async function(e) {
    e.preventDefault();

    const payload = {
        N: document.getElementById('N').value,
        P: document.getElementById('P').value,
        K: document.getElementById('K').value,
        ph: document.getElementById('ph').value,
        temperature: document.getElementById('temperature').value,
        humidity: document.getElementById('humidity').value,
        rainfall: document.getElementById('rainfall').value
    };

    try {
        const res = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await res.json();

        if (data.success) {
            renderResults(data);
        } else {
            alert('Error making prediction: ' + data.error);
        }
    } catch (err) {
        console.error(err);
        alert('Server request failed.');
    }
});

function renderResults(data) {
    // Reveal Hero Card
    document.querySelector('.hero-placeholder').classList.add('hidden');
    document.getElementById('hero-content').classList.remove('hidden');

    const top = data.top_recommendations[0];
    document.getElementById('hero-crop-name').innerText = top.name;
    document.getElementById('hero-icon').innerText = top.icon;
    document.getElementById('hero-confidence').innerText = `${top.confidence}% Match`;
    document.getElementById('meta-category').innerText = top.category;
    document.getElementById('meta-season').innerText = top.season;
    document.getElementById('meta-water').innerText = top.water;

    // Render Rankings List
    const listContainer = document.getElementById('rankings-list');
    listContainer.innerHTML = '';

    data.top_recommendations.forEach((item, index) => {
        const row = document.createElement('div');
        row.className = 'ranking-item';
        row.innerHTML = `
            <div class="crop-info">
                <span>${item.icon}</span>
                <span>${item.name}</span>
            </div>
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="font-size: 0.8rem; font-weight: 600;">${item.confidence}%</span>
                <div class="rank-bar-bg">
                    <div class="rank-bar-fill" style="width: ${item.confidence}%;"></div>
                </div>
            </div>
        `;
        listContainer.appendChild(row);
    });

    // Render Chart
    renderChart(data.feature_importance);
}

function renderChart(importances) {
    const ctx = document.getElementById('importanceChart').getContext('2d');

    const labels = Object.keys(importances).map(k => k.toUpperCase());
    const values = Object.values(importances);

    if (importanceChartInstance) {
        importanceChartInstance.destroy();
    }

    importanceChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Relative Weight',
                data: values,
                backgroundColor: 'rgba(16, 185, 129, 0.6)',
                borderColor: '#10b981',
                borderWidth: 1.5,
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: { ticks: { color: '#94a3b8' }, grid: { display: false } },
                y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255, 255, 255, 0.05)' } }
            }
        }
    });
}

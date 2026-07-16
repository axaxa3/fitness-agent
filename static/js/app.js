// ========== 状态管理 ==========
const state = {
    userId: localStorage.getItem('fitness_user_id') || '',
    currentPage: 'onboarding',
    onboardSessionId: '',
    planSessionId: '',
    trainingSessionId: '',
    chatSessionId: '',
};

// ========== 导航切换 ==========
document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', () => {
        document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
        item.classList.add('active');
        document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
        const page = item.dataset.page;
        document.getElementById(`page-${page}`).classList.add('active');
        state.currentPage = page;
    });
});

// ========== SSE 工具函数 ==========
function connectSSE(endpoint, onProgress, onDelta, onFinal, onError) {
    const evtSource = new EventSource(endpoint);
    evtSource.addEventListener('progress', e => { if (onProgress) onProgress(JSON.parse(e.data)); });
    evtSource.addEventListener('delta', e => { if (onDelta) onDelta(JSON.parse(e.data)); });
    evtSource.addEventListener('final', e => { if (onFinal) onFinal(JSON.parse(e.data)); evtSource.close(); });
    evtSource.addEventListener('error', e => { if (onError) onError(JSON.parse(e.data)); evtSource.close(); });
    evtSource.onerror = () => { if (onError) onError({ message: 'Connection failed' }); evtSource.close(); };
    return evtSource;
}

// ========== Onboarding ==========
document.getElementById('onboard-start').addEventListener('click', async () => {
    const res = await fetch('/api/onboard/start', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({})
    });
    const data = await res.json();
    state.userId = data.user_id;
    state.onboardSessionId = data.session_id;
    localStorage.setItem('fitness_user_id', state.userId);
    document.getElementById('user-id-display').textContent = state.userId;
    document.getElementById('onboard-start').style.display = 'none';
    addOnboardMsg('assistant', data.message);
});

document.getElementById('onboard-send').addEventListener('click', async () => {
    const input = document.getElementById('onboard-input');
    const msg = input.value.trim();
    if (!msg) return;
    addOnboardMsg('user', msg);
    input.value = '';

    const res = await fetch('/api/onboard/message', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            user_id: state.userId, session_id: state.onboardSessionId, message: msg
        })
    });
    const { session_id } = await res.json();
    let reply = '';
    connectSSE(
        `/api/onboard/stream/${session_id}`,
        null,
        d => { reply += d.content; updateLastOnboardMsg(reply); },
        d => {
            if (d.status === 'complete') {
                addOnboardMsg('assistant', '✅ 用户画像已生成！点击"我的计划"生成训练计划吧~');
            }
        },
        d => addOnboardMsg('assistant', `❌ 出错了: ${d.message}`)
    );
});

function addOnboardMsg(role, content) {
    const container = document.getElementById('onboard-messages');
    const div = document.createElement('div');
    div.className = `chat-msg ${role}`;
    div.textContent = content;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}

function updateLastOnboardMsg(content) {
    const msgs = document.querySelectorAll('#onboard-messages .chat-msg.assistant');
    if (msgs.length > 0) msgs[msgs.length - 1].textContent = content;
}

// ========== Plan ==========
document.getElementById('plan-generate').addEventListener('click', async () => {
    if (!state.userId) { alert('请先完成新手引导'); return; }
    const progressDiv = document.getElementById('plan-progress');
    progressDiv.style.display = 'block';
    progressDiv.textContent = '正在生成训练计划...';

    const res = await fetch('/api/plan/generate', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ user_id: state.userId })
    });
    const { session_id } = await res.json();

    connectSSE(
        `/api/plan/stream/${session_id}`,
        d => { progressDiv.textContent = d.message; },
        d => { progressDiv.textContent += '\n' + d.content; },
        d => { progressDiv.style.display = 'none'; loadCurrentPlan(); },
        d => { progressDiv.textContent = '生成失败: ' + d.message; }
    );
});

async function loadCurrentPlan() {
    const res = await fetch(`/api/plan/current?user_id=${state.userId}`);
    const plan = await res.json();
    if (plan.error) {
        document.getElementById('plan-display').innerHTML = `<p style="color:#ef4444">${plan.error}</p>`;
        return;
    }
    document.getElementById('plan-display').innerHTML = plan.sessions.map(s => `
        <div class="plan-session">
            <h4>${s.day} - ${s.name}</h4>
            ${s.exercises.map(e => `
                <div class="plan-exercise">
                    <span class="exercise-name">${e.order}. ${e.exercise_name || e.exercise_id}</span>
                    <span class="exercise-detail">${e.sets}组 × ${e.target_reps}次 @ RPE ${e.rpe_target} | 休息${e.rest_seconds || 90}s</span>
                </div>
            `).join('')}
        </div>
    `).join('');
}

// ========== Training ==========
document.getElementById('training-load').addEventListener('click', loadTodayTraining);

async function loadTodayTraining() {
    if (!state.userId) { alert('请先完成新手引导'); return; }
    const res = await fetch(`/api/train/today?user_id=${state.userId}`);
    const data = await res.json();
    if (data.error) {
        document.getElementById('training-display').innerHTML = `<p style="color:#ef4444">${data.error}</p>`;
        return;
    }
    document.getElementById('training-display').innerHTML = `
        <p style="margin-bottom:16px;color:#a0a0a0">${data.progression_strategy || ''}</p>
        ${data.sessions.map(s => `
            <h3>${s.name || '训练日'}</h3>
            ${s.exercises.map(e => `
                <div class="training-exercise-card">
                    <div class="exercise-name">${e.exercise_name || e.exercise_id} — ${e.sets}组 × ${e.target_reps}次</div>
                    <div class="set-rows" data-exercise="${e.exercise_id}">
                        ${Array.from({length: e.sets}, (_, i) => `
                            <div class="set-row" data-set="${i+1}">
                                <span>第${i+1}组</span>
                                <input type="number" placeholder="次数" class="set-reps" min="0" />
                                <input type="number" placeholder="重量kg" class="set-weight" min="0" step="0.5" />
                                <input type="number" placeholder="RPE(1-10)" class="set-rpe" min="1" max="10" />
                                <button class="btn-done" onclick="toggleSetDone(this)">✓</button>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `).join('')}
            <button class="btn-primary" onclick="completeTraining('${state.userId}')" style="margin-top:16px">完成训练</button>
        `).join('')}
    `;
}

function toggleSetDone(btn) {
    btn.classList.toggle('set-done');
    const row = btn.parentElement;
    row.querySelectorAll('input').forEach(i => i.classList.toggle('set-done'));
}

async function completeTraining(userId) {
    const feel = prompt('整体感受？(good/ok/hard)', 'good') || 'good';
    const notes = prompt('有什么想记录的？（可选）', '') || '';
    const res = await fetch('/api/train/complete', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ user_id: userId, session_id: Date.now().toString(), overall_feel: feel, notes })
    });
    const { session_id } = await res.json();
    let summary = '';
    connectSSE(
        `/api/train/stream/${session_id}`,
        null,
        d => { summary += d.content; },
        d => { alert('训练复盘完成！\n' + (d.summary?.feedback || '')); loadTodayTraining(); },
        d => alert('复盘出错: ' + d.message)
    );
}

// ========== Chat ==========
let chatBuffer = '';
document.getElementById('onboard-input').addEventListener('keydown', e => {
    if (e.key === 'Enter') document.getElementById('onboard-send').click();
});
document.getElementById('chat-input').addEventListener('keydown', e => {
    if (e.key === 'Enter') document.getElementById('chat-send').click();
});
document.getElementById('chat-send').addEventListener('click', async () => {
    const input = document.getElementById('chat-input');
    const msg = input.value.trim();
    if (!msg) return;

    addChatMsg('user', msg);
    input.value = '';

    const res = await fetch('/api/chat/message', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ user_id: state.userId || 'anonymous', message: msg })
    });
    const { session_id } = await res.json();
    addChatMsg('assistant', '');
    chatBuffer = '';

    connectSSE(
        `/api/chat/stream/${session_id}`,
        null,
        d => { chatBuffer += d.content; updateLastChatMsg(chatBuffer); },
        d => { updateLastChatMsg(d.answer || chatBuffer); },
        d => { updateLastChatMsg('❌ ' + d.message); }
    );
});

function addChatMsg(role, content) {
    const container = document.getElementById('chat-messages');
    const div = document.createElement('div');
    div.className = `chat-msg ${role}`;
    div.textContent = content;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}

function updateLastChatMsg(content) {
    const msgs = document.querySelectorAll('#chat-messages .chat-msg.assistant');
    if (msgs.length > 0) {
        msgs[msgs.length - 1].innerHTML = content.replace(/\n/g, '<br>');
    }
}

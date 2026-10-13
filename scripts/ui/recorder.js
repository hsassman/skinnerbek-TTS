// Reference-clip recorder. Runs as a gr.HTML `js_on_load`, so `element`,
// `props`, `trigger` and `upload` are provided by Gradio.
//
// Why this exists instead of gr.Audio(sources=["microphone"]): Gradio's
// recorder calls getUserMedia and, when that fails or the permission prompt is
// dismissed, swallows the error -- the Record button simply does nothing. Its
// device list is also read once, before permission, so it shows "No microphone
// found" forever. This recorder asks explicitly, explains every failure with a
// fix, and encodes a clean 24 kHz mono 16-bit WAV in the browser (the model's
// own rate) so no server-side conversion or ffmpeg is needed.
//
// The template never interpolates ${value}, so setting props.value does not
// re-render (and wipe) this live DOM; it only carries the uploaded path back to
// Python via trigger('change').
(() => {
  const TARGET_SR = 24000, MIN_S = 5, GOOD_S = 30, MAX_S = 45;
  const host = element.querySelector('.tts-rec') || element;
  const MIC = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3"/></svg>';
  host.innerHTML = `
    <div class="tts-rec__row">
      <button type="button" class="tts-rec__btn" data-act="toggle" aria-label="Start recording">${MIC}<span class="sq"></span></button>
      <div class="tts-rec__info">
        <div class="tts-rec__title"><span class="live" aria-hidden="true"></span><span data-el="title">Record in the browser</span></div>
        <div class="tts-rec__sub" data-el="sub">Press the button and read the passage above in your natural voice. Aim for 10&ndash;20 seconds.</div>
      </div>
      <span class="tts-rec__level" data-el="level" hidden>Level</span>
      <div class="tts-rec__time" data-el="time" aria-live="off">0:00</div>
    </div>
    <canvas data-el="wave" height="64" aria-hidden="true"></canvas>
    <div class="tts-rec__zone" aria-hidden="true"><div class="ok"></div><div class="fill" data-el="fill"></div></div>
    <div class="tts-rec__legend" aria-hidden="true"><span>0s</span><span>5s &middot; good from here</span><span>30s</span><span>45s max</span></div>
    <div class="tts-rec__row" data-el="devrow" hidden>
      <label style="font-size:.8rem;color:var(--tts-muted)">Microphone&nbsp;
        <select data-el="device" aria-label="Microphone"></select></label>
    </div>
    <div class="tts-rec__msg" data-el="msg" role="status" aria-live="polite"></div>
    <div class="tts-rec__review">
      <audio data-el="player" controls preload="auto"></audio>
      <div class="tts-rec__actions">
        <button type="button" class="primary" data-act="use">Use this take</button>
        <button type="button" data-act="again">Record again</button>
      </div>
    </div>`;

  const $ = (k) => host.querySelector(`[data-el="${k}"]`);
  const btn = host.querySelector('[data-act="toggle"]');
  const canvas = $('wave'), ctx2d = canvas.getContext('2d');
  let state = 'idle', stream = null, ac = null, src = null, proc = null, analyser = null;
  let chunks = [], inRate = 48000, started = 0, raf = 0, timer = 0, blob = null, peaks = [];
  let deviceId = '';

  const setState = (s) => { state = s; host.dataset.state = s;
    btn.setAttribute('aria-label', s === 'recording' ? 'Stop recording' : 'Start recording'); };
  const say = (html, tone = '') => { const m = $('msg'); m.innerHTML = html; m.dataset.tone = tone; };
  const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
  setState('idle');

  const FIXES = {
    denied: `<strong>The browser is not allowed to use the microphone.</strong><ol>
      <li>Click the <b>microphone / lock icon</b> at the left of the address bar and set Microphone to <b>Allow</b>.</li>
      <li>On Windows, open <b>Settings &rsaquo; Privacy &amp; security &rsaquo; Microphone</b> and switch on
          <b>Microphone access</b> and <b>Let desktop apps access your microphone</b>.</li>
      <li>Then press the record button again.</li></ol>`,
    notfound: `<strong>No microphone was found.</strong> Plug one in (or enable it in Windows Sound settings) and try again.`,
    busy: `<strong>The microphone is busy or blocked.</strong> Close other apps that may be using it (Teams, Zoom, Discord),
      check it isn't disabled in Windows Sound settings, then try again.`,
    insecure: `<strong>Recording needs a secure page.</strong> Open the app at <code>http://127.0.0.1</code> on this machine,
      or start it with <code>--share</code> for an https link.`,
  };

  async function listDevices() {
    try {
      const devs = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'audioinput' && d.deviceId);
      const sel = $('device');
      sel.innerHTML = devs.map((d, i) => `<option value="${d.deviceId}">${(d.label || `Microphone ${i + 1}`).replace(/</g, '&lt;')}</option>`).join('');
      if (deviceId) sel.value = deviceId;
      $('devrow').hidden = devs.length < 2 && !!devs.length;
      if (!devs.length) $('devrow').hidden = true;
    } catch (_) { /* labels are a nicety */ }
  }
  $('device').addEventListener('change', (e) => { deviceId = e.target.value; });
  if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
    navigator.mediaDevices.addEventListener('devicechange', listDevices);
  }

  async function openMic() {
    if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      say(FIXES.insecure, 'err'); return null;
    }
    $('title').textContent = 'Waiting for permission…';
    $('sub').textContent = 'Your browser is asking to use the microphone. Choose Allow.';
    say('Look for the permission prompt near the address bar and choose <strong>Allow</strong>.', 'warn');
    const base = { channelCount: 1, echoCancellation: false, noiseSuppression: true, autoGainControl: true };
    try {
      const s = await navigator.mediaDevices.getUserMedia({ audio: deviceId ? { ...base, deviceId: { exact: deviceId } } : base });
      say('');
      return s;
    } catch (e) {
      const n = e && e.name;
      if (n === 'OverconstrainedError' && deviceId) { deviceId = ''; return openMic(); }
      say(n === 'NotAllowedError' || n === 'SecurityError' ? FIXES.denied
        : n === 'NotFoundError' ? FIXES.notfound
        : n === 'NotReadableError' || n === 'AbortError' ? FIXES.busy
        : `<strong>Could not open the microphone.</strong> ${n || 'Error'}: ${(e && e.message) || ''}`, 'err');
      return null;
    } finally {
      $('title').textContent = 'Record in the browser';
    }
  }

  function drawLive() {
    const w = canvas.width = canvas.clientWidth * devicePixelRatio, h = canvas.height = 64 * devicePixelRatio;
    const buf = new Float32Array(analyser.fftSize); analyser.getFloatTimeDomainData(buf);
    let pk = 0, sq = 0; for (const v of buf) { const a = Math.abs(v); if (a > pk) pk = a; sq += v * v; }
    const rms = Math.sqrt(sq / buf.length);
    peaks.push(pk); if (peaks.length > 160) peaks.shift();
    ctx2d.clearRect(0, 0, w, h);
    const g = ctx2d.createLinearGradient(0, 0, w, 0); g.addColorStop(0, '#ff6b5a'); g.addColorStop(1, '#ffb547');
    ctx2d.fillStyle = g;
    const bw = w / 160;
    peaks.forEach((p, i) => { const bh = Math.max(2 * devicePixelRatio, Math.min(1, p * 1.6) * (h - 8)); ctx2d.fillRect(i * bw, (h - bh) / 2, Math.max(1, bw - 2 * devicePixelRatio), bh); });
    const lvl = $('level'); lvl.hidden = false;
    const db = 20 * Math.log10(Math.max(rms, 1e-6));
    lvl.dataset.l = pk > 0.98 ? 'loud' : db < -42 ? 'quiet' : 'good';
    lvl.textContent = pk > 0.98 ? 'Too loud' : db < -42 ? 'Speak up' : 'Good level';
    raf = requestAnimationFrame(drawLive);
  }

  function drawStatic(samples) {
    const w = canvas.width = canvas.clientWidth * devicePixelRatio, h = canvas.height = 64 * devicePixelRatio;
    ctx2d.clearRect(0, 0, w, h);
    const g = ctx2d.createLinearGradient(0, 0, w, 0); g.addColorStop(0, '#ff6b5a'); g.addColorStop(1, '#ffb547');
    ctx2d.fillStyle = g;
    const bars = 160, step = Math.max(1, Math.floor(samples.length / bars)), bw = w / bars;
    for (let i = 0; i < bars; i++) {
      let pk = 0; for (let j = i * step; j < Math.min(samples.length, (i + 1) * step); j++) { const a = Math.abs(samples[j]); if (a > pk) pk = a; }
      const bh = Math.max(2 * devicePixelRatio, Math.min(1, pk * 1.6) * (h - 8));
      ctx2d.fillRect(i * bw, (h - bh) / 2, Math.max(1, bw - 2 * devicePixelRatio), bh);
    }
  }

  async function countIn() {
    const t = $('time');
    for (const n of [3, 2, 1]) {
      t.innerHTML = `<span class="tts-rec__count">${n}</span>`;
      await new Promise((r) => setTimeout(r, 650));
      if (state !== 'counting') return false;
    }
    return true;
  }

  async function start() {
    blob = null; peaks = []; chunks = [];
    stream = await openMic();
    if (!stream) { setState('idle'); return; }
    await listDevices();
    const track = stream.getAudioTracks()[0];
    if (track && track.getSettings) deviceId = track.getSettings().deviceId || deviceId;
    setState('counting');
    $('title').textContent = 'Get ready…';
    $('sub').textContent = 'Recording starts after the count.';
    if (!(await countIn())) return;

    ac = new (window.AudioContext || window.webkitAudioContext)();
    inRate = ac.sampleRate;
    src = ac.createMediaStreamSource(stream);
    analyser = ac.createAnalyser(); analyser.fftSize = 1024;
    proc = ac.createScriptProcessor(4096, 1, 1);
    proc.onaudioprocess = (e) => { if (state === 'recording') chunks.push(new Float32Array(e.inputBuffer.getChannelData(0))); };
    src.connect(analyser); src.connect(proc);
    const mute = ac.createGain(); mute.gain.value = 0; proc.connect(mute); mute.connect(ac.destination);

    setState('recording');
    $('title').textContent = 'Recording';
    $('sub').textContent = 'Speak naturally. Press stop when you are done.';
    say('');
    started = performance.now();
    drawLive();
    timer = setInterval(() => {
      const s = (performance.now() - started) / 1000;
      $('time').textContent = fmt(s);
      $('fill').style.width = `${Math.min(100, (s / MAX_S) * 100)}%`;
      if (s >= MAX_S) stop();
    }, 200);
  }

  function release() {
    cancelAnimationFrame(raf); clearInterval(timer);
    try { proc && proc.disconnect(); src && src.disconnect(); } catch (_) {}
    if (stream) stream.getTracks().forEach((t) => t.stop());   // turns the browser's mic indicator off
    if (ac) ac.close();
    stream = ac = src = proc = analyser = null;
  }

  function resample(input, from, to) {
    if (from === to) return input;
    const ratio = from / to, out = new Float32Array(Math.floor(input.length / ratio));
    for (let i = 0; i < out.length; i++) {
      // Box-filter over the source span: cheap anti-aliasing for a downsample.
      const a = i * ratio, b = Math.min(input.length, a + ratio);
      let s = 0, n = 0; for (let j = Math.floor(a); j < b; j++) { s += input[j]; n++; }
      out[i] = n ? s / n : 0;
    }
    return out;
  }

  function wav(samples, rate) {
    const buf = new ArrayBuffer(44 + samples.length * 2), v = new DataView(buf);
    const w = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
    w(0, 'RIFF'); v.setUint32(4, 36 + samples.length * 2, true); w(8, 'WAVE'); w(12, 'fmt ');
    v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
    v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
    w(36, 'data'); v.setUint32(40, samples.length * 2, true);
    for (let i = 0; i < samples.length; i++) { const x = Math.max(-1, Math.min(1, samples[i])); v.setInt16(44 + i * 2, x < 0 ? x * 0x8000 : x * 0x7fff, true); }
    return new Blob([buf], { type: 'audio/wav' });
  }

  function stop() {
    if (state === 'counting') { release(); setState('idle'); $('time').textContent = '0:00'; return; }
    if (state !== 'recording') return;
    const secs = (performance.now() - started) / 1000;
    release();
    const total = chunks.reduce((n, c) => n + c.length, 0), all = new Float32Array(total);
    let o = 0; for (const c of chunks) { all.set(c, o); o += c.length; }
    const pcm = resample(all, inRate, TARGET_SR);
    blob = wav(pcm, TARGET_SR);
    drawStatic(pcm);
    $('player').src = URL.createObjectURL(blob);
    $('level').hidden = true;
    setState('review');
    $('title').textContent = `Take ready · ${fmt(secs)}`;
    $('sub').textContent = 'Listen back. If it sounds clear, use it; otherwise record again.';
    let peak = 0; for (const x of pcm) { const a = Math.abs(x); if (a > peak) peak = a; }
    if (peak < 0.01) say(`<strong>That take is silent.</strong> The microphone may be muted, or Windows may be blocking it. ${FIXES.denied.replace(/^<strong>.*?<\/strong>/, '')}`, 'err');
    else if (secs < MIN_S) say(`<strong>Short take (${secs.toFixed(1)}s).</strong> 10&ndash;20 seconds clones best.`, 'warn');
    else if (secs > GOOD_S) say(`<strong>Long take.</strong> Anything over 30 seconds only slows encoding; it will be trimmed of silence.`, 'warn');
    else say('');
  }

  btn.addEventListener('click', () => { (state === 'recording' || state === 'counting') ? stop() : start(); });
  host.querySelector('[data-act="again"]').addEventListener('click', () => {
    setState('idle'); say(''); $('time').textContent = '0:00'; $('fill').style.width = '0';
    ctx2d.clearRect(0, 0, canvas.width, canvas.height); start();
  });
  host.querySelector('[data-act="use"]').addEventListener('click', async (e) => {
    if (!blob) return;
    const b = e.currentTarget; b.disabled = true; b.textContent = 'Sending…';
    try {
      const file = new File([blob], `recording_${Date.now()}.wav`, { type: 'audio/wav' });
      const res = await upload(file);
      props.value = res.path;
      trigger('change');
      say('<strong>Take added.</strong> It is now the clip below &mdash; check the quality report, add the transcript and save.', 'ok');
    } catch (err) {
      say(`<strong>Could not send the take to the app.</strong> ${(err && err.message) || err}`, 'err');
    } finally { b.disabled = false; b.textContent = 'Use this take'; }
  });

  // Space toggles recording when the recorder has focus.
  host.tabIndex = -1;
  host.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && e.target === host) { e.preventDefault(); btn.click(); }
  });
  listDevices();
})();

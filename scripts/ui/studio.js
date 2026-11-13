// App-wide behaviour, injected via launch(head=...). Gradio 6 inserts `head` twice,
// so this guards against running twice.
(() => {
if (window.__skbStudio) return;
window.__skbStudio = true;
const boot = () => {
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch (_) { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch (_) {} },
  };

  // Theme: remember an explicit choice; otherwise Gradio follows the OS.
  // Gradio sets its own `dark` class once it mounts, so re-apply an explicit
  // choice after that rather than racing it.
  const syncToggle = () => {
    document.querySelectorAll('[data-skb="theme"]').forEach((b) =>
      b.setAttribute('aria-pressed', String(document.body.classList.contains('dark'))));
  };
  const applyTheme = () => {
    const saved = store.get('skb-theme');
    if (saved === 'dark') document.body.classList.add('dark');
    if (saved === 'light') document.body.classList.remove('dark');
    syncToggle();
  };
  if (!document.documentElement.lang) document.documentElement.lang = 'en-ZA';
  applyTheme();
  [0, 400, 1500].forEach((ms) => setTimeout(applyTheme, ms));
  window.addEventListener('load', applyTheme);

  // The masthead is gr.HTML, rendered after this runs, so
  // listen at the document instead of binding to elements.
  document.addEventListener('click', (e) => {
    const t = e.target.closest('[data-skb]');
    if (!t) return;
    const act = t.dataset.skb;
    if (act === 'theme') {
      const dark = !document.body.classList.contains('dark');
      document.body.classList.toggle('dark', dark);
      store.set('skb-theme', dark ? 'dark' : 'light');
      syncToggle();
    }
  });

  // --- Hover/focus help. Any element with data-tip gets a tooltip; TIPS adds
  // data-tip to Gradio controls by their visible label.
  const TIPS = {
    'Steady': 'Calmer and more even. The best match for the 0.6B model.',
    'Natural': 'The model’s own delivery.',
    'Expressive': 'More varied and lively. Can wander on long lines.',
    'Accurate': 'The large speech model. Best with accents and names. About a minute a clip here.',
    'Fast': 'The small speech model. About four times quicker, a little less accurate.',
    'Both languages': 'Use the whole clip and the transcript that follows it in order.',
    'English only': 'Use only the English parts of the clip, with the English transcript.',
    'Afrikaans only': 'Use only the Afrikaans parts of the clip, with the Afrikaans transcript.',
    'Preview': 'Show how the text will be split and read. No audio is made, so it is instant.',
    'Stop': 'Stops after the line that is rendering now. Finished lines are kept.',
    'Refresh': 'Reload the list.',
    'Transcribe again': 'Write the clip out again with the options above.',
    'Create my voice': 'Learn the voice from this clip and save it on this computer.',
    'Use in Speak': 'Open the Speak tab with this voice selected.',
    'Download': 'Save this recording to your Downloads folder.',
    'RAM free': 'Memory free right now. A loaded 0.6B model uses about 3.6 GB, so this drops once it loads.',
    'Precision': 'float32 is the fast path on this computer. bfloat16 is emulated and much slower here.',
    'Speed': 'Measured from your own takes. Twelve frames make one second of audio.',
    'Torch threads': 'How many processor cores the voice model uses at once.',
    'Vector unit': 'The processor’s maths extensions. AVX2 is what this computer has.',
    'CPU': 'Physical cores and threads. Generation speed scales with cores and memory speed.',
  };
  const SIZE_TIPS = {
    '0.6B': 'Installed. Every voice in this app runs on the 0.6B models.',
    '1.7B': 'Not on this computer. A separate download of about 4.5 GB, and it needs about 3.8 GB of free memory while it runs.',
  };
  const tipFor = (el) => {
    const text = (el.textContent || '').trim();
    if (el.closest('.tts-seg') && /^(0\.6B|1\.7B)/.test(text)) {
      const size = text.slice(0, 4);
      return /not installed/.test(text) ? SIZE_TIPS['1.7B'] : SIZE_TIPS[size];
    }
    return TIPS[text];
  };
  const tip = document.createElement('div');
  tip.className = 'tts-tip';
  tip.id = 'tts-tip';
  tip.setAttribute('role', 'tooltip');
  document.body.appendChild(tip);
  let tipFor_ = null;
  const showTip = (el) => {
    const text = el.getAttribute('data-tip');
    if (!text) return;
    tipFor_ = el;
    tip.textContent = text;
    tip.classList.add('is-on');
    el.setAttribute('aria-describedby', 'tts-tip');
    const r = el.getBoundingClientRect();
    const w = tip.offsetWidth, h = tip.offsetHeight;
    let x = r.left + r.width / 2 - w / 2;
    x = Math.max(8, Math.min(x, window.innerWidth - w - 8));
    const above = r.top - h - 10;
    tip.style.left = `${x}px`;
    tip.style.top = `${above > 8 ? above : r.bottom + 10}px`;
    tip.dataset.side = above > 8 ? 'top' : 'bottom';
  };
  const hideTip = () => {
    if (tipFor_) tipFor_.removeAttribute('aria-describedby');
    tipFor_ = null;
    tip.classList.remove('is-on');
  };
  const tipTarget = (e) => e.target.closest && e.target.closest('[data-tip]');
  document.addEventListener('pointerover', (e) => { const t = tipTarget(e); if (t && t !== tipFor_) showTip(t); });
  document.addEventListener('pointerout', (e) => {
    const t = tipTarget(e);
    if (t && !t.contains(e.relatedTarget)) hideTip();
  });
  document.addEventListener('focusin', (e) => { const t = tipTarget(e); if (t) showTip(t); });
  document.addEventListener('focusout', hideTip);
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') hideTip(); });
  window.addEventListener('scroll', hideTip, { passive: true });

  const models = window.__skbModels || {};
  const setClass = (el, cls, on) => { if (el && el.classList.contains(cls) !== on) el.classList.toggle(cls, on); };

  // The Clone tab's coach names the card to do next (data-next) and the cards
  // that can't be used yet (data-wait).
  const applyCoach = () => {
    const c = document.querySelector('#tts-coach .tts-coach');
    if (!c) return;
    const pick = (attr) => new Set((c.getAttribute(attr) || '').split(',').filter(Boolean)
      .map((sel) => document.querySelector(sel)).filter(Boolean));
    const next = pick('data-next'), wait = pick('data-wait');
    ['#tts-clone-card', '#tts-transcript-card', '#tts-save-card', '#tts-try-card'].forEach((sel) => {
      const card = document.querySelector(sel);
      setClass(card, 'tts-next', next.has(card));
      setClass(card, 'tts-wait', wait.has(card));
      if (card) {
        const inert = wait.has(card);
        if (card.hasAttribute('aria-disabled') !== inert) {
          if (inert) card.setAttribute('aria-disabled', 'true'); else card.removeAttribute('aria-disabled');
        }
      }
    });
  };

  // A primary button that is disabled and says "…" is busy: spin it and light
  // up the card (and, in a split view, the result card beside it).
  const applyBusy = () => {
    document.querySelectorAll('button.tts-go').forEach((b) => {
      const busy = b.disabled && /…$/.test((b.textContent || '').trim());
      setClass(b, 'tts-busy', busy);
      const card = b.closest('.tts-card');
      setClass(card, 'tts-card--busy', busy);
      const split = b.closest('.tts-split');
      const out = split && split.querySelector('.tts-card--sticky');
      if (out && out !== card) setClass(out, 'tts-card--output', busy);
    });
  };

  // Model sizes this computer doesn't have: shown, but locked, with the reason.
  const applyLocks = () => {
    document.querySelectorAll('.tts-seg label').forEach((l) => {
      const off = /not installed/.test(l.textContent || '');
      setClass(l, 'tts-seg__off', off);
      const input = l.querySelector('input');
      if (input && off && !input.disabled) { input.disabled = true; l.setAttribute('aria-disabled', 'true'); }
    });
    if (models['VoiceDesign:1.7B'] === false) {
      const tab = [...document.querySelectorAll('.tts-tabs button[role="tab"]')]
        .find((b) => /design/i.test(b.textContent || ''));
      if (tab && !tab.querySelector('.tts-tabflag')) {
        const flag = document.createElement('span');
        flag.className = 'tts-tabflag';
        flag.textContent = '1.7B';
        tab.appendChild(flag);
        tab.setAttribute('data-tip', 'Needs the 1.7B Voice Design model, which isn’t on this computer.');
      }
    }
  };

  const applyTips = () => {
    document.querySelectorAll('.tts-seg label, button, .tts-hstat > span').forEach((el) => {
      const host = el.matches('.tts-hstat > span') ? el.parentElement : el;
      if (host.hasAttribute('data-tip')) return;
      const t = tipFor(el);
      if (t) host.setAttribute('data-tip', t);
    });
  };

  // Rotating tips: one at a time, a new one every few seconds.
  setInterval(() => {
    document.querySelectorAll('.tts-tips[data-tips]').forEach((box) => {
      if (box.offsetParent === null || box.matches(':hover')) return;
      let list;
      try { list = JSON.parse(box.getAttribute('data-tips')); } catch (_) { return; }
      const span = box.querySelector('span');
      if (!span || !list || list.length < 2) return;
      const i = ((parseInt(box.dataset.i || '0', 10) || 0) + 1) % list.length;
      box.dataset.i = String(i);
      span.classList.add('is-out');
      setTimeout(() => { span.innerHTML = list[i]; span.classList.remove('is-out'); }, 220);
    });
  }, 9000);

  // Gradio renders lazily: label late nodes (theme toggle state, polite status regions).
  let queued = false;
  const onDom = () => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => { queued = false; refresh(); });
  };
  const refresh = () => {
    applyCoach();
    applyBusy();
    applyLocks();
    applyTips();
    document.querySelectorAll('.tts-status:not([aria-live])').forEach((el) => {
      el.setAttribute('aria-live', 'polite');
    });
    // Gradio's accordion headers toggle an .open class but no ARIA state.
    document.querySelectorAll('.tts-pane > button.label-wrap').forEach((b) => {
      const open = String(b.classList.contains('open'));
      if (b.getAttribute('aria-expanded') !== open) b.setAttribute('aria-expanded', open);
    });
    syncToggle();
  };
  new MutationObserver(onDom).observe(document.body, {
    childList: true, subtree: true, characterData: true,
    attributes: true, attributeFilter: ['class', 'disabled'],
  });
  onDom();

  // Guided tour (driver.js). Auto-shows on the first MAX_AUTO visits until finished
  // or dismissed; "Take the tour" replays it.
  const MAX_AUTO = 3;
  const visits = (parseInt(store.get('skb-visits') || '0', 10) || 0) + 1;
  store.set('skb-visits', String(visits));

  const cloneTab = () => [...document.querySelectorAll('.tts-tabs button[role="tab"]')]
    .find((b) => /clone/i.test(b.textContent || ''));

  const startTour = () => {
    const make = window.driver && window.driver.js && window.driver.js.driver;
    if (!make) return;
    const tab = cloneTab();
    if (tab && tab.getAttribute('aria-selected') !== 'true') tab.click();
    let stopForGood = store.get('skb-tour') === 'done';
    const steps = [
      { popover: { title: 'Welcome to Skinnerbek Studio',
        description: 'Clone a voice from a short clip, then make it say anything. Everything runs on this computer. This tour takes about 30 seconds.' } },
      { element: '#tts-coach', popover: { title: 'This line always says what to do next', side: 'bottom', align: 'start',
        description: 'It follows you through making a voice. The card to use next glows, and steps that aren’t ready yet are greyed out until they are. Hover over any control for a short explanation.' } },
      { element: '#tts-drop', popover: { title: 'Start with a voice clip', side: 'bottom', align: 'start',
        description: 'Drop in an MP3, WAV, M4A or FLAC of someone talking. 10 to 20 seconds of clear speech is ideal. Or open “Record yourself” underneath to use your microphone.' } },
      { element: '#tts-transcript-card', popover: { title: 'The words are written for you', side: 'right', align: 'start',
        description: 'As soon as a clip arrives it is transcribed automatically. Words it was unsure of are highlighted, so you only check those. If English and Afrikaans are both spoken, each gets its own transcript and you choose which the voice learns from.' } },
      { element: '#tts-profile-card', popover: { title: 'See how the voice speaks', side: 'left', align: 'start',
        description: 'Pitch, pace, pauses and more, read from the clip. It sets the delivery and pauses Speak uses for this voice.' } },
      { element: '#tts-save-card', popover: { title: 'Name it and create it', side: 'right', align: 'start',
        description: 'Give the voice a name and press “Create my voice”. It is saved on this computer for next time.' } },
      { element: '#tts-try-card', popover: { title: 'Then hear it', side: 'left', align: 'start',
        description: 'Once your voice is created, type anything here and press Speak.' } },
      { element: '.tts-tabs [role="tablist"]', popover: { title: 'Everything else is up here', side: 'bottom', align: 'start',
        description: 'Speak is for longer scripts. My voices keeps every voice you have made. Recordings has everything you have generated.' } },
      { element: '[data-skb="theme"]', popover: { title: 'Day or night', side: 'left', align: 'start',
        description: 'Switch between light and dark here. “Take the tour” next to it plays this again whenever you like.' } },
    ].filter((s) => !s.element || document.querySelector(s.element));

    const tour = make({
      steps,
      animate: !matchMedia('(prefers-reduced-motion: reduce)').matches,
      smoothScroll: true,
      showProgress: true,
      progressText: '{{current}} of {{total}}',
      nextBtnText: 'Next',
      prevBtnText: 'Back',
      doneBtnText: 'Start cloning',
      popoverClass: 'skb-tour',
      overlayColor: '#0a0e18',
      overlayOpacity: 0.62,
      stagePadding: 6,
      stageRadius: 14,
      onPopoverRender: (popover) => {
        const row = document.createElement('div');
        row.className = 'skb-tour__skip';
        row.innerHTML = '<label><input type="checkbox"> Don’t show this again</label>' +
                        '<button type="button">Skip tour</button>';
        const box = row.querySelector('input');
        box.checked = stopForGood;
        box.addEventListener('change', () => { stopForGood = box.checked; });
        row.querySelector('button').addEventListener('click', () => tour.destroy());
        popover.wrapper.appendChild(row);
      },
      // Read "was this the last step?" before driver.js clears its state.
      onDestroyStarted: () => {
        if (stopForGood || !tour.hasNextStep()) store.set('skb-tour', 'done');
        tour.destroy();
      },
      onDestroyed: () => {
        const drop = document.querySelector('#tts-drop button, #tts-drop input');
        if (drop && drop.focus) drop.focus();
      },
    });
    tour.drive();
  };
  document.addEventListener('click', (e) => {
    if (e.target.closest('[data-skb="tour"]')) startTour();
  });
  if (store.get('skb-tour') !== 'done' && visits <= MAX_AUTO) {
    // Wait for Gradio to mount the landing tab before pointing at it.
    let tries = 0;
    const wait = setInterval(() => {
      tries += 1;
      if (document.querySelector('#tts-drop') || tries > 40) {
        clearInterval(wait);
        if (document.querySelector('#tts-drop')) setTimeout(startTour, 700);
      }
    }, 250);
  }

  // Ctrl/Cmd+Enter renders from whichever tab is showing.
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Enter' || !(e.ctrlKey || e.metaKey)) return;
    const ids = ['tts-speak-btn', 'tts-design-btn'];
    for (const id of ids) {
      const b = document.getElementById(id);
      if (b && b.offsetParent !== null && !b.disabled) { e.preventDefault(); b.click(); return; }
    }
  });
};
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
else boot();
})();

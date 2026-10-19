// App-wide behaviour, injected through launch(head=...) as a plain script.
// Gradio 6 inserts `head` twice (in the served HTML and again from the SPA,
// possibly after DOMContentLoaded), and launch(js=...) did not fire here, so
// this guards against running twice and starts whenever it lands.
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

  // Gradio renders lazily, so late-arriving nodes get the same treatment:
  // the theme toggle reports its state, and status slots become polite live regions (their content is what changes).
  const onDom = () => {
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
    childList: true, subtree: true, attributes: true, attributeFilter: ['class'],
  });
  onDom();

  // Guided tour (driver.js, vendored). Shows itself on the first few visits
  // until it is finished or someone ticks "Don't show this again"; after
  // MAX_AUTO visits it stops offering itself either way. "Take the tour" in
  // the header replays it any time.
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
      { element: '#tts-drop', popover: { title: 'Start with a voice clip', side: 'bottom', align: 'start',
        description: 'Drop in an MP3, WAV, M4A or FLAC of someone talking. 10 to 20 seconds of clear speech is ideal. Or open “Record yourself” underneath to use your microphone.' } },
      { element: '#tts-transcript-card', popover: { title: 'The words are written for you', side: 'right', align: 'start',
        description: 'As soon as a clip arrives it is transcribed automatically. Words it was unsure of are highlighted, so you only check those. You can edit anything.' } },
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

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

  // The masthead and welcome strip are gr.HTML, rendered after this runs, so
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
    } else if (act === 'welcome-close') {
      const w = t.closest('.tts-welcome');
      if (w) w.hidden = true;
      store.set('skb-welcome', 'done');
      // The button just vanished; put focus somewhere useful, not on <body>.
      const box = document.querySelector('.tts-card textarea');
      if (box) box.focus();
    }
  });

  // Gradio renders lazily, so late-arriving nodes get the same treatment:
  // dismissed welcome stays hidden, the theme toggle reports its state, and
  // status slots become polite live regions (their content is what changes).
  const onDom = () => {
    if (store.get('skb-welcome') === 'done') {
      document.querySelectorAll('.tts-welcome').forEach((w) => { w.hidden = true; });
    }
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

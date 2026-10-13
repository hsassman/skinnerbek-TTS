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
  const applyTheme = () => {
    const saved = store.get('skb-theme');
    if (saved === 'dark') document.body.classList.add('dark');
    if (saved === 'light') document.body.classList.remove('dark');
  };
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
    } else if (act === 'welcome-close') {
      const w = t.closest('.tts-welcome');
      if (w) w.hidden = true;
      store.set('skb-welcome', 'done');
    }
  });

  const hideWelcome = () => {
    if (store.get('skb-welcome') !== 'done') return;
    document.querySelectorAll('.tts-welcome').forEach((w) => { w.hidden = true; });
  };
  new MutationObserver(hideWelcome).observe(document.body, { childList: true, subtree: true });
  hideWelcome();

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

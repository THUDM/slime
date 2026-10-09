/* Read the existing Sphinx guides in place, keeping the builder and its history alive. */
(function() {
  'use strict';
  const dialog = document.getElementById('learn-dialog');
  if (!dialog) return;
  const content = document.getElementById('learn-content');
  const title = document.getElementById('learn-title');
  const zh = document.documentElement.lang.startsWith('zh');
  const base = new URL('./', location.href);
  let previousFocus, request = 0,
    mathReady, mermaidReady;
  const cache = new Map();

  function isGuide(url) {
    return url.origin === base.origin && url.pathname.startsWith(base.pathname) &&
      url.pathname.endsWith('.html') && !/\/(?:index|library|search|genindex)\.html$/.test(url.pathname);
  }

  async function typeset(element) {
    if (element.querySelector('.math')) {
      if (!mathReady) {
        mathReady = new Promise((resolve, reject) => {
          window.MathJax = {
            startup: {
              typeset: false
            },
            tex: {
              inlineMath: [
                ['\\(', '\\)']
              ]
            }
          };
          const script = document.createElement('script');
          script.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js';
          script.onload = () => resolve(window.MathJax.startup.promise);
          script.onerror = reject;
          document.head.appendChild(script);
        });
      }
      try {
        await mathReady;
        if (element.isConnected) await window.MathJax.typesetPromise([element]);
      } catch (_) {
        /* Keep readable TeX if the CDN cannot be reached. */
      }
    }
    if (element.querySelector('.mermaid')) {
      if (!mermaidReady) mermaidReady = import(
        'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs');
      try {
        const {
          default: mermaid
        } = await mermaidReady;
        mermaid.initialize({
          startOnLoad: false,
          securityLevel: 'strict',
          theme: 'neutral'
        });
        if (element.isConnected) await mermaid.run({
          nodes: element.querySelectorAll('.mermaid')
        });
      } catch (_) {
        /* The source diagram is still available. */
      }
    }
  }

  async function open(href, source, info) {
    const url = new URL(href, base);
    if (!isGuide(url)) return;
    const ticket = ++request;
    if (!dialog.open) {
      previousFocus = source || document.activeElement;
      dialog.showModal();
    }
    title.textContent = info?.title || (zh ? '页内阅读' : 'Read in place');
    content.replaceChildren();
    const status = document.createElement('p');
    status.setAttribute('role', 'status');
    status.textContent = zh ? '正在加载完整说明与推导…' : 'Loading the full guide and derivation…';
    content.appendChild(status);
    dialog.scrollTop = 0;
    const documentUrl = new URL(url);
    documentUrl.hash = '';
    try {
      let html = cache.get(documentUrl.href);
      if (!html) {
        const response = await fetch(documentUrl);
        if (!response.ok) throw new Error(response.status);
        html = await response.text();
        cache.set(documentUrl.href, html);
      }
      if (ticket !== request || !dialog.open) return;
      const doc = new DOMParser().parseFromString(html, 'text/html');
      const article = doc.querySelector('article.bd-article');
      if (!article) throw new Error('Missing article');
      const anchor = url.hash ? doc.getElementById(decodeURIComponent(url.hash.slice(1))) : null;
      const section = anchor?.closest('section') || article;
      const body = document.createElement('div');
      body.className = 'reader-article';
      body.innerHTML = section.innerHTML;
      body.querySelectorAll('script, .headerlink').forEach(node => node.remove());
      body.querySelectorAll('[href], [src]').forEach(node => {
        for (const attr of ['href', 'src']) {
          if (node.hasAttribute(attr)) node.setAttribute(attr, new URL(node.getAttribute(attr),
            documentUrl).href);
        }
        if (node.tagName === 'A' && new URL(node.href).origin !== base.origin) {
          node.target = '_blank';
          node.rel = 'noopener';
        }
      });
      title.textContent = info?.title || body.querySelector('h1, h2, h3')?.textContent.trim() || doc
      .title;
      const heading = body.querySelector('h1, h2, h3');
      if (heading?.textContent.trim() === title.textContent.trim()) heading.remove();
      const links = document.createElement('p');
      links.className = 'reader-sources';
      const full = document.createElement('a');
      full.href = url.href;
      full.target = '_blank';
      full.rel = 'noopener';
      full.textContent = zh ? '单独打开本节 ↗' : 'Open this section separately ↗';
      links.append(full);
      if (info?.paper) {
        const paper = document.createElement('a');
        paper.href = info.paper;
        paper.target = '_blank';
        paper.rel = 'noopener';
        paper.textContent = zh ? '原论文 / 技术文章 ↗' : 'Original paper / article ↗';
        links.append(paper);
      }
      content.replaceChildren(links);
      if (info?.text) {
        const summary = document.createElement('p');
        summary.className = 'reader-summary';
        summary.textContent = info.text;
        content.append(summary);
      }
      content.append(body);
      void typeset(body);
    } catch (_) {
      if (ticket !== request) return;
      status.textContent = zh ? '加载失败，请重试或直接打开原文。' :
        'Could not load the guide. Retry or open the original page.';
      const retry = document.createElement('button');
      retry.type = 'button';
      retry.textContent = zh ? '重试' : 'Retry';
      retry.onclick = () => open(href, source, info);
      const link = document.createElement('a');
      link.href = url.href;
      link.target = '_blank';
      link.rel = 'noopener';
      link.textContent = zh ? '打开原文 ↗' : 'Open guide ↗';
      content.append(retry, link);
    }
  }

  document.addEventListener('click', event => {
    const a = event.target.closest('a[href]');
    if (!a || a.target || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey || event
      .button !== 0) return;
    const url = new URL(a.href);
    if (a.closest('.reader-article') && url.origin === base.origin &&
      url.pathname === new URL('index.html', base).pathname) {
      event.preventDefault();
      dialog.close();
      document.getElementById('lab').scrollIntoView();
      return;
    }
    if (!isGuide(url)) return;
    event.preventDefault();
    if (a.closest('.reader-article') && url.hash) {
      const target = content.querySelector('#' + CSS.escape(decodeURIComponent(url.hash.slice(1))));
      if (target && new URL(a.href).pathname === new URL(content.querySelector('.reader-sources a')
          .href).pathname) {
        target.scrollIntoView({
          block: 'start'
        });
        return;
      }
    }
    void open(url, a);
  });
  document.getElementById('close-learn').addEventListener('click', () => dialog.close());
  dialog.addEventListener('close', () => {
    request++;
    previousFocus?.focus({
      preventScroll: true
    });
  });
  window.SlimeReader = {
    open
  };
})();

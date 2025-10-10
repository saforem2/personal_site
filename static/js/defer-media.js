const initDeferredMedia = () => {
  const lazyNodes = document.querySelectorAll('[data-defer-src]');
  if (!lazyNodes.length) {
    return;
  }

  const loadNode = (node) => {
    if (!node || !node.dataset.deferSrc) {
      return;
    }
    const src = node.dataset.deferSrc;
    const srcset = node.dataset.deferSrcset;
    if (srcset) {
      node.setAttribute('srcset', srcset);
      node.removeAttribute('data-defer-srcset');
    }
    if (node.tagName === 'SOURCE' && !node.hasAttribute('srcset')) {
      node.setAttribute('srcset', src);
    } else if (node.tagName === 'LINK') {
      node.setAttribute('href', src);
    } else {
      node.setAttribute(node.dataset.deferAttr || 'src', src);
    }
    if ('loading' in node && !node.getAttribute('loading')) {
      node.setAttribute('loading', 'lazy');
    }
    node.removeAttribute('data-defer-src');
  };

  const maybeLoadChildren = (root) => {
    root.querySelectorAll('[data-defer-src]').forEach((child) => {
      loadNode(child);
    });
  };

  const onIntersection = (entries, observer) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting || entry.intersectionRatio > 0) {
        const target = entry.target;
        observer.unobserve(target);
        loadNode(target);
        if (target.tagName === 'DETAILS' && target.open) {
          maybeLoadChildren(target);
        }
      }
    });
  };

  const observerOptions = {
    rootMargin: '200px 0px',
    threshold: 0.01,
  };

  let observer;
  if ('IntersectionObserver' in window) {
    observer = new IntersectionObserver(onIntersection, observerOptions);
  }

  lazyNodes.forEach((node) => {
    if (observer) {
      observer.observe(node);
    } else {
      loadNode(node);
    }
  });

  document.querySelectorAll('details').forEach((detailsEl) => {
    detailsEl.addEventListener('toggle', () => {
      if (detailsEl.open) {
        maybeLoadChildren(detailsEl);
      }
    });
  });
};

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initDeferredMedia, { once: true });
} else {
  initDeferredMedia();
}

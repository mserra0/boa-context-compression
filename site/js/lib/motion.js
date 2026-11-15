const REDUCED_MOTION = "(prefers-reduced-motion: reduce)";

export function prefersReducedMotion() {
  return typeof window !== "undefined" && typeof window.matchMedia === "function"
    && window.matchMedia(REDUCED_MOTION).matches;
}

// Calls `enter` when the root scrolls into view and `leave` when it leaves,
// so a figure can play its timeline on screen and pause it off screen.
export function onVisible(root, {enter = () => {}, leave = () => {}} = {}) {
  if (typeof IntersectionObserver === "undefined") {
    enter();
    return;
  }
  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (entry.isIntersecting) enter();
      else leave();
    }
  }, {threshold: 0.15});
  observer.observe(root);
}

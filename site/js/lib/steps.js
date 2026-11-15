// The active step of a scroll-driven sequence: the number of steps whose start scroll position
// has been reached, at least 1, so the first step is shown before the reader gets to it.
export function stepAt(starts, scroll) {
  const reached = starts.filter((start) => scroll >= start).length;
  return Math.max(1, reached);
}

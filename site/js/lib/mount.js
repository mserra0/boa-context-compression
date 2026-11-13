// Mounts each figure module into its root. One failing figure never stops the others.
export function mountFigures(figures, data, {getRoot, onError}) {
  const mounted = [];
  for (const {id, mount} of figures) {
    const root = getRoot(id);
    if (!root) continue;
    try {
      mount(root, data);
      mounted.push(id);
    } catch (error) {
      onError(id, error);
    }
  }
  return mounted;
}

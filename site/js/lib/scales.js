export function linearScale([d0, d1], [r0, r1]) {
  return (x) => r0 + ((x - d0) / (d1 - d0)) * (r1 - r0);
}

export function logScale([d0, d1], [r0, r1]) {
  const l0 = Math.log10(d0);
  const l1 = Math.log10(d1);
  return (x) => r0 + ((Math.log10(x) - l0) / (l1 - l0)) * (r1 - r0);
}

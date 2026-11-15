export class DataLoadError extends Error {
  constructor(url, cause) {
    super(`Could not load ${url}`);
    this.name = "DataLoadError";
    this.url = url;
    if (cause) this.cause = cause;
  }
}

async function fetchJson(url) {
  let response;
  try {
    response = await fetch(url);
  } catch (err) {
    throw new DataLoadError(url, err);
  }
  if (!response.ok) throw new DataLoadError(url);
  try {
    return await response.json();
  } catch (err) {
    throw new DataLoadError(url, err);
  }
}

export async function loadData(baseUrl = "data/") {
  const [results, latency] = await Promise.all([
    fetchJson(`${baseUrl}results.json`),
    fetchJson(`${baseUrl}latency.json`),
  ]);
  return { results, latency };
}

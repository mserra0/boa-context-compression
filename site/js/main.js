import { loadData } from "./lib/data.js";
import { mountFigures } from "./lib/mount.js";

// Figure modules register here as {id, mount}, where id is the <figure> id in index.html.
const FIGURE_MODULES = [];

const LOAD_FAILED = "The data for this figure could not be loaded. Try reloading the page.";
const MOUNT_FAILED = "This figure could not be drawn.";

function setFallback(figure, text) {
  const fallback = figure?.querySelector(".figure__fallback");
  if (fallback) fallback.textContent = text;
}

function cell(tag, text, attrs = {}) {
  const node = document.createElement(tag);
  node.textContent = text;
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  return node;
}

function scoreCell(value, isBest) {
  const td = document.createElement("td");
  const text = value.toFixed(2);
  if (isBest) td.append(cell("strong", text));
  else td.textContent = text;
  return td;
}

// The references do not depend on the ratio; repeating them under every ratio keeps them
// next to the scores they bound.
function referenceRow(label, reference, ratios, metrics) {
  const row = document.createElement("tr");
  row.append(cell("th", label, {scope: "row"}));
  for (let i = 0; i < ratios.length; i++) {
    for (const [key] of metrics) row.append(cell("td", reference[key].toFixed(2)));
  }
  return row;
}

// Fills one table from a split of results.json: ratios as column groups, methods as rows,
// the full-context and no-context references in the footer.
function renderResultsTable(table, results, split) {
  const {ratios, methods} = results;
  const metrics = [["em", "EM"], ["f1", "F1"]];

  const groupRow = document.createElement("tr");
  groupRow.append(cell("td", ""));
  for (const ratio of ratios) {
    groupRow.append(cell("th", `${ratio}×`, {colspan: String(metrics.length), scope: "colgroup", class: "data-table__group"}));
  }
  const metricRow = document.createElement("tr");
  metricRow.append(cell("th", "Method", {scope: "col"}));
  for (let i = 0; i < ratios.length; i++) {
    for (const [, label] of metrics) metricRow.append(cell("th", label, {scope: "col"}));
  }
  const thead = document.createElement("thead");
  thead.append(groupRow, metricRow);

  const best = new Map();
  for (const ratio of ratios) {
    for (const [key] of metrics) {
      const values = methods.map(({id}) => split.byRatio[ratio][id][key]);
      best.set(`${ratio}-${key}`, Math.max(...values));
    }
  }

  const tbody = document.createElement("tbody");
  for (const {id, label} of methods) {
    const row = document.createElement("tr");
    row.append(cell("th", label, {scope: "row"}));
    for (const ratio of ratios) {
      for (const [key] of metrics) {
        const value = split.byRatio[ratio][id][key];
        row.append(scoreCell(value, value === best.get(`${ratio}-${key}`)));
      }
    }
    tbody.append(row);
  }

  const tfoot = document.createElement("tfoot");
  tfoot.append(
    referenceRow("Full context", split.full, ratios, metrics),
    referenceRow("No context", split.none, ratios, metrics),
  );

  table.append(thead, tbody, tfoot);
}

function renderResults(results) {
  for (const node of document.querySelectorAll("[data-results-decoder]")) node.textContent = results.decoder;
  const inDomain = document.getElementById("table-in-domain");
  const outOfDomain = document.getElementById("table-out-of-domain");
  if (inDomain) renderResultsTable(inDomain, results, results.inDomain);
  if (outOfDomain) renderResultsTable(outOfDomain, results, results.outOfDomain);
}

function selectContents(node) {
  const range = document.createRange();
  range.selectNodeContents(node);
  const selection = window.getSelection();
  selection.removeAllRanges();
  selection.addRange(range);
}

function wireCopyButtons() {
  for (const button of document.querySelectorAll("[data-copy-target]")) {
    const source = document.getElementById(button.dataset.copyTarget);
    const status = button.parentElement.querySelector("[data-copy-status]");
    if (!source) continue;
    button.addEventListener("click", async () => {
      const report = (text) => { if (status) status.textContent = text; };
      try {
        if (!navigator.clipboard?.writeText) throw new Error("Clipboard API unavailable");
        await navigator.clipboard.writeText(source.textContent);
        report("Copied");
      } catch {
        selectContents(source);
        report("Selected, press Ctrl+C or ⌘C to copy");
      }
    });
  }
}

function showLoadFailure() {
  for (const fallback of document.querySelectorAll(".figure__fallback")) fallback.textContent = LOAD_FAILED;
  const status = document.querySelector("[data-results-status]");
  if (status) {
    status.textContent = "The results could not be loaded. Try reloading the page.";
    status.hidden = false;
  }
}

async function start() {
  if (window.gsap && window.ScrollTrigger) window.gsap.registerPlugin(window.ScrollTrigger);
  wireCopyButtons();

  let data;
  try {
    data = await loadData();
  } catch (error) {
    console.error(error);
    showLoadFailure();
    return;
  }

  renderResults(data.results);
  mountFigures(FIGURE_MODULES, data, {
    getRoot: (id) => document.getElementById(id),
    onError: (id, error) => {
      console.error(`Figure ${id} failed to mount`, error);
      setFallback(document.getElementById(id), MOUNT_FAILED);
    },
  });
}

start();

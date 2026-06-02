const form = document.getElementById("scholarForm");
const answerBox = document.getElementById("answerBox");
const searchCommandBox = document.getElementById("searchCommand");
const resultsContainer = document.getElementById("resultsContainer");
const toolCycles = document.getElementById("toolCycles");
const messageCount = document.getElementById("messageCount");
const latencyDisplay = document.getElementById("latency");
const spinner = document.getElementById("spinner");
const pipelineSteps = Array.from(document.querySelectorAll(".pipeline-steps .step"));

const setLoading = (isLoading) => {
  spinner.classList.toggle("hidden", !isLoading);
  answerBox.textContent = isLoading ? "ScholarAgent is thinking..." : answerBox.textContent;
};

const highlightStep = (step) => {
  pipelineSteps.forEach((node, index) => {
    node.classList.toggle("active", index === step);
  });
};

const renderMatch = (match) => {
  const article = document.createElement("article");
  article.className = "result-card";

  const title = document.createElement("h3");
  if (match.url) {
    const link = document.createElement("a");
    link.href = match.url;
    link.target = "_blank";
    link.rel = "noreferrer";
    link.textContent = match.name;
    title.appendChild(link);
  } else {
    title.textContent = match.name;
  }

  const head = document.createElement("div");
  head.className = "result-head";
  const meta = document.createElement("div");
  meta.className = "result-meta";

  const scoreBadge = document.createElement("span");
  scoreBadge.className = "result-badge";
  scoreBadge.textContent = `Score ${Number(match.match_score || 0).toFixed(1)}%`;

  const residencyBadge = document.createElement("span");
  residencyBadge.className = "result-badge";
  residencyBadge.textContent = `Residency: ${match.required_residency || "Any"}`;

  meta.append(scoreBadge, residencyBadge);
  if (match.min_gpa != null && Number(match.min_gpa) > 0) {
    const gpaBadge = document.createElement("span");
    gpaBadge.className = "result-badge";
    gpaBadge.textContent = `Minimum GPA: ${Number(match.min_gpa).toFixed(2)}`;
    meta.append(gpaBadge);
  }
  head.append(title, meta);

  const summary = document.createElement("p");
  summary.textContent = match.summary || match.why_relevant || "No summary available.";

  article.append(head, summary);

  if (match.details) {
    const details = document.createElement("p");
    details.className = "result-meta";
    details.textContent = match.details;
    article.appendChild(details);
  }

  if (match.url) {
    const linkDetails = document.createElement("p");
    linkDetails.className = "result-meta";
    linkDetails.innerHTML = `Website: <a href="${match.url}" target="_blank" rel="noreferrer">${match.url}</a>`;
    article.appendChild(linkDetails);
  }

  if (match.evidence) {
    const evidence = document.createElement("p");
    evidence.className = "result-meta";
    evidence.textContent = `Evidence: ${match.evidence}`;
    article.appendChild(evidence);
  }

  return article;
};

const renderResults = (matches) => {
  resultsContainer.innerHTML = "";
  if (!Array.isArray(matches) || matches.length === 0) {
    const noMatches = document.createElement("div");
    noMatches.textContent = "No grounded scholarships were found. Try a broader prompt or switch to university-only mode.";
    noMatches.style.color = "#c7d0ff";
    resultsContainer.appendChild(noMatches);
    return;
  }

  matches.forEach((match) => {
    resultsContainer.appendChild(renderMatch(match));
  });
};

const updateSearchCommand = (metadata = {}) => {
  const command = metadata.search_command || "";
  const source = metadata.search_source ? ` via ${metadata.search_source}` : "";
  if (command) {
    searchCommandBox.textContent = `Searching${source}: ${command}`;
    searchCommandBox.classList.remove("hidden");
    return;
  }
  searchCommandBox.classList.add("hidden");
};

const updateDiagnostics = (metadata = {}) => {
  toolCycles.textContent = metadata.tool_calls_made ?? 0;
  messageCount.textContent = metadata.total_messages ?? 0;
  latencyDisplay.textContent = `${Number(metadata.latency_ms ?? 0).toFixed(1)}ms`;
  document.getElementById("diagnostics").classList.remove("hidden");
};

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  highlightStep(0);
  setLoading(true);
  resultsContainer.innerHTML = "";
  document.getElementById("diagnostics").classList.add("hidden");

  const prompt = document.getElementById("prompt").value.trim();
  const gpa = parseFloat(document.getElementById("gpa").value) || 0;
  const major = document.getElementById("major").value.trim();
  const residency = document.getElementById("residency").value.trim();
  const interests = document
    .getElementById("interests")
    .value.split(",")
    .map((item) => item.trim())
    .filter(Boolean);
  const searchMode = document.getElementById("searchMode").value;

  answerBox.textContent = "Planning scholarship search...";

  try {
    highlightStep(1);
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        prompt,
        student_profile: { gpa, major, residency, interests },
        search_mode: searchMode,
      }),
    });

    highlightStep(2);
    const payload = await response.json();
    setLoading(false);

    if (!response.ok) {
      answerBox.textContent = payload.detail || "The ScholarAgent backend returned an error.";
      return;
    }

    answerBox.textContent = payload.answer || "No summary returned.";
    renderResults(payload.matches || []);
    updateSearchCommand(payload.metadata || {});
    updateDiagnostics(payload.metadata || {});
    highlightStep(payload.matches && payload.matches.length ? 4 : 3);
  } catch (error) {
    console.error(error);
    answerBox.textContent = "Unable to contact the ScholarAgent backend. Refresh and try again.";
    setLoading(false);
    highlightStep(0);
  }
});

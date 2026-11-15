/* DSoM Explorer. Native browser modules only; all graph facts come from data/demo.json. */
"use strict";

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const escapeHtml = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (char) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        char
      ],
  );
const svgNS = "http://www.w3.org/2000/svg";
const kindLabels = {
  animal: "Động vật",
  object: "Đồ vật",
  plant: "Thực vật",
  concept: "Khái niệm",
};
const colors = {
  animal: "#11766c",
  object: "#6684b2",
  plant: "#cb8659",
  concept: "#9b87ab",
};
const storageKey = "dsom-explorer:WN9:quiz:1";
let data, entities, relations;
let selectedId = "n02112826";
let currentQuestion = 0,
  currentCase = 0,
  selectedOption = null,
  activeHint = null;
let answers = {};
let explanationOpen = false,
  selectedEvidenceId = "answer";
let graphMode = "overview",
  view = { x: 0, y: 0, scale: 1 };
let graphDragging = false;

function entityName(id) {
  return entities.get(id)?.label || id.replaceAll("_", " ");
}
function relationName(id) {
  return relations.get(id) || id;
}
function initial(entity) {
  return [...entity.label][0].toUpperCase();
}
function imageMarkup(entity, className = "") {
  return entity.image
    ? `<img class="${className} ${entity.kind === "person" ? "person-photo" : ""}" src="${escapeHtml(entity.image)}" alt="Ảnh trong dataset của ${escapeHtml(entity.label)}" loading="lazy" width="440" height="320">`
    : `<div class="${className} entity-placeholder" aria-label="${entity.imageAvailable ? "Ảnh nguồn chưa được đóng gói" : "Nguồn thiếu ảnh"}">${escapeHtml(initial(entity))}</div>`;
}
function neighbors(id) {
  return data.edges.filter((edge) => edge.source === id || edge.target === id);
}
function otherEnd(edge, id) {
  return edge.source === id ? edge.target : edge.source;
}

function renderInspector() {
  const entity = entities.get(selectedId);
  const linked = neighbors(selectedId);
  $("#entity-inspector").innerHTML = `
    <div class="entity-type"><span class="status-dot" style="background:${colors[entity.kind]}"></span>${escapeHtml(kindLabels[entity.kind])}</div>
    ${imageMarkup(entity, "entity-photo")}
    <h3>${escapeHtml(entity.label)}</h3><p class="entity-synset">${escapeHtml(entity.id)}</p>
    ${entity.description ? `<p class="entity-description" lang="en">${escapeHtml(entity.description)}</p>` : '<p class="missing-note">Nguồn WN9 không có mô tả cho thực thể này.</p>'}
    <a class="entity-source" href="${escapeHtml(entity.sourceUrl)}" target="_blank" rel="noopener noreferrer">Bản ghi định danh nguồn ↗</a>
    <hr class="inspector-divider"><p class="neighbors-title">${linked.length} LIÊN KẾT TRAIN TRONG LÁT CẮT</p>
    <div class="neighbor-list">${
      linked
        .slice(0, 5)
        .map(
          (edge) =>
            `<button type="button" data-entity="${escapeHtml(otherEnd(edge, selectedId))}"><small>${edge.source === selectedId ? "→" : "←"} ${escapeHtml(relationName(edge.relation))}</small>${escapeHtml(entityName(otherEnd(edge, selectedId)))}</button>`,
        )
        .join("") ||
      '<p class="muted" style="font-size:14px">Chưa có cạnh train trong lát cắt đã chọn.</p>'
    }</div>`;
}

function hash(string) {
  let n = 2166136261;
  for (const char of string) n = Math.imul(n ^ char.charCodeAt(0), 16777619);
  return n >>> 0;
}

function calculateLayout(nodes, edges) {
  const nodeMap = new Map(
    nodes.map((node, index) => {
      const angle = (hash(node.id) % 6283) / 1000;
      const radius = 80 + (hash(node.id + "radius") % 150);
      return [
        node.id,
        {
          ...node,
          x: 410 + Math.cos(angle) * radius,
          y: 260 + Math.sin(angle) * radius * 0.75,
          vx: 0,
          vy: 0,
          degree: 0,
          index,
        },
      ];
    }),
  );
  edges.forEach((edge) => {
    nodeMap.get(edge.source).degree++;
    nodeMap.get(edge.target).degree++;
  });
  const points = [...nodeMap.values()];
  const isLocal = graphMode === "focus";
  for (let step = 0; step < 230; step++) {
    const cooling = 1 - step / 280;
    for (let i = 0; i < points.length; i++) {
      const a = points[i];
      for (let j = i + 1; j < points.length; j++) {
        const b = points[j];
        let dx = a.x - b.x,
          dy = a.y - b.y;
        if (dx === 0 && dy === 0) dx = 0.1;
        const d2 = Math.max(dx * dx + dy * dy, 70);
        const force = (isLocal ? 110 : 42) / d2;
        a.vx += dx * force;
        a.vy += dy * force;
        b.vx -= dx * force;
        b.vy -= dy * force;
      }
    }
    edges.forEach((edge) => {
      const a = nodeMap.get(edge.source),
        b = nodeMap.get(edge.target);
      const dx = b.x - a.x,
        dy = b.y - a.y;
      const distance = Math.max(1, Math.hypot(dx, dy));
      const force = (distance - (isLocal ? 110 : 66)) * 0.032;
      a.vx += (dx / distance) * force;
      a.vy += (dy / distance) * force;
      b.vx -= (dx / distance) * force;
      b.vy -= (dy / distance) * force;
    });
    points.forEach((point) => {
      point.vx += (410 - point.x) * 0.02;
      point.vy += (260 - point.y) * 0.045;
      if (isLocal && point.id === selectedId) {
        point.vx += (410 - point.x) * 0.08;
        point.vy += (260 - point.y) * 0.08;
      }
      point.x += point.vx * cooling;
      point.y += point.vy * cooling;
      point.vx *= 0.65;
      point.vy *= 0.65;
    });
  }
  // Fit the completed layout, rather than clamping nodes onto the frame edges.
  if (points.length > 1) {
    const minX = Math.min(...points.map((point) => point.x));
    const maxX = Math.max(...points.map((point) => point.x));
    const minY = Math.min(...points.map((point) => point.y));
    const maxY = Math.max(...points.map((point) => point.y));
    const scale = Math.min(
      700 / Math.max(1, maxX - minX),
      420 / Math.max(1, maxY - minY),
      1.8,
    );
    points.forEach((point) => {
      point.x = 410 + (point.x - (minX + maxX) / 2) * scale;
      point.y = 260 + (point.y - (minY + maxY) / 2) * scale;
    });
  }
  return points;
}

function makeSvg(tag, attributes) {
  const element = document.createElementNS(svgNS, tag);
  Object.entries(attributes).forEach(([key, value]) =>
    element.setAttribute(key, value),
  );
  return element;
}

function renderGraph() {
  const filter = $("#relation-filter").value;
  let edges = data.edges.filter(
    (edge) => filter === "all" || edge.relation === filter,
  );
  let ids;
  if (graphMode === "focus") {
    ids = new Set([selectedId]);
    edges
      .filter(
        (edge) => edge.source === selectedId || edge.target === selectedId,
      )
      .slice(0, 20)
      .forEach((edge) => {
        ids.add(edge.source);
        ids.add(edge.target);
      });
    const firstRing = new Set(ids);
    for (const edge of edges) {
      if (ids.size >= 32) break;
      if (firstRing.has(edge.source) || firstRing.has(edge.target)) {
        ids.add(edge.source);
        ids.add(edge.target);
      }
    }
    edges = edges.filter(
      (edge) => ids.has(edge.source) && ids.has(edge.target),
    );
  } else if (filter !== "all") {
    ids = new Set(edges.flatMap((edge) => [edge.source, edge.target]));
  } else ids = new Set(data.entities.map((entity) => entity.id));
  const points = calculateLayout(
    data.entities.filter((entity) => ids.has(entity.id)),
    edges,
  );
  const positions = new Map(points.map((point) => [point.id, point]));
  const directIds = new Set(
    neighbors(selectedId).map((edge) => otherEnd(edge, selectedId)),
  );
  const root = $("#graph-viewport");
  root.replaceChildren();
  const edgeLayer = makeSvg("g", { "aria-hidden": "true" });
  const shownRelations = new Set();
  edges.forEach((edge) => {
    const source = positions.get(edge.source),
      target = positions.get(edge.target);
    const connected = edge.source === selectedId || edge.target === selectedId;
    edgeLayer.append(
      makeSvg("line", {
        x1: source.x,
        y1: source.y,
        x2: target.x,
        y2: target.y,
        class: `graph-edge ${connected ? "connected" : ""}`,
        "marker-end": "url(#arrow)",
      }),
    );
    if (
      connected &&
      (graphMode === "focus" || directIds.size < 5) &&
      (graphMode === "focus" || !shownRelations.has(edge.relation))
    ) {
      shownRelations.add(edge.relation);
      const label = makeSvg("text", {
        x: (source.x + target.x) / 2,
        y: (source.y + target.y) / 2 - 6,
        class: "edge-label",
        "text-anchor": "middle",
      });
      label.textContent = edge.relation;
      edgeLayer.append(label);
    }
  });
  root.append(edgeLayer);
  const highestDegree = new Set(
    [...points]
      .sort((a, b) => b.degree - a.degree)
      .slice(0, 8)
      .map((point) => point.id),
  );
  const keyboardId = positions.has(selectedId) ? selectedId : points[0]?.id;
  points
    .sort((a, b) => Number(a.id === selectedId) - Number(b.id === selectedId))
    .forEach((point) => {
      const selected = point.id === selectedId;
      const radius = selected ? 12 : Math.min(9, 4.5 + point.degree * 0.5);
      const group = makeSvg("g", {
        class: "graph-node",
        transform: `translate(${point.x},${point.y})`,
        role: "button",
        tabindex: point.id === keyboardId ? "0" : "-1",
        "aria-label": `${point.label}, ${point.degree} liên kết train`,
        "data-node-id": point.id,
      });
      const title = makeSvg("title", {});
      title.textContent = point.label;
      group.append(title);
      if (selected)
        group.append(makeSvg("circle", { r: 19, class: "selected-ring" }));
      group.append(
        makeSvg("circle", {
          r: radius,
          fill: colors[point.kind] || colors.concept,
        }),
      );
      const visible =
        selected ||
        directIds.has(point.id) ||
        highestDegree.has(point.id) ||
        graphMode === "focus";
      group.dataset.labelPriority = selected
        ? "selected"
        : directIds.has(point.id) || graphMode === "focus"
          ? "neighbor"
          : "context";
      group.dataset.labelVisible = visible ? "1" : "0";
      group.dataset.labelCandidate = visible ? "1" : "0";
      group.dataset.positionX = point.x;
      group.dataset.radius = radius;
      const label = makeSvg("text", {
        x: 0,
        y: radius + 15,
        "text-anchor": "middle",
        opacity: visible ? 1 : 0,
      });
      label.textContent =
        point.label.length > 25 ? point.label.slice(0, 23) + "…" : point.label;
      group.append(label);
      group.addEventListener("pointerenter", () =>
        label.setAttribute("opacity", "1"),
      );
      group.addEventListener("pointerleave", () =>
        label.setAttribute("opacity", group.dataset.labelVisible),
      );
      group.addEventListener("focus", () => label.setAttribute("opacity", "1"));
      group.addEventListener("blur", () =>
        label.setAttribute("opacity", group.dataset.labelVisible),
      );
      group.addEventListener("click", () => {
        if (!graphDragging) selectEntity(point.id);
      });
      group.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          selectEntity(point.id);
          $$(".graph-node")
            .find((node) => node.dataset.nodeId === point.id)
            ?.focus();
        } else if (
          ["ArrowRight", "ArrowDown", "ArrowLeft", "ArrowUp"].includes(
            event.key,
          )
        ) {
          event.preventDefault();
          const nodes = $$(".graph-node"),
            index = nodes.indexOf(group);
          const direction = ["ArrowRight", "ArrowDown"].includes(event.key)
            ? 1
            : -1;
          group.setAttribute("tabindex", "-1");
          const next = nodes[(index + direction + nodes.length) % nodes.length];
          next.setAttribute("tabindex", "0");
          next.focus();
        }
      });
      root.append(group);
    });
  $("#graph-count").textContent =
    `${points.length} thực thể · ${edges.length} cạnh`;
  $("#graph-empty").hidden = edges.length > 0;
  $("#overview-button").setAttribute(
    "aria-pressed",
    String(graphMode === "overview"),
  );
  view = { x: 0, y: 0, scale: 1 };
  updateViewport();
  syncGraphTypography();
}

function syncGraphTypography() {
  const graph = $("#knowledge-graph");
  const { width, height } = graph.getBoundingClientRect();
  if (!width || !height) return;
  // SVG viewBox scaling must not shrink labels below their CSS-pixel size.
  const textScale = 1 / Math.min(width / 820, height / 540);
  graph.style.setProperty("--graph-text-scale", textScale);
  const priorities = { selected: 0, neighbor: 1, context: 2 };
  const occupied = [];
  $$(".graph-node", graph)
    .sort(
      (a, b) =>
        priorities[a.dataset.labelPriority] -
        priorities[b.dataset.labelPriority],
    )
    .forEach((node) => {
      const label = $("text", node);
      const priority = node.dataset.labelPriority;
      let visible =
        node.dataset.labelCandidate === "1" &&
        (priority === "selected" ||
          (priority === "neighbor" && width >= 500) ||
          (priority === "context" && width >= 700));
      label.setAttribute("y", Number(node.dataset.radius) + 19 * textScale);
      const halfWidth = label.getBBox().width / 2;
      const x = Number(node.dataset.positionX);
      label.setAttribute(
        "text-anchor",
        x + halfWidth > 800 ? "end" : x - halfWidth < 20 ? "start" : "middle",
      );
      if (visible) {
        const box = label.getBoundingClientRect();
        visible = !occupied.some(
          (other) =>
            box.left < other.right + 6 &&
            box.right + 6 > other.left &&
            box.top < other.bottom + 4 &&
            box.bottom + 4 > other.top,
        );
        if (visible) occupied.push(box);
      }
      node.dataset.labelVisible = visible ? "1" : "0";
      label.setAttribute("opacity", node.dataset.labelVisible);
    });
}

function selectEntity(id, scroll = false) {
  if (!entities.has(id)) return;
  selectedId = id;
  graphMode = "focus";
  $("#relation-filter").value = "all";
  $("#search-results").hidden = true;
  $("#entity-search").value = "";
  renderInspector();
  renderGraph();
  if (scroll) $("#explore").scrollIntoView({ behavior: "smooth" });
}

function updateViewport() {
  $("#graph-viewport").setAttribute(
    "transform",
    `translate(${view.x},${view.y}) scale(${view.scale})`,
  );
}
function changeZoom(factor) {
  const next = Math.max(0.55, Math.min(3, view.scale * factor)),
    ratio = next / view.scale;
  view.x = 410 - (410 - view.x) * ratio;
  view.y = 270 - (270 - view.y) * ratio;
  view.scale = next;
  updateViewport();
}
function bindGraph() {
  new ResizeObserver(syncGraphTypography).observe($("#knowledge-graph"));
  $("#entity-inspector").addEventListener("click", (event) => {
    const button = event.target.closest("[data-entity]");
    if (button) selectEntity(button.dataset.entity);
  });
  $("#relation-filter").addEventListener("change", () => {
    graphMode = "overview";
    renderGraph();
  });
  $("#overview-button").addEventListener("click", () => {
    graphMode = "overview";
    $("#relation-filter").value = "all";
    renderGraph();
  });
  $("#zoom-in").addEventListener("click", () => changeZoom(1.2));
  $("#zoom-out").addEventListener("click", () => changeZoom(1 / 1.2));
  $("#zoom-reset").addEventListener("click", () => {
    view = { x: 0, y: 0, scale: 1 };
    updateViewport();
  });
  const graph = $("#knowledge-graph");
  let start = null;
  graph.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      changeZoom(event.deltaY > 0 ? 0.9 : 1.1);
    },
    { passive: false },
  );
  graph.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 || event.target.closest(".graph-node")) return;
    start = {
      clientX: event.clientX,
      clientY: event.clientY,
      x: view.x,
      y: view.y,
    };
    graphDragging = false;
    graph.setPointerCapture(event.pointerId);
  });
  graph.addEventListener("pointermove", (event) => {
    if (!start) return;
    const dx = event.clientX - start.clientX,
      dy = event.clientY - start.clientY;
    if (Math.abs(dx) + Math.abs(dy) > 4) graphDragging = true;
    const factor = 820 / graph.getBoundingClientRect().width;
    view.x = start.x + dx * factor;
    view.y = start.y + dy * factor;
    updateViewport();
  });
  const end = () => {
    start = null;
    setTimeout(() => {
      graphDragging = false;
    }, 0);
  };
  graph.addEventListener("pointerup", end);
  graph.addEventListener("pointercancel", end);
  const search = $("#entity-search"),
    results = $("#search-results");
  search.addEventListener("input", () => {
    const term = search.value.trim().toLocaleLowerCase("vi");
    if (!term) {
      results.hidden = true;
      return;
    }
    const matches = data.entities
      .filter((entity) =>
        [entity.label, entity.id, ...(entity.aliases || [])].some((value) =>
          value.toLocaleLowerCase("vi").includes(term),
        ),
      )
      .slice(0, 8);
    results.innerHTML =
      matches
        .map(
          (entity) =>
            `<button type="button" data-search-id="${escapeHtml(entity.id)}">${escapeHtml(entity.label)} <small class="muted">· ${escapeHtml(entity.id)}</small></button>`,
        )
        .join("") || "<p>Không có thực thể phù hợp trong 150 mẫu đã chọn.</p>";
    results.hidden = false;
  });
  search.addEventListener("keydown", (event) => {
    if (event.key === "Escape") results.hidden = true;
    if (event.key === "ArrowDown") {
      event.preventDefault();
      $("button", results)?.focus();
    }
    if (event.key === "Enter") {
      const first = $("button", results);
      if (first && !results.hidden) selectEntity(first.dataset.searchId);
    }
  });
  results.addEventListener("click", (event) => {
    const button = event.target.closest("[data-search-id]");
    if (button) selectEntity(button.dataset.searchId);
  });
  document.addEventListener("click", (event) => {
    if (!event.target.closest(".search-box")) results.hidden = true;
  });
}

function readSavedAnswers() {
  try {
    const stored = JSON.parse(localStorage.getItem(storageKey) || "{}");
    if (!stored || typeof stored !== "object") return;
    data.questions.forEach((question) => {
      if (
        typeof stored[question.id] === "string" &&
        question.options.includes(stored[question.id])
      )
        answers[question.id] = stored[question.id];
    });
  } catch {
    /* Storage may be unavailable. The quiz still works for this visit. */
  }
}
function saveAnswers() {
  try {
    localStorage.setItem(storageKey, JSON.stringify(answers));
  } catch {
    /* Optional browser-local progress. */
  }
}
function isCorrect(question, answer) {
  return question.acceptedIds.includes(answer);
}

function renderQuiz() {
  const question = data.questions[currentQuestion],
    submitted = answers[question.id];
  selectedOption = submitted || null;
  $("#question-nav").innerHTML = data.questions
    .map((q, index) => {
      const answered = answers[q.id];
      return `<button type="button" data-question="${index}" ${index === currentQuestion ? 'aria-current="step"' : ""} class="${answered ? (isCorrect(q, answered) ? "correct" : "incorrect") : ""}" aria-label="Câu ${index + 1}${answered ? (isCorrect(q, answered) ? ", đã trả lời đúng" : ", đã trả lời sai") : ""}">${String(index + 1).padStart(2, "0")}</button>`;
    })
    .join("");
  const count = Object.keys(answers).length;
  const score = data.questions.filter(
    (q) => answers[q.id] && isCorrect(q, answers[q.id]),
  ).length;
  $("#quiz-score").textContent =
    `${count} / ${data.questions.length} đã trả lời · ${score} đúng`;
  $("#quiz-question").innerHTML =
    `<div class="quiz-meta">CÂU ${String(currentQuestion + 1).padStart(2, "0")} / ${data.questions.length} <span class="split-tag">${question.split === "valid" ? "VALIDATION" : "TEST"}</span></div>
    <h3 id="active-question">${escapeHtml(question.prompt)}</h3>
    <div class="query-equation">${escapeHtml(entityName(question.head))} <b>→ ${escapeHtml(question.relation)} → ?</b></div>
    <fieldset class="quiz-options" aria-labelledby="active-question"><legend class="sr-only">Chọn một đáp án</legend>${question.options.map((id, index) => `<label class="quiz-option ${submitted && isCorrect(question, id) ? "right-answer" : ""} ${submitted === id && !isCorrect(question, id) ? "wrong-answer" : ""}"><input type="radio" name="quiz-answer" value="${escapeHtml(id)}" ${selectedOption === id ? "checked" : ""} ${submitted ? "disabled" : ""}><span>${escapeHtml(entityName(id))}</span></label>`).join("")}</fieldset>
    <div class="quiz-controls"><button id="submit-answer" type="button" class="button primary" ${submitted || !selectedOption ? "disabled" : ""}>Kiểm tra đáp án <span aria-hidden="true">→</span></button><button id="next-question" type="button" class="text-link">${currentQuestion === data.questions.length - 1 ? "Về câu đầu" : "Câu tiếp theo"} →</button></div>
    <div id="quiz-feedback" role="status">${submitted ? feedbackMarkup(question, submitted) : ""}</div>`;
  $("#quiz-summary").hidden = count !== data.questions.length;
  if (count === data.questions.length)
    $("#quiz-summary").innerHTML =
      `<strong>Bạn đã tìm đúng ${score}/${data.questions.length} liên kết.</strong><span>Đây là kết quả chơi của bạn trên lát cắt minh họa, không phải chỉ số đánh giá model.</span>`;
  renderHint();
  renderExplanation();
}
function feedbackMarkup(question, answer) {
  const correct = isCorrect(question, answer);
  return `<div class="quiz-feedback ${correct ? "" : "incorrect"}"><strong>${correct ? "Đúng với liên kết trong dataset!" : "Cùng khám phá đáp án trong dataset."}</strong><p>${escapeHtml(question.explanation)}</p><span class="feedback-triple">${escapeHtml(entityName(question.head))} → ${escapeHtml(relationName(question.relation))} → <b>${escapeHtml(entityName(question.answer))}</b></span></div><button id="show-explanation" type="button" class="button evidence-button" aria-expanded="${explanationOpen}" aria-controls="explanation-panel">Khám phá bằng chứng <span aria-hidden="true">↗</span></button>`;
}

function evidenceRecords(question) {
  const evidence = question.evidence;
  const edgeIds = [
    ...new Set([...evidence.inverseEdgeIds, ...evidence.contextEdgeIds]),
  ];
  return [
    { id: "answer", ...evidence.answerRecord },
    ...edgeIds.map((id) => {
      const edge = data.edges.find((item) => item.id === id);
      return {
        id,
        split: "train",
        row: edge.sourceRow,
        triple: [edge.source, edge.relation, edge.target],
      };
    }),
  ];
}

function recordHref(record) {
  const revision =
    data.meta.sourceRevision || "192d9ff2882b5be6ef8239a37cbd90c5dd102b5a";
  return `https://github.com/OreOZhao/MoSE4MKGC/blob/${revision}/src_data/WN9/${record.split}#L${record.row}`;
}

function tripleText(triple) {
  return `${entityName(triple[0])} → ${triple[1]} → ${entityName(triple[2])}`;
}

function illustrativeRanking(question) {
  // Authored teaching examples, not output from a model or a global entity ranking.
  const distractors = question.options.filter((id) => id !== question.answer);
  const order = [question.answer, ...distractors];
  if (["q03", "q05"].includes(question.id))
    [order[0], order[1]] = [order[1], order[0]];
  const scores = [8.4, 6.7, 4.2, 2.4];
  return order.map((id, index) => ({ id, score: scores[index] }));
}

function renderExplanation() {
  const panel = $("#explanation-panel"),
    question = data.questions[currentQuestion];
  panel.hidden = !explanationOpen || !answers[question.id];
  if (panel.hidden) {
    panel.replaceChildren();
    return;
  }
  const records = evidenceRecords(question),
    ranking = illustrativeRanking(question);
  panel.innerHTML = `<div class="explanation-heading"><div><p class="eyebrow">DỮ KIỆN ĐẾN TỪ ĐÂU?</p><h3 id="explanation-title">Mở từng liên kết. Hiểu từng nguồn.</h3></div><button type="button" class="text-button" data-close-explanation>Thu gọn ↑</button></div>
    <p class="explanation-intro">${escapeHtml(question.evidence.explanation)}</p>
    <div class="explanation-grid"><div class="evidence-data"><div class="evidence-legend"><span><i></i>Cạnh train</span><span><i class="answer-line"></i>Đáp án ${question.split === "valid" ? "validation" : "test"}</span></div><div class="evidence-graph-wrap"><svg id="evidence-graph" role="group" aria-label="Đồ thị bằng chứng. Chọn cạnh hoặc nút để xem nguồn."></svg></div><p class="evidence-instruction">Bấm một cạnh để xem triple, split và dòng dữ liệu gốc. Bấm nút để xem synset.</p><div class="evidence-record-list">${records.map((record) => `<button type="button" data-evidence-record="${escapeHtml(record.id)}" aria-pressed="${selectedEvidenceId === record.id}"><span class="record-split ${record.id === "answer" ? "heldout" : ""}">${record.split}:${record.row}</span><span>${escapeHtml(tripleText(record.triple))}</span></button>`).join("")}</div><div id="evidence-record" class="evidence-record" role="status"></div></div>
    <aside class="illustrative-ranking"><p class="eyebrow">XẾP HẠNG MINH HỌA</p><h4>Đọc một bảng xếp hạng</h4><p class="ranking-caption">Trong 4 lựa chọn của câu hỏi này.</p><ol>${ranking.map((item, index) => `<li><div class="ranking-row"><span class="ranking-position">${index + 1}</span><strong>${escapeHtml(entityName(item.id))}</strong><span class="ranking-value">${item.score.toFixed(1)}</span></div><div class="ranking-bar"><span style="width:${(item.score / 8.4) * 100}%"></span></div>${item.id === question.answer ? '<span class="ranking-truth">Đáp án dataset</span>' : ""}</li>`).join("")}</ol><p class="ranking-caption">Điểm minh họa, không phải xác suất.</p><div class="ranking-reading">${ranking[0].id === question.answer ? "Trong ví dụ này, ứng viên đứng đầu khớp đáp án dataset." : `Trong ví dụ này, ${escapeHtml(entityName(ranking[0].id))} đứng đầu nhưng khác đáp án dataset. Điểm cao không tự xác nhận một liên kết đúng.`}</div><details class="ranking-about"><summary>Cách đọc phần minh họa</summary><p>Thứ tự và điểm được soạn sẵn để minh họa cả trường hợp đúng và sai; không chạy checkpoint. Bảng chỉ xếp hạng bốn lựa chọn, không phải toàn bộ 6.555 thực thể WN9.</p></details></aside></div>
    <p class="explanation-limit">Các cạnh highlight là ngữ cảnh có thật trong dataset. Chúng giúp đọc dữ kiện, chưa chứng minh model đã dựa vào chính các cạnh này. Ảnh và mô tả là gợi ý cho người xem.</p>`;
  renderEvidenceGraph();
  renderEvidenceRecord();
}

function renderEvidenceRecord() {
  const question = data.questions[currentQuestion];
  const record = evidenceRecords(question).find(
    (item) => item.id === selectedEvidenceId,
  );
  const target = $("#evidence-record");
  if (!target) return;
  $$("[data-evidence-record]").forEach((button) =>
    button.setAttribute(
      "aria-pressed",
      String(button.dataset.evidenceRecord === selectedEvidenceId),
    ),
  );
  if (record) {
    const [head, relation, tail] = record.triple;
    target.innerHTML = `<p class="eyebrow">${record.id === "answer" ? "LIÊN KẾT CỦA CÂU HỎI" : "DỮ KIỆN ĐÃ CÓ TRONG TRAIN"}</p><p class="record-triple">${escapeHtml(tripleText(record.triple))}</p><code>${escapeHtml(head)} ${escapeHtml(relation)} ${escapeHtml(tail)}</code><p><b>${escapeHtml(record.split)}</b> · Dòng ${record.row} · WN9 / MoSE</p><a class="text-link" href="${escapeHtml(recordHref(record))}" target="_blank" rel="noopener noreferrer">Xem đúng dòng dữ liệu gốc ↗</a>`;
  } else {
    const entity = entities.get(selectedEvidenceId);
    if (!entity) return;
    target.innerHTML = `<p class="eyebrow">SYNSET TRONG DATASET</p><p class="record-triple">${escapeHtml(entity.label)} <code>${escapeHtml(entity.id)}</code></p><p lang="en">${escapeHtml(entity.description || "No source description available.")}</p><a class="text-link" href="${escapeHtml(entity.sourceUrl)}" target="_blank" rel="noopener noreferrer">Xem định danh nguồn ↗</a>`;
  }
}

function renderEvidenceGraph() {
  const graph = $("#evidence-graph");
  if (!graph) return;
  const question = data.questions[currentQuestion];
  const ids = [
    ...new Set([
      question.head,
      question.answer,
      ...question.evidence.entityIds,
    ]),
  ];
  const width = Math.max(240, graph.parentElement.clientWidth),
    columns = width >= 560 ? 3 : 2;
  const nodeWidth = Math.min(164, width / columns - 22),
    nodeHeight = 58;
  const height = Math.ceil(ids.length / columns) * 136 + 35;
  graph.setAttribute("viewBox", `0 0 ${width} ${height}`);
  graph.style.height = `${height}px`;
  graph.replaceChildren();
  const defs = makeSvg("defs", {}),
    marker = makeSvg("marker", {
      id: "evidence-arrow",
      viewBox: "0 0 10 10",
      refX: 9,
      refY: 5,
      markerWidth: 6,
      markerHeight: 6,
      orient: "auto",
    });
  marker.append(
    makeSvg("path", { d: "M0 0L10 5L0 10z", fill: "context-stroke" }),
  );
  defs.append(marker);
  graph.append(defs);
  const positions = new Map(
    ids.map((id, index) => [
      id,
      {
        x: (((index % columns) + 0.5) * width) / columns,
        y: 75 + Math.floor(index / columns) * 136,
      },
    ]),
  );
  const edgePriority = (record) =>
    record.id === "answer"
      ? 2
      : question.evidence.inverseEdgeIds.includes(record.id)
        ? 1
        : 0;
  evidenceRecords(question)
    .sort((a, b) => edgePriority(a) - edgePriority(b))
    .forEach((record, index) => {
      const [head, relation, tail] = record.triple,
        source = positions.get(head),
        target = positions.get(tail);
      if (!source || !target) return;
      const dx = target.x - source.x,
        dy = target.y - source.y,
        distance = Math.max(1, Math.hypot(dx, dy)),
        ux = dx / distance,
        uy = dy / distance;
      const padding = Math.min(
        nodeWidth / 2 / Math.max(0.01, Math.abs(ux)),
        nodeHeight / 2 / Math.max(0.01, Math.abs(uy)),
      );
      const x1 = source.x + ux * padding,
        y1 = source.y + uy * padding,
        x2 = target.x - ux * (padding + 5),
        y2 = target.y - uy * (padding + 5);
      const bend =
        Math.abs(dy) < 10 ? (dx > 0 ? -90 : 90) : (index % 2 ? 1 : -1) * 28;
      const cx = (x1 + x2) / 2,
        cy = (y1 + y2) / 2 + bend;
      const path = makeSvg("path", {
        d: `M${x1} ${y1} Q${cx} ${cy} ${x2} ${y2}`,
        class: `evidence-path ${record.id === "answer" ? "answer-path" : ""} ${record.id === selectedEvidenceId ? "active" : ""}`,
        "marker-end": "url(#evidence-arrow)",
        "data-evidence-record": record.id,
      });
      const hit = makeSvg("path", {
        d: `M${x1} ${y1} Q${cx} ${cy} ${x2} ${y2}`,
        class: "evidence-hit",
        "data-evidence-record": record.id,
        role: "button",
        tabindex: 0,
        "aria-label": `${tripleText(record.triple)}, ${record.split} dòng ${record.row}`,
      });
      const title = makeSvg("title", {});
      title.textContent = `${tripleText(record.triple)} · ${record.split}:${record.row}`;
      path.append(title);
      graph.append(path, hit);
      if (record.id === selectedEvidenceId) {
        const text = makeSvg("text", {
          x: cx,
          y: (y1 + y2) / 2 + bend / 2 - 5,
          "text-anchor": "middle",
          class: "evidence-relation",
        });
        text.textContent = relation;
        graph.append(text);
      }
    });
  ids.forEach((id) => {
    const entity = entities.get(id),
      point = positions.get(id),
      group = makeSvg("g", {
        class: `evidence-node ${id === question.answer ? "answer-node" : ""}`,
        transform: `translate(${point.x},${point.y})`,
        role: "button",
        tabindex: 0,
        "data-evidence-node": id,
        "aria-label": `${entity.label}, ${id}`,
      });
    group.append(
      makeSvg("rect", {
        x: -nodeWidth / 2,
        y: -nodeHeight / 2,
        width: nodeWidth,
        height: nodeHeight,
        rx: 9,
      }),
    );
    const label = makeSvg("text", {
      x: 0,
      y: -2,
      "text-anchor": "middle",
      class: "evidence-node-name",
    });
    const maxCharacters = Math.max(9, Math.floor((nodeWidth - 12) / 8));
    label.textContent =
      entity.label.length > maxCharacters
        ? entity.label.slice(0, maxCharacters - 1) + "…"
        : entity.label;
    const synset = makeSvg("text", {
      x: 0,
      y: 18,
      "text-anchor": "middle",
      class: "evidence-node-id",
    });
    synset.textContent = id;
    group.append(label, synset);
    graph.append(group);
  });
}

function bindExplanation() {
  const panel = $("#explanation-panel");
  panel.addEventListener("click", (event) => {
    if (event.target.closest("[data-close-explanation]")) {
      explanationOpen = false;
      renderExplanation();
      $("#show-explanation").setAttribute("aria-expanded", "false");
      $("#show-explanation").focus();
      return;
    }
    const record = event.target.closest("[data-evidence-record]"),
      node = event.target.closest("[data-evidence-node]");
    if (!record && !node) return;
    const fromGraph = Boolean(event.target.closest("#evidence-graph"));
    selectedEvidenceId = record
      ? record.dataset.evidenceRecord
      : node.dataset.evidenceNode;
    renderEvidenceRecord();
    renderEvidenceGraph();
    if (fromGraph) {
      const selector = record ? ".evidence-hit" : ".evidence-node";
      $$(selector)
        .find(
          (item) =>
            (record
              ? item.dataset.evidenceRecord
              : item.dataset.evidenceNode) === selectedEvidenceId,
        )
        ?.focus({ preventScroll: true });
    }
  });
  panel.addEventListener("keydown", (event) => {
    if (
      (event.key === "Enter" || event.key === " ") &&
      event.target.matches(".evidence-hit,.evidence-node")
    ) {
      event.preventDefault();
      event.target.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    }
  });
  let resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      if (explanationOpen) renderEvidenceGraph();
    }, 100);
  });
}
function renderHint() {
  const question = data.questions[currentQuestion],
    entity = entities.get(question.head);
  $$("[data-hint]").forEach((button) =>
    button.setAttribute(
      "aria-pressed",
      String(button.dataset.hint === activeHint),
    ),
  );
  const container = $("#hint-content");
  if (!activeHint) {
    container.innerHTML =
      '<div class="hint-placeholder"><span>↗</span>Chọn một loại dữ liệu ở trên<br>để mở thêm gợi ý.</div>';
    return;
  }
  if (activeHint === "image") {
    container.innerHTML = entity.image
      ? `<img src="${escapeHtml(entity.image)}" alt="Ảnh gợi ý của ${escapeHtml(entity.label)}" width="440" height="320"><p style="font-size:14px;margin-top:8px">${escapeHtml(entity.label)} · Ảnh nguồn WN9, kế thừa từ WN18</p>`
      : `<div class="hint-placeholder"><span>▧</span>${entity.imageAvailable ? "Ảnh có trong nguồn nhưng chưa đóng gói trong bản demo nhỏ này." : "Nguồn WN9 không có ảnh cho thực thể này."}</div>`;
  } else if (activeHint === "text") {
    container.innerHTML = entity.description
      ? `<h4>${escapeHtml(entity.label)}</h4><p class="hint-quote" lang="en">“${escapeHtml(entity.description)}”</p><p style="font-size:14px;margin-top:10px">Trích đoạn mô tả gốc tiếng Anh trong dataset.</p>`
      : '<div class="hint-placeholder"><span>≡</span>Không có mô tả cho thực thể này trong nguồn WN9. Thiếu modality cũng là một phần của bài toán.</div>';
  } else {
    const linked = neighbors(question.head).slice(0, 5);
    container.innerHTML = `<h4>${escapeHtml(entity.label)} · ngữ cảnh train</h4>${linked.map((edge) => `<div class="hint-neighbor"><span>${edge.source === question.head ? "→" : "←"} ${escapeHtml(edge.relation)}</span><br>${escapeHtml(entityName(otherEnd(edge, question.head)))}</div>`).join("") || "<p>Chưa có cạnh train trong lát cắt này. Bạn có thể thử một nguồn gợi ý khác.</p>"}<button class="text-link" type="button" data-explore-head="${escapeHtml(question.head)}" style="margin-top:13px;font-size:14px">Xem trên đồ thị ↗</button>`;
  }
}
function goQuestion(index, scroll = false) {
  explanationOpen = false;
  selectedEvidenceId = "answer";
  currentQuestion = index;
  activeHint = null;
  renderQuiz();
  if (scroll) $("#challenge").scrollIntoView({ behavior: "smooth" });
}
function bindQuiz() {
  $("#question-nav").addEventListener("click", (event) => {
    const button = event.target.closest("[data-question]");
    if (button) goQuestion(Number(button.dataset.question));
  });
  $("#quiz-question").addEventListener("change", (event) => {
    if (event.target.name !== "quiz-answer") return;
    selectedOption = event.target.value;
    $("#submit-answer").disabled = false;
  });
  $("#quiz-question").addEventListener("click", (event) => {
    if (event.target.closest("#show-explanation")) {
      explanationOpen = !explanationOpen;
      renderExplanation();
      $("#show-explanation").setAttribute(
        "aria-expanded",
        String(explanationOpen),
      );
      if (explanationOpen)
        $("#explanation-panel").scrollIntoView({
          behavior: "smooth",
          block: "start",
        });
      return;
    }
    if (event.target.closest("#next-question")) {
      goQuestion((currentQuestion + 1) % data.questions.length);
      return;
    }
    if (event.target.closest("#submit-answer") && selectedOption) {
      const question = data.questions[currentQuestion];
      if (answers[question.id]) return;
      answers[question.id] = selectedOption;
      saveAnswers();
      renderQuiz();
      $("#quiz-feedback").scrollIntoView({
        behavior: "smooth",
        block: "nearest",
      });
    }
  });
  $("#hint-tabs").addEventListener("click", (event) => {
    const button = event.target.closest("[data-hint]");
    if (!button) return;
    activeHint =
      activeHint === button.dataset.hint ? null : button.dataset.hint;
    renderHint();
  });
  $("#hint-content").addEventListener("click", (event) => {
    const button = event.target.closest("[data-explore-head]");
    if (button) selectEntity(button.dataset.exploreHead, true);
  });
  $("#quiz-reset").addEventListener("click", () => {
    answers = {};
    saveAnswers();
    goQuestion(0);
    $("#quiz-score").textContent =
      `Đã bắt đầu lại · 0 / ${data.questions.length} đã trả lời`;
  });
}

function renderCases() {
  const study = data.cases[currentCase];
  $("#case-nav").innerHTML = data.cases
    .map(
      (item, index) =>
        `<button type="button" data-case="${index}" aria-pressed="${index === currentCase}"><span>${String(index + 1).padStart(2, "0")}</span>${escapeHtml(item.title)}</button>`,
    )
    .join("");
  $("#case-detail").innerHTML =
    `<p class="eyebrow">CASE STUDY ${String(currentCase + 1).padStart(2, "0")} / 05</p><h3>${escapeHtml(study.title)}</h3><p class="case-subtitle">${escapeHtml(study.subtitle)}</p>
    <div class="case-entities">${study.entityIds
      .map((id) => {
        const entity = entities.get(id);
        return `<button type="button" class="case-entity" data-case-entity="${escapeHtml(id)}">${entity.image ? `<img src="${escapeHtml(entity.image)}" alt="" width="52" height="58" loading="lazy">` : `<span class="case-initial" aria-hidden="true">${escapeHtml(initial(entity))}</span>`}<span><strong>${escapeHtml(entity.label)}</strong><small>${escapeHtml(kindLabels[entity.kind])} · Xem dữ liệu ↗</small></span></button>`;
      })
      .join("")}</div>
    <p class="case-observation">${escapeHtml(study.observation)}</p><div class="case-lesson"><span aria-hidden="true">↳</span><p>${escapeHtml(study.lesson)}</p></div><div class="case-actions">${study.questionId ? `<button type="button" class="text-link" data-case-question="${escapeHtml(study.questionId)}">Thử câu hỏi liên quan →</button>` : ""}<p class="case-disclaimer">Minh họa từ dữ liệu · Không phải phân tích dự đoán model</p></div>`;
}
function bindCases() {
  $("#case-nav").addEventListener("click", (event) => {
    const button = event.target.closest("[data-case]");
    if (button) {
      currentCase = Number(button.dataset.case);
      renderCases();
    }
  });
  $("#case-detail").addEventListener("click", (event) => {
    const entity = event.target.closest("[data-case-entity]");
    if (entity) selectEntity(entity.dataset.caseEntity, true);
    const question = event.target.closest("[data-case-question]");
    if (question)
      goQuestion(
        data.questions.findIndex((q) => q.id === question.dataset.caseQuestion),
        true,
      );
  });
}
function renderNotes() {
  const meta = data.meta;
  $("#data-notes-content").className = "data-notes-content";
  $("#data-notes-content").innerHTML =
    `<div><p><b>Dữ liệu thật, phạm vi nhỏ.</b> ${escapeHtml(meta.selectionNote)} Đồ thị gồm ${meta.edgeCount} cạnh train. ${escapeHtml(meta.graphNote)}</p><p>${escapeHtml(meta.optionsNote)}</p><p><b>Giải thích bằng dữ kiện:</b> mỗi câu có nguồn đáp án, số dòng, các cạnh train liên quan và nhãn quan hệ đảo chiều nếu có. Vùng đồ thị là ngữ cảnh đọc dữ liệu; không phải kết luận về ảnh hưởng nhân quả của từng triple lên model.</p><p><b>Tiến độ:</b> câu trả lời chỉ lưu trong localStorage của trình duyệt hiện tại; bấm “Làm lại” để xóa.</p></div><div><p><b>Xếp hạng minh họa:</b> thứ tự và điểm được soạn sẵn trong bốn lựa chọn để minh họa cả trường hợp đúng và sai. Không chạy checkpoint; không phải xác suất hay bảng xếp hạng toàn bộ thực thể.</p><p><b>Phương pháp:</b> ${escapeHtml(meta.protocol)}</p><p><b>Ảnh và văn bản:</b> ${meta.thumbnailCount} thumbnail được đóng gói; ${meta.selectedRawImageCount}/${meta.entityCount} thực thể có ảnh nguồn, ${meta.selectedTextCount}/${meta.entityCount} có mô tả. Không có thumbnail trong bundle không đồng nghĩa nguồn thiếu ảnh.</p><p>${escapeHtml(meta.imageCredit)} ${escapeHtml(meta.textCredit)}</p><p>${meta.sourceLinks.map((link) => `<a href="${escapeHtml(link.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(link.title)} ↗</a>`).join(" · ")} · <a href="data/provenance.json" target="_blank" rel="noopener noreferrer">Manifest dữ liệu ↗</a></p></div>`;
}

async function init() {
  try {
    const response = await fetch("data/demo.json?v=3");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    data = await response.json();
    entities = new Map(data.entities.map((entity) => [entity.id, entity]));
    relations = new Map(
      data.relations.map((relation) => [relation.id, relation.label]),
    );
    selectedId = entities.has(selectedId) ? selectedId : data.questions[0].head;
    data.relations.forEach((relation) => {
      const option = document.createElement("option");
      option.value = relation.id;
      option.textContent = relation.label;
      option.title = relation.meaning || relation.label;
      $("#relation-filter").append(option);
    });
    $$("[data-stat='entities']").forEach(
      (element) => (element.textContent = data.meta.entityCount),
    );
    $("#selection-note").textContent =
      `${data.meta.entityCount} thực thể được tuyển chọn từ benchmark`;
    if (entities.get(selectedId)?.image)
      $("#hero-entity-image").innerHTML =
        `<img src="${escapeHtml(entities.get(selectedId).image)}" alt="corgi trong dataset WN9" width="50" height="50">`;
    readSavedAnswers();
    renderInspector();
    renderGraph();
    renderQuiz();
    renderCases();
    renderNotes();
    bindGraph();
    bindQuiz();
    bindExplanation();
    bindCases();
    $("#load-status").hidden = true;
    $("#app").hidden = false;
  } catch (error) {
    console.error("DSoM Explorer could not load:", error);
    $("#load-status").innerHTML =
      "Chưa tải được dữ liệu demo. Hãy tải lại trang hoặc chạy qua HTTP server theo README trong thư mục demo.";
  }
}
init();

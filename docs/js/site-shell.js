import { chapterMap, currentChapter } from './systems/navigation.js';

const initMythology = (domRuntime) => {
  const owner = 'site-shell';
  const acts = [
    "I · NATURE",
    "II · COMPILER",
    "III · COMPUTEIMAGE",
    "IV · EXECUTION",
    "V · FABRIC",
    "VI · RESEARCH",
    "VII · FUTURE",
    "I · NATURE",
  ];
  const current = currentChapter();
  document.body.dataset.prismAct = String(
    Math.min(
      5,
      Math.max(
        0,
        current === 0
          ? 1
          : current < 4
            ? 2
            : current === 4
              ? 4
              : current === 5
                ? 3
                : current === 6
                  ? 5
                  : 1,
      ),
    ),
  );
  const marker = document.createElement("div");
  marker.className = "mythology-marker";
  marker.setAttribute("aria-label", `Prism act ${acts[current]}`);
  marker.textContent = `ACT ${acts[current]}`;
  const shell = document.querySelector('.component-header') || document.querySelector('header') || document.body;
  shell.append(marker);
  domRuntime?.claimNode?.(owner, marker);
  const signatureGroups = [
    ...document.querySelectorAll(
      ".pipeline-list,.image-layout,.runtime-map,.runtime-contract,.workflow,.phase-flow,.target-grid,.validation-grid,.gate-line,.loop,.system-map",
    ),
  ];
  signatureGroups.forEach((group) => {
    group.classList.add("mythology-signature");
    [...group.children]
      .filter((node) => node.matches("div,article,button"))
      .forEach((node, index) =>
        node.style.setProperty("--mythology-delay", `${index * 70}ms`),
      );
  });
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  const observer = new IntersectionObserver(
    (entries) =>
      entries.forEach((entry) =>
        entry.target.toggleAttribute(
          "data-mythology-visible",
          entry.isIntersecting,
        ),
      ),
    { threshold: 0.18 },
  );
  signatureGroups.forEach((group) => observer.observe(group));
};
const initSignatureMoment = (domRuntime) => {
  const owner = 'site-shell';
  const current = currentChapter();
  const moments = [
    null,
    {
      selector: ".hero-visual",
      label: "SIGNATURE MOMENT / ORIGIN",
      title: "Light enters. Structure emerges.",
      copy: "The site begins as the compiler begins: one input enters, hidden structure separates, and the deployment world becomes legible.",
      status: "beam → prism → spectrum",
    },
    {
      selector: ".system-map",
      label: "SIGNATURE MOMENT / COMPILER",
      title: "The graph is alive.",
      copy: "Select a subsystem. Its dependencies, outputs, and proof boundary change together because the architecture is modeled as state, not a poster.",
      status: "select a node to propagate state",
    },
    {
      selector: ".instrument-grid,.workflow",
      label: "SIGNATURE MOMENT / SEARCH",
      title: "Candidates disappear honestly.",
      copy: "Scroll or select a stage to watch the representation frontier narrow under quality, legality, and resource gates.",
      status: "search → admission → seal",
    },
    {
      selector: ".phase-flow,.target-grid",
      label: "SIGNATURE MOMENT / EXECUTION",
      title: "Work finds a home.",
      copy: "Follow one serving requirement as phases, KV state, and handoffs cross explicit provider boundaries.",
      status: "prefill → handoff → decode",
    },
    {
      selector: ".runtime-map,.runtime-contract",
      label: "SIGNATURE MOMENT / COMPUTEIMAGE",
      title: "The artifact remembers why.",
      copy: "The sealed image carries the views, residency, plan, and receipts that let runtime execution remain attributable.",
      status: "views → residency → receipt",
    },
    {
      selector: ".loop,.scale",
      label: "SIGNATURE MOMENT / RESEARCH",
      title: "Representation meets deployment.",
      copy: "The model and the machine remain separate systems, connected by an inspectable artifact and a feedback loop.",
      status: "model → admission → evidence",
    },
    {
      selector: ".validation-grid,.gate-line",
      label: "SIGNATURE MOMENT / FUTURE",
      title: "The future must be proven.",
      copy: "Roadmap progress is not decoration: each target advances only when its boundary has a measured, reproducible claim.",
      status: "validate → compare → promote",
    },
  ][current];
  if (!moments) return;
  const target = document.querySelector(moments.selector);
  if (!target) return;
  target.classList.add("signature-target");
  const panel = document.createElement("aside");
  panel.className = "signature-moment";
  panel.innerHTML = `<span class="signature-moment-label">${moments.label}</span><h3>${moments.title}</h3><p>${moments.copy}</p><span class="signature-moment-status" aria-live="polite">${moments.status}</span>`;
  target.parentNode.insertBefore(panel, target);
  domRuntime?.claimNode?.(owner, panel);
  const status = panel.querySelector(".signature-moment-status");
  const nodes = [
    ...target.querySelectorAll(
      ":scope > div, :scope > article, :scope > button",
    ),
  ];
  const activate = (index) => {
    target.dataset.signatureActive = "true";
    if (nodes[index]) {
      nodes.forEach((node, nodeIndex) =>
        node.toggleAttribute("data-signature-focus", nodeIndex === index),
      );
      status.textContent = `${moments.status} / focus ${String(index + 1).padStart(2, "0")}`;
    }
  };
  nodes.forEach((node, index) => {
    node.addEventListener("mouseenter", () => activate(index));
    node.addEventListener("focus", () => activate(index));
    node.addEventListener("click", () => activate(index));
  });
  const observer = new IntersectionObserver(
    ([entry]) => panel.classList.toggle("is-visible", entry.isIntersecting),
    { threshold: 0.25 },
  );
  observer.observe(panel);
};
const initPrismAtmosphere = (kernel) => {
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  let scrollTimer;
  const markScrolling = () => {
    document.body.dataset.prismScrolling = "true";
    clearTimeout(scrollTimer);
    scrollTimer = setTimeout(
      () => delete document.body.dataset.prismScrolling,
      260,
    );
  };
  kernel?.on("scroll", markScrolling);
};
const initActCorrection = () => {
  const acts = [
    "I · NATURE",
    "I · NATURE",
    "II · COMPILER",
    "II · COMPILER",
    "IV · EXECUTION",
    "III · COMPUTEIMAGE",
    "VI · RESEARCH",
    "V · FUTURE",
  ];
  const accentStage = [1, 1, 2, 2, 4, 3, 5, 5];
  const current = currentChapter();
  document.body.dataset.prismAct = String(accentStage[current]);
  const marker = document.querySelector(".mythology-marker");
  if (marker) {
    marker.textContent = `ACT ${acts[current]}`;
    marker.setAttribute("aria-label", `Prism act ${acts[current]}`);
  }
};
const initComputeImageSignature = (kernel, domRuntime) => {
  const owner = 'site-shell';
  if (currentChapter() !== 5 || document.querySelector(".signature-moment"))
    return;
  const target = document.querySelector(".boundary,.validation,.adds");
  if (!target) return;
  target.classList.add("signature-target");
  const panel = document.createElement("aside");
  panel.className = "signature-moment is-visible";
  panel.innerHTML =
    '<span class="signature-moment-label">SIGNATURE MOMENT / COMPUTEIMAGE</span><h3>The artifact crosses a provider boundary.</h3><p>ComputeImages preserve the physical reality of specialized hardware while keeping the deployment contract inspectable and comparable.</p><span class="signature-moment-status">capability → placement → evidence</span>';
  target.parentNode.insertBefore(panel, target);
  domRuntime?.claimNode?.(owner, panel);
};
const initStartHereSignature = (domRuntime) => {
  const owner = 'site-shell';
  if (currentChapter() !== 1) return;
  const target = document.querySelector(".guide-flow");
  if (!target) return;
  const panel = document.createElement("aside");
  panel.className = "signature-moment is-visible";
  panel.innerHTML =
    '<span class="signature-moment-label">SIGNATURE MOMENT / ORIGIN</span><h3>The reading path is born from one beam.</h3><p>Move through model, compiler, ComputeImage, and runtime as one continuous transformation.</p><span class="signature-moment-status">model → compiler → artifact → runtime</span>';
  target.parentNode.insertBefore(panel, target);
  domRuntime?.claimNode?.(owner, panel);
};
const initPrismStages = () => {
  const stages = [...document.querySelectorAll("main section")].filter(
    (section) => section.querySelector(".eyebrow,.kicker,label"),
  );
  stages.forEach((stage, index) =>
    stage
      .querySelector(".eyebrow,.kicker,label")
      ?.setAttribute("data-prism-index", String(index + 1).padStart(2, "0")),
  );
  if (!stages.length) return;
  let activeIndex = -1,
    lastScroll = scrollY;
  const activate = (index) => {
    if (index < 0 || index >= stages.length || index === activeIndex) return;
    const previous = activeIndex;
    activeIndex = index;
    stages.forEach((stage, stageIndex) =>
      stage.toggleAttribute("data-prism-active", stageIndex === index),
    );
    document.body.dataset.prismStage = String(index + 1).padStart(2, "0");
    document.body.style.setProperty(
      "--prism-stage-progress",
      `${stages.length < 2 ? 0 : (index / (stages.length - 1)) * 100}%`,
    );
    document.body.classList.remove("prism-stage-transition");
    void document.body.offsetWidth;
    document.body.classList.add("prism-stage-transition");
    setTimeout(
      () => document.body.classList.remove("prism-stage-transition"),
      420,
    );
    document.body.dataset.prismStageDirection =
      previous < 0 ? "forward" : index > previous ? "forward" : "backward";
  };
  const nearest = () => {
    const focus = innerHeight * 0.42;
    let best = 0,
      closest = Infinity;
    stages.forEach((stage, index) => {
      const distance = Math.abs(stage.getBoundingClientRect().top - focus);
      if (distance < closest) {
        best = index;
        closest = distance;
      }
    });
    return best;
  };
  const update = () => {
    const direction =
      scrollY > lastScroll
        ? "forward"
        : scrollY < lastScroll
          ? "backward"
          : document.body.dataset.prismStageDirection || "forward";
    document.body.dataset.prismScrollDirection = direction;
    lastScroll = scrollY;
    activate(nearest());
  };
  kernel?.on("scroll", update);
  addEventListener("resize", update);
  activate(0);
  update();
};
const initRepresentationLayers = () => {
  const layers = [...document.querySelectorAll(".layer-grid article")];
  layers.forEach((article) => {
    const title = article.querySelector("h3")?.textContent.toLowerCase() || "";
    const layer = title.includes("logical")
      ? "logical"
      : title.includes("physical")
        ? "physical"
        : title.includes("execution")
          ? "execution"
          : "";
    if (!layer) return;
    article.dataset.layer = layer;
    article.tabIndex = 0;
    article.setAttribute("role", "button");
    article.setAttribute("aria-pressed", "false");
    const activate = () => {
      layers.forEach((item) => {
        item.removeAttribute("data-layer-active");
        item.setAttribute("aria-pressed", "false");
      });
      article.setAttribute("data-layer-active", "true");
      article.setAttribute("aria-pressed", "true");
      document.body.dataset.prismLayer = layer;
    };
    article.addEventListener("click", activate);
    article.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        activate();
      }
    });
  });
  if (layers[0]) layers[0].click();
};
const initLivingArchitecture = () => {
  const groups = [
    ...document.querySelectorAll(
      ".pipeline-list,.image-layout,.runtime-contract,.workflow,.gate-line,.scale,.target-grid,.validation-grid",
    ),
  ];
  groups.forEach((group) => {
    const nodes = [...group.children].filter((node) =>
      node.matches("div,article"),
    );
    if (nodes.length < 2) return;
    nodes.forEach((node, index) => {
      node.dataset.livingNode = String(index + 1).padStart(2, "0");
      node.tabIndex = 0;
      node.addEventListener("mouseenter", () => {
        nodes.forEach((item, itemIndex) =>
          item.toggleAttribute(
            "data-living-active",
            itemIndex === index ||
              itemIndex === index - 1 ||
              itemIndex === index + 1,
          ),
        );
      });
      node.addEventListener("mouseleave", () =>
        nodes.forEach((item) => item.removeAttribute("data-living-active")),
      );
      node.addEventListener("focus", () =>
        node.dispatchEvent(new Event("mouseenter")),
      );
      node.addEventListener("blur", () =>
        node.dispatchEvent(new Event("mouseleave")),
      );
    });
    group.dataset.livingArchitecture = "true";
  });
};
const initLivingAtlas = () => {
  const atlas = document.querySelector(".living-atlas");
  if (!atlas) return;
  const content = {
    compiler: [
      "COMPILER / SEARCH",
      "Compiler",
      "Preserves representation identity while searching placement and target legality.",
      "Model graph + quality contract",
      "ECS world / CPU analysis",
      "Admitted candidates + PrismIR",
      "ComputeImage · Scheduler · Evidence",
      "identity → search → admission",
    ],
    image: [
      "ARTIFACT / CIMAGE",
      "ComputeImage",
      "Carries logical tensors, physical layouts, execution views, residency, and receipts across the compiler/runtime boundary.",
      "PrismIR + target capabilities",
      "Mapped payloads / shared memory",
      "Sealed deployment contract",
      "Compiler · Scheduler · Evidence",
      "views → residency → seal",
    ],
    scheduler: [
      "RUNTIME / CONTROL",
      "Scheduler",
      "Turns the sealed plan into ordered work, choosing lanes, residency, handoffs, and barriers without inventing policy.",
      "CImage + request + capabilities",
      "CPU · GPU · NPU lanes",
      "Tokens + execution receipt",
      "ComputeImage · Fabric · Evidence",
      "admit → place → handoff → run",
    ],
    evidence: [
      "GOVERNANCE / PROOF",
      "Evidence",
      "Keeps the architecture honest by recording what was admitted, planned, executed, measured, and replayable.",
      "Compiler decisions + runtime outcomes",
      "Receipt store / validation boundary",
      "Proof, metrics, recovery class",
      "Compiler · ComputeImage · Scheduler",
      "admitted → planned → measured → replayable",
    ],
    fabric: [
      "SYSTEM / FABRIC",
      "Fabric",
      "Scales the same decomposition from one machine to multiple GPUs, nodes, and edge placements.",
      "Execution graph + provider topology",
      "Machine, node, and network tiers",
      "Distributed plan + handoff graph",
      "Scheduler · Residency · Evidence",
      "machine → node → fleet → receipt",
    ],
  };
  const nodes = [...atlas.querySelectorAll("[data-atlas-node]")];
  const inspector = atlas.querySelector(".atlas-inspector");
  const initial =
    {
      Runtime: "scheduler",
      ComputeImages: "image",
      Research: "evidence",
      Roadmap: "fabric",
    }[atlas.dataset.atlasContext] || "compiler";
  const set = (key) => {
    const d = content[key];
    nodes.forEach((node) =>
      node.classList.toggle("is-active", node.dataset.atlasNode === key),
    );
    inspector.querySelector(".atlas-kicker").textContent = d[0];
    inspector.querySelector("h3").textContent = d[1];
    inspector.querySelector("p").textContent = d[2];
    const values = inspector.querySelectorAll("dd");
    values[0].textContent = d[3];
    values[1].textContent = d[4];
    values[2].textContent = d[5];
    inspector.querySelector(".atlas-proof strong").textContent = d[6];
    inspector.querySelector("[data-atlas-trace]").textContent = d[7];
    document.body.dataset.prismAtlas = key;
  };
  nodes.forEach((node) => {
    node.addEventListener("click", () => set(node.dataset.atlasNode));
    node.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        set(node.dataset.atlasNode);
      }
    });
  });
  set(initial);
};
const initAtlasScroll = (kernel) => {
  const atlas = document.querySelector(".living-atlas");
  if (!atlas || matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  const nodes = [...atlas.querySelectorAll("[data-atlas-node]")];
  let frame = 0;
  const update = () => {
    frame = 0;
    const rect = atlas.getBoundingClientRect(),
      travel = Math.max(atlas.offsetHeight - innerHeight * 0.35, 1),
      amount = Math.max(
        0,
        Math.min(1, (innerHeight * 0.58 - rect.top) / travel),
      ),
      index = Math.min(nodes.length - 1, Math.floor(amount * nodes.length));
    if (rect.bottom > innerHeight * 0.18 && rect.top < innerHeight * 0.82) {
      const node = nodes[index];
      if (node && !node.classList.contains("is-active")) node.click();
    }
  };
  const request = () => {
    if (!frame) frame = requestAnimationFrame(update);
  };
  kernel?.on("scroll", request);
  addEventListener("resize", request);
  request();
};
const initDiscoveryRooms = (domRuntime) => {
  const owner = 'site-shell';
  if (document.querySelector(".discovery-room")) return;
  const questions = [
    "What is computation?",
    "Why does the graph need a compiler?",
    "How does structure become a legal artifact?",
    "Where does execution find a home?",
    "What does the sealed object remember?",
    "How does representation meet deployment?",
    "What can be proven, and what remains ahead?",
    "How does the field become a fabric?",
  ];
  const current = currentChapter(),
    header = document.querySelector(".component-header");
  if (!header || !questions[current]) return;
  const room = document.createElement("div");
  room.className = "discovery-room";
  room.innerHTML = `<span class="discovery-room-act">ACT ${String(current + 1).padStart(2, "0")} / ${chapterMap[current][0]}</span><span class="discovery-room-question">${questions[current]}</span><span class="discovery-room-light" aria-hidden="true"></span>`;
  header.after(room);
  domRuntime?.claimNode?.(owner, room);
  document.body.dataset.prismDiscovery = String(current + 1).padStart(2, "0");
};
const initSurfaceConstellation = () => {
  const constellation = document.querySelector(".illumination-constellation");
  if (!constellation) return;
  const data = {
    cimage: [
      "ARTIFACT / CIMAGE",
      "ComputeImage",
      "Carries representation across physical layouts, execution views, residency, and receipts without making runtime rediscover deployment policy.",
      "sealed deployment contract",
      "compiler · scheduler · evidence",
      "intent → views → execution",
    ],
    compiler: [
      "COMPILER / SEARCH",
      "Spatial Compiler",
      "Searches representation and placement together while preserving identity and explicit admission boundaries.",
      "representation frontier",
      "ComputeImage · runtime · evidence",
      "identity → search → admission",
    ],
    runtime: [
      "RUNTIME / REALIZATION",
      "Runtime",
      "Turns a sealed plan into ordered work across validated CPU, GPU, and NPU execution boundaries.",
      "observable execution",
      "ComputeImage · scheduler · evidence",
      "plan → dispatch → receipt",
    ],
    scheduler: [
      "RUNTIME / CONTROL",
      "Scheduler",
      "Chooses lanes, residency, handoffs, and barriers without inventing policy that was not admitted at compile time.",
      "ordered work",
      "runtime · fabric · evidence",
      "admit → place → handoff",
    ],
    evidence: [
      "GOVERNANCE / PROOF",
      "Evidence Engine",
      "Records what was admitted, planned, executed, measured, and replayable so the field remains honest.",
      "receipts and replay facts",
      "compiler · ComputeImage · runtime",
      "claim → measure → compare",
    ],
    fabric: [
      "SYSTEM / FIELD",
      "Fabric",
      "Extends the same explicit decomposition from one machine to multiple devices, nodes, and future placements.",
      "portable execution field",
      "scheduler · residency · evidence",
      "machine → node → fleet",
    ],
  };
  const nodes = [...constellation.querySelectorAll("[data-surface]")],
    inspector = constellation.querySelector(".constellation-inspector"),
    set = (key) => {
      const d = data[key];
      if (!d) return;
      nodes.forEach((node) =>
        node.classList.toggle("is-active", node.dataset.surface === key),
      );
      inspector.querySelector(".constellation-kicker").textContent = d[0];
      inspector.querySelector("h3").textContent = d[1];
      inspector.querySelector("p").textContent = d[2];
      const values = inspector.querySelectorAll("dd");
      values[0].textContent = d[3];
      values[1].textContent = d[4];
      inspector.querySelector(".constellation-trace").textContent = d[5];
      document.body.dataset.prismSurface = key;
    };
  nodes.forEach((node) => {
    node.addEventListener("click", () => set(node.dataset.surface));
    node.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        set(node.dataset.surface);
      }
    });
  });
  set("cimage");
};
const initStoryHooks = () => {
  const hooks = [
    "The field is where intent becomes visible.",
    "The graph is where structure becomes law.",
    "Compilation is only the beginning.",
    "An artifact without execution is only potential.",
    "A route without evidence is only a guess.",
    "Representation is only meaningful when it can travel.",
    "The future must earn its proof.",
    "The field becomes a fabric.",
  ];
  const current = currentChapter(),
    next = document.querySelector(".chapter-nav a:last-child");
  if (next && hooks[current] && current < chapterMap.length - 1) {
    next.querySelector("small").textContent = hooks[current];
    next.querySelector("strong").textContent =
      `${chapterMap[current + 1][0]} →`;
  }
};

export const createSiteShellSystem = () => {
  const start = (context) => {
    const domRuntime = context?.domRuntime;
    const kernel = context?.kernel;
    initDiscoveryRooms(domRuntime);
    initStoryHooks();
    initMythology(domRuntime);
    initActCorrection();
    initSignatureMoment(domRuntime);
    initComputeImageSignature(kernel, domRuntime);
    initStartHereSignature(domRuntime);
    initPrismStages();
    initRepresentationLayers();
    initLivingArchitecture();
    initLivingAtlas();
    initAtlasScroll(kernel);
    initSurfaceConstellation();
    initPrismAtmosphere(kernel);
    domRuntime?.claim('site-shell', '.mythology-marker, .signature-moment, .discovery-room');
    return { stop() {} };
  };
  return { start };
};

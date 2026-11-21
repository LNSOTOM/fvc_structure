(() => {
  const root = document.getElementById("model-comparison");
  if (!root) return;
  const scenes = window.FVC_MODEL_COMPARISONS;
  const status = root.querySelector(".comparison-status");
  if (!Array.isArray(scenes) || !scenes.length) {
    status.textContent = "Comparison data could not be loaded.";
    return;
  }

  const select = root.querySelector("#comparison-scene");
  const density = root.querySelector("#comparison-density");
  const opacity = root.querySelector("#comparison-opacity");
  const legend = root.querySelector(".comparison-legend");
  const details = root.querySelector(".comparison-details");
  const inspector = root.querySelector(".comparison-inspector");
  const panels = [...root.querySelectorAll("canvas")];
  const sliders = [...root.querySelectorAll(".comparison-range input")];
  const semanticPalettes = {
    paper: ["#d9b449", "#693814", "#2d8a56"],
    accessible: ["#0072b2", "#e69f00", "#009e73"]
  };
  const instanceColors = ["#0072b2", "#e69f00", "#009e73", "#cc79a7", "#d55e00", "#56b4e9", "#f0e442", "#6a3d9a", "#006d77", "#9c6644", "#4464ad", "#a05195"];
  let scene = scenes[0];
  let split = 0.5;
  let palette = "paper";
  let yaw = -0.65;
  let pitch = 0.65;
  let zoom = 1;
  let projected = [];
  let drag = null;
  let frame = null;

  for (const item of scenes) {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = item.site === "forest"
      ? `New Zealand · SCION plot ${item.tile}`
      : `${item.site === "medium" ? "Medium" : "Low"} site · tile ${item.tile}`;
    select.append(option);
  }

  function className(id) {
    const names = scene.site === "forest"
      ? ["Ground", "Wood", "Leaf"]
      : scene.site === "medium"
        ? ["Ground / grass hummock", "Wood / shrub", "Leaf / mallee"]
        : ["Ground / forb", "Wood / shrub", "Leaf / mallee"];
    return names[id] || `Unknown (${id})`;
  }

  function color(point, reference, mode) {
    if (mode === "semantic") return semanticPalettes[palette][point[reference ? 3 : 4]] || "#8b9391";
    const id = point[reference ? 5 : 6];
    if (reference ? id <= 0 : id < 0) return "#c7cecb";
    return instanceColors[id % instanceColors.length];
  }

  function updateDescription() {
    const experiment = scene.site === "forest"
      ? "Experiment 1A, forest-trained model (epoch 3000)"
      : `Experiment 2B ${scene.site === "medium" ? "stage 1, medium" : "stage 2, low"}; final epoch 3000`;
    const shown = Math.min(Number(density.value), scene.points.length);
    details.textContent = `${experiment}. ${shown.toLocaleString()} of ${scene.total.toLocaleString()} points shown. Full file: ${scene.reference_instances} reference and ${scene.predicted_instances} predicted instance IDs. Source: ${scene.source}.`;
    root.querySelector(".comparison-counts").textContent = `Full-file semantic counts (IDs 0 / 1 / 2): reference ${scene.semantic_counts[0].map(n => n.toLocaleString()).join(" / ")}; prediction ${scene.semantic_counts[1].map(n => n.toLocaleString()).join(" / ")}.`;
    legend.replaceChildren();
    const entries = [
      ...semanticPalettes[palette].map((value, i) => [value, `${i}: ${className(i)}`]),
      ["#c7cecb", "Instance: unassigned"],
      [instanceColors[0], "Instance IDs (colours repeat)"]
    ];
    for (const [value, label] of entries) {
      const item = document.createElement("span");
      const swatch = document.createElement("span");
      swatch.className = "comparison-swatch";
      swatch.style.backgroundColor = value;
      item.append(swatch, document.createTextNode(label));
      legend.append(item);
    }
    status.textContent = scene.reference_instances === 0
      ? `No reference instances in this scene. The prediction file reports ${scene.predicted_instances} instances.`
      : "Reference and prediction instance IDs are independent; matching colours do not identify matching crowns.";
    inspector.textContent = "Point inspection: hover over either view.";
  }

  function draw() {
    frame = null;
    const points = scene.points.slice(0, Number(density.value));
    const centers = [0, 1, 2].map(axis => {
      let min = Infinity;
      let max = -Infinity;
      for (const point of scene.points) {
        min = Math.min(min, point[axis]);
        max = Math.max(max, point[axis]);
      }
      return [(min + max) / 2, max - min];
    });
    const span = Math.max(...centers.map(value => value[1]), 1);
    const cy = Math.cos(yaw), sy = Math.sin(yaw);
    const cp = Math.cos(pitch), sp = Math.sin(pitch);
    projected = points.map(point => {
      const x = point[0] - centers[0][0];
      const y = point[1] - centers[1][0];
      const z = point[2] - centers[2][0];
      const across = cy * x - sy * y;
      const along = sy * x + cy * y;
      return {point, x: across, y: cp * z - sp * along, depth: sp * z + cp * along};
    }).sort((a, b) => a.depth - b.depth);

    panels.forEach((canvas, index) => {
      const rect = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.round(rect.width * ratio);
      canvas.height = Math.round(rect.height * ratio);
      const context = canvas.getContext("2d");
      if (!context) {
        status.textContent = "This browser cannot display the point-cloud canvas.";
        return;
      }
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.fillStyle = "#fff";
      context.fillRect(0, 0, rect.width, rect.height);
      const scale = Math.min(rect.width, rect.height) * 0.82 * zoom / span;
      const boundary = rect.width * split;
      context.globalAlpha = Number(opacity.value) / 100;
      for (const item of projected) {
        const x = rect.width / 2 + item.x * scale;
        const y = rect.height / 2 - item.y * scale;
        if (x < boundary) {
          const width = Math.min(2, boundary - x);
          context.fillStyle = color(item.point, index === 0, "semantic");
          context.fillRect(x, y, width, 2);
          if (width < 2) {
            context.fillStyle = color(item.point, index === 0, "instance");
            context.fillRect(boundary, y, 2 - width, 2);
          }
        } else {
          context.fillStyle = color(item.point, index === 0, "instance");
          context.fillRect(x, y, 2, 2);
        }
      }
      context.globalAlpha = 1;
      if (split > 0 && split < 1) {
        context.strokeStyle = "rgba(16, 44, 43, 0.6)";
        context.lineWidth = 2;
        context.beginPath();
        context.moveTo(rect.width * split, 0);
        context.lineTo(rect.width * split, rect.height);
        context.stroke();
      }
      canvas.dataset.scale = scale;
    });
  }

  function redraw() {
    if (frame === null) frame = requestAnimationFrame(draw);
  }

  select.addEventListener("change", () => {
    scene = scenes.find(item => item.id === select.value);
    yaw = -0.65;
    pitch = 0.65;
    zoom = 1;
    updateDescription();
    redraw();
  });
  density.addEventListener("change", () => {
    updateDescription();
    redraw();
  });
  opacity.addEventListener("input", () => {
    root.querySelector("#comparison-opacity-value").textContent = `${opacity.value}%`;
    redraw();
  });
  function setSplit(value) {
    split = Math.max(0, Math.min(1, value));
    sliders.forEach(slider => {
      slider.value = Math.round(split * 100);
      slider.setAttribute("aria-valuetext", `${Math.round(split * 100)}% semantic, ${Math.round((1 - split) * 100)}% instance`);
    });
    redraw();
  }
  sliders.forEach(slider => slider.addEventListener("input", () => setSplit(Number(slider.value) / 100)));
  root.querySelectorAll("[data-palette]").forEach(button => button.addEventListener("click", () => {
    palette = button.dataset.palette;
    root.querySelectorAll("[data-palette]").forEach(item => item.setAttribute("aria-pressed", String(item === button)));
    updateDescription();
    redraw();
  }));
  root.querySelectorAll("[data-action]").forEach(button => button.addEventListener("click", () => {
    switch (button.dataset.action) {
      case "left": yaw -= 0.2; break;
      case "right": yaw += 0.2; break;
      case "up": pitch = Math.min(1.4, pitch + 0.15); break;
      case "down": pitch = Math.max(-1.4, pitch - 0.15); break;
      case "in": zoom = Math.min(3, zoom * 1.2); break;
      case "out": zoom = Math.max(0.5, zoom / 1.2); break;
      case "reset": yaw = -0.65; pitch = 0.65; zoom = 1; break;
    }
    redraw();
  }));
  panels.forEach((canvas, index) => {
    canvas.addEventListener("pointerdown", event => {
      const rect = canvas.getBoundingClientRect();
      const onDivider = split > 0 && split < 1 && Math.abs(event.clientX - rect.left - rect.width * split) <= 15;
      drag = {x: event.clientX, y: event.clientY, onDivider};
      canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener("pointerup", () => { drag = null; });
    canvas.addEventListener("pointercancel", () => { drag = null; });
    canvas.addEventListener("pointermove", event => {
      if (drag) {
        if (drag.onDivider) {
          const rect = canvas.getBoundingClientRect();
          setSplit((event.clientX - rect.left) / rect.width);
        } else {
          yaw += (event.clientX - drag.x) * 0.01;
          pitch = Math.max(-1.4, Math.min(1.4, pitch + (event.clientY - drag.y) * 0.01));
          redraw();
        }
        drag.x = event.clientX;
        drag.y = event.clientY;
        return;
      }
      const rect = canvas.getBoundingClientRect();
      canvas.style.cursor = split > 0 && split < 1 && Math.abs(event.clientX - rect.left - rect.width * split) <= 15
        ? "col-resize" : "grab";
      const scale = Number(canvas.dataset.scale);
      let nearest = null, distance = 64;
      for (const item of projected) {
        const dx = rect.width / 2 + item.x * scale - (event.clientX - rect.left);
        const dy = rect.height / 2 - item.y * scale - (event.clientY - rect.top);
        const current = dx * dx + dy * dy;
        if (current < distance) { distance = current; nearest = item.point; }
      }
      if (nearest) {
        const sem = nearest[index === 0 ? 3 : 4];
        const id = nearest[index === 0 ? 5 : 6];
        const assigned = index === 0 ? id > 0 : id >= 0;
        inspector.textContent = `${index === 0 ? "Reference" : "Prediction"}: semantic ${sem} (${className(sem)}); instance ${assigned ? id : "unassigned"}.`;
      } else {
        inspector.textContent = "Point inspection: hover over either view.";
      }
    });
  });
  window.addEventListener("resize", redraw);
  updateDescription();
  setSplit(0.5);
})();

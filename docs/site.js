"use strict";

const dimensions = {dinov2_b: 768, dinov2_g: 1536, dinov3_7b: 4096, franca_cls: 1536, franca_mean: 1536, ibot_swin: 768, ibot_b: 768, mpnet: 768, qwen3_embedding: 4096, qwen3_generative: 4096};
const vision = document.querySelector("#vision-select");
const text = document.querySelector("#text-select");
function updateInterface() {
  document.querySelector("#map-dimension").textContent = `${dimensions[vision.value]} × ${dimensions[text.value]}`;
  document.querySelector("#interface-code").textContent = `model.interface("${vision.value}", "${text.value}")`;
}
vision.addEventListener("change", updateInterface);
text.addEventListener("change", updateInterface);

const examples = {
  smoke: {
    code: "git clone https://github.com/xuanhuayin/ConsensusAlign.git\ncd ConsensusAlign\npython -m venv .venv\nsource .venv/bin/activate\npython -m pip install torch==2.10.0 \\\n  --index-url https://download.pytorch.org/whl/cpu\npython -m pip install -e \".[data]\"\npython examples/smoke.py --output smoke_output",
    note: "A small synthetic check of the actual implementation. No feature download required."
  },
  coco: {
    code: "consensusalign-data download \\\n  --manifest configs/coco_train.json configs/coco_val.json \\\n  --revision b39767e18cdb5031020a0e1c4b8d9fba6ae13453\n\nconsensusalign-reproduce \\\n  --plan configs/reproduce_coco.json \\\n  --output outputs/coco",
    note: "Five seeds, one pooled bridge per seed. The runner validates input hashes and resumes matching runs."
  },
  anchors: {
    code: "consensusalign fit \\\n  --manifest configs/coco_train.json \\\n  --anchors 10 \\\n  --seed-index 0 \\\n  --output outputs/anchors.pt",
    note: "The same paired anchors are shared across the encoder collection. The base construction uses no pairs."
  }
};
const tabs = [...document.querySelectorAll("[data-example]")];
function setExample(tab) {
  for (const item of tabs) {
    const selected = item === tab;
    item.setAttribute("aria-selected", String(selected));
    item.tabIndex = selected ? 0 : -1;
  }
  document.querySelector("#run-command").textContent = examples[tab.dataset.example].code;
  document.querySelector("#example-note").textContent = examples[tab.dataset.example].note;
  document.querySelector("#run-code").setAttribute("aria-labelledby", tab.id);
}
tabs.forEach((tab, index) => {
  tab.addEventListener("click", () => setExample(tab));
  tab.addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const offset = event.key === "ArrowRight" ? 1 : -1;
    const next = event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1 : (index + offset + tabs.length) % tabs.length;
    setExample(tabs[next]);
    tabs[next].focus();
  });
});
setExample(tabs[0]);

for (const button of document.querySelectorAll("[data-copy]")) {
  button.addEventListener("click", async () => {
    const value = document.getElementById(button.dataset.copy).textContent.trim();
    try {
      await navigator.clipboard.writeText(value);
      const original = button.textContent;
      button.textContent = "Copied";
      document.querySelector("#copy-status").textContent = "Copied to clipboard.";
      window.setTimeout(() => {button.textContent = original;}, 1800);
    } catch {
      const range = document.createRange();
      range.selectNodeContents(document.getElementById(button.dataset.copy));
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      document.querySelector("#copy-status").textContent = "Text selected. Copy it using your browser.";
    }
  });
}

const dialog = document.querySelector("#figure-dialog");
let zoomTrigger = null;
document.querySelectorAll("[data-zoom]").forEach((button) => {
  button.addEventListener("click", () => {
    zoomTrigger = button;
    const enlarged = document.querySelector("#expanded-figure");
    enlarged.src = button.dataset.zoom;
    enlarged.alt = button.querySelector("img").alt;
    dialog.showModal();
  });
});
document.querySelector("#close-figure").addEventListener("click", () => dialog.close());
dialog.addEventListener("click", (event) => {if (event.target === dialog) dialog.close();});
dialog.addEventListener("close", () => {if (zoomTrigger) zoomTrigger.focus();});

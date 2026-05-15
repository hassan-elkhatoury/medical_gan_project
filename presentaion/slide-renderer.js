(function () {
  const root = document.getElementById("slide-root");
  if (!root) return;

  const slides = window.PRESENTATION_SLIDES || [];
  const slideId = Number(root.dataset.slide || new URLSearchParams(location.search).get("slide") || 1);
  const slide = slides.find((item) => item.id === slideId) || slides[0];
  const total = slides.length;
  const embedded = window.self !== window.top;
  if (window.self !== window.top) {
    document.body.classList.add("embedded-slide");
  }
  const badgeList = (items) => items.map((item) => `<span class="tech-badge">${item}</span>`).join("");
  const highlight = (text) => String(text)
    .replaceAll("COVID-19", "<mark>COVID-19</mark>")
    .replaceAll("GAN", "<mark>GAN</mark>")
    .replaceAll("CNN", "<mark>CNN</mark>")
    .replaceAll("Flask", "<mark>Flask</mark>")
    .replaceAll("Conditional DCGAN", "<mark>Conditional DCGAN</mark>")
    .replaceAll("SimpleCNN", "<mark>SimpleCNN</mark>");
  const team = (window.TEAM_MEMBERS || []).map((name) => `<span>${name}</span>`).join("");

  document.title = `Slide ${slide.id} - ${slide.title}`;
  root.innerHTML = `
    <section class="single-slide text-only-slide slide-${slide.id}">
      <div class="slide-bg-grid"></div>
      <div class="slide-glow-orbit"></div>
      <div class="slide-ghost-number">${String(slide.id).padStart(2, "0")}</div>
      <div class="slide-topline">
        <span>${slide.kicker}</span>
        <span>${String(slide.id).padStart(2, "0")} / ${total}</span>
      </div>
      <div class="slide-layout">
        <article class="slide-copy">
          <div class="badge-row">${badgeList(slide.tags || [])}</div>
          <h1>${highlight(slide.title)}</h1>
          <p class="slide-subtitle">${highlight(slide.subtitle || "")}</p>
          <ul>${(slide.bullets || []).map((bullet) => `<li>${highlight(bullet)}</li>`).join("")}</ul>
          ${slide.id === 1 ? `<div class="slide-team">${team}</div>` : ""}
        </article>
      </div>
      <div class="slide-bottomline">
        <span>Deep Learning Module</span>
        <span>COVID-19 Radiography Database</span>
      </div>
      <div class="slide-footer">
        <button class="mini-control" id="prevSlide">Précédent</button>
        <button class="mini-control" id="toggleNotes">Afficher les notes</button>
        <button class="mini-control" id="nextSlide">Suivant</button>
      </div>
      <div class="notes-panel" id="notesPanel">
        <b>Notes orales</b>
        <p>${slide.note || ""}</p>
      </div>
    </section>
  `;

  const go = (delta) => {
    const next = Math.min(total, Math.max(1, slide.id + delta));
    if (next === slide.id) return;
    if (embedded) {
      window.parent.postMessage({ type: "presentation-slide-move", delta }, "*");
    } else {
      location.href = `slide${next}.html`;
    }
  };

  document.getElementById("prevSlide").addEventListener("click", () => go(-1));
  document.getElementById("nextSlide").addEventListener("click", () => go(1));
  document.getElementById("toggleNotes").addEventListener("click", () => {
    document.getElementById("notesPanel").classList.toggle("open");
  });
  window.addEventListener("keydown", (event) => {
    if (event.key === "ArrowLeft") go(-1);
    if (event.key === "ArrowRight") go(1);
    if (event.key.toLowerCase() === "n") document.getElementById("notesPanel").classList.toggle("open");
  });
})();

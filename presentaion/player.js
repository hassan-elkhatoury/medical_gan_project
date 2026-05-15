(function () {
  const slideFrame = document.getElementById("slide-frame");
  const prevBtn = document.getElementById("prev-btn");
  const nextBtn = document.getElementById("next-btn");
  const slideInfo = document.getElementById("slide-info");
  const progressBar = document.getElementById("progress-bar");
  const fullscreenBtn = document.getElementById("fullscreen-btn");
  const slides = window.PRESENTATION_SLIDES || [];

  let currentSlide = 1;
  const totalSlides = slides.length || 30;

  function updateSlide() {
    slideFrame.src = `slide${currentSlide}.html`;
    slideInfo.textContent = `${currentSlide} / ${totalSlides}`;
    progressBar.style.width = `${(currentSlide / totalSlides) * 100}%`;
    prevBtn.disabled = currentSlide === 1;
    nextBtn.disabled = currentSlide === totalSlides;
    window.location.hash = `slide${currentSlide}`;
  }

  function nextSlide() {
    if (currentSlide < totalSlides) {
      currentSlide += 1;
      updateSlide();
    }
  }

  function prevSlide() {
    if (currentSlide > 1) {
      currentSlide -= 1;
      updateSlide();
    }
  }

  function toggleFullscreen() {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen?.();
    } else {
      document.exitFullscreen?.();
    }
  }

  function handleKey(event) {
    if (event.key === "ArrowRight") nextSlide();
    if (event.key === "ArrowLeft") prevSlide();
    if (event.key === "f" || event.key === "F") toggleFullscreen();
  }

  window.addEventListener("keydown", handleKey);

  slideFrame.addEventListener("load", () => {
    try {
      slideFrame.contentWindow.focus();
      slideFrame.contentWindow.addEventListener("keydown", handleKey);
    } catch (error) {
      console.warn("Impossible d'attacher les raccourcis au slide.", error);
    }
  });

  window.addEventListener("message", (event) => {
    if (event.data?.type === "presentation-slide-move") {
      if (event.data.delta > 0) nextSlide();
      if (event.data.delta < 0) prevSlide();
    }
  });

  document.addEventListener("fullscreenchange", () => {
    document.body.classList.toggle("is-fullscreen", Boolean(document.fullscreenElement));
  });

  const hash = window.location.hash;
  if (hash && hash.startsWith("#slide")) {
    const slideNum = Number(hash.replace("#slide", ""));
    if (!Number.isNaN(slideNum) && slideNum >= 1 && slideNum <= totalSlides) {
      currentSlide = slideNum;
    }
  }

  prevBtn.addEventListener("click", prevSlide);
  nextBtn.addEventListener("click", nextSlide);
  fullscreenBtn.addEventListener("click", toggleFullscreen);

  updateSlide();
})();

/* ============================================
   shuaibo919.github.io — Main JS
   Fluid grid — responsive / DPR-aware / dark purple
   ============================================ */

class FluidSim {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');

    this._tier = -1;
    this._allocGrid();          // picks GX/GY based on viewport width
    this._allocArrays();

    this.dt = 0.15;
    this.dissipation = 0.9985;
    this.velDamping = 0.997;
    this.jacobiIters = 20;

    this.mx = -999;  this.my = -999;
    this.pmx = -999; this.pmy = -999;

    this.resize();
    this._bindMouse();
  }

  // ---- pick grid density by viewport width ----
  _allocGrid() {
    const w = window.innerWidth;
    let gx, gy, tier;
    if (w < 768)      { gx = 50; gy = 30; tier = 0; }
    else if (w < 1400) { gx = 80; gy = 48; tier = 1; }
    else               { gx = 100; gy = 60; tier = 2; }

    if (tier === this._tier) return false;
    this._tier = tier;
    this.GX = gx; this.GY = gy; this.N = gx * gy;
    return true;
  }

  _allocArrays() {
    this.u   = new Float32Array(this.N);
    this.v   = new Float32Array(this.N);
    this.u0  = new Float32Array(this.N);
    this.v0  = new Float32Array(this.N);
    this.d   = new Float32Array(this.N);
    this.d0  = new Float32Array(this.N);
    this.div = new Float32Array(this.N);
    this.p   = new Float32Array(this.N);
  }

  _ix(i, j) { return i + j * this.GX; }

  _sample(f, x, y) {
    x = x < 0.5 ? 0.5 : (x > this.GX - 1.5 ? this.GX - 1.5 : x);
    y = y < 0.5 ? 0.5 : (y > this.GY - 1.5 ? this.GY - 1.5 : y);
    const i0 = x | 0, j0 = y | 0, i1 = i0 + 1, j1 = j0 + 1;
    const fx = x - i0, fy = y - j0;
    return (1 - fx) * (1 - fy) * f[this._ix(i0, j0)] +
           fx       * (1 - fy) * f[this._ix(i1, j0)] +
           (1 - fx) * fy       * f[this._ix(i0, j1)] +
           fx       * fy       * f[this._ix(i1, j1)];
  }

  _advect(dst, src, uf, vf, dt) {
    const dtx = dt * this.GX, dty = dt * this.GY;
    for (let j = 0; j < this.GY; j++)
      for (let i = 0; i < this.GX; i++) {
        const k = this._ix(i, j);
        dst[k] = this._sample(src, i - dtx * uf[k], j - dty * vf[k]);
      }
  }

  _project() {
    const h = 1.0;
    for (let j = 1; j < this.GY - 1; j++)
      for (let i = 1; i < this.GX - 1; i++) {
        const k = this._ix(i, j);
        this.div[k] = -0.5 * h * (
          this.u[this._ix(i + 1, j)] - this.u[this._ix(i - 1, j)] +
          this.v[this._ix(i, j + 1)] - this.v[this._ix(i, j - 1)]);
        this.p[k] = 0;
      }
    for (let iter = 0; iter < this.jacobiIters; iter++)
      for (let j = 1; j < this.GY - 1; j++)
        for (let i = 1; i < this.GX - 1; i++) {
          const k = this._ix(i, j);
          this.p[k] = (this.div[k] + this.p[this._ix(i - 1, j)] + this.p[this._ix(i + 1, j)]
            + this.p[this._ix(i, j - 1)] + this.p[this._ix(i, j + 1)]) * 0.25;
        }
    for (let j = 1; j < this.GY - 1; j++)
      for (let i = 1; i < this.GX - 1; i++) {
        const k = this._ix(i, j);
        this.u[k] -= 0.5 * (this.p[this._ix(i + 1, j)] - this.p[this._ix(i - 1, j)]) / h;
        this.v[k] -= 0.5 * (this.p[this._ix(i, j + 1)] - this.p[this._ix(i, j - 1)]) / h;
      }
  }

  _addSource(px, py, dx, dy) {
    const gi = (px / this.cw) * this.GX;
    const gj = (py / this.ch) * this.GY;
    const R = 6;
    const i0 = Math.round(gi), j0 = Math.round(gj);
    for (let dj = -R; dj <= R; dj++)
      for (let di = -R; di <= R; di++) {
        const ni = i0 + di, nj = j0 + dj;
        if (ni < 0 || ni >= this.GX || nj < 0 || nj >= this.GY) continue;
        const dd = Math.sqrt(di * di + dj * dj);
        if (dd > R) continue;
        const w = (1 - dd / R);
        const k = this._ix(ni, nj);
        this.d0[k] += 0.15 * w;
        this.u0[k] += dx * w * 0.03;
        this.v0[k] += dy * w * 0.03;
      }
  }

  step() {
    this._advect(this.u, this.u0, this.u0, this.v0, this.dt);
    this._advect(this.v, this.v0, this.u0, this.v0, this.dt);
    this._project();
    this._advect(this.d, this.d0, this.u, this.v, this.dt);
    for (let k = 0; k < this.N; k++) {
      this.u0[k] = this.u[k] * this.velDamping;
      this.v0[k] = this.v[k] * this.velDamping;
      this.d0[k] = this.d[k] * this.dissipation;
    }
  }

  // ---- dark purple per-edge render ----
  render() {
    const ctx = this.ctx;
    const cw = this.cw, ch = this.ch;

    ctx.fillStyle = '#08080c';
    ctx.fillRect(0, 0, cw, ch);

    const cellW = cw / this.GX;
    const cellH = ch / this.GY;
    const thr = 0.006, maxA = 0.85;
    const d = this.d;
    const ix = (i, j) => this._ix(i, j);

    // colours — dark purple
    const glowR = 110, glowG = 55, glowB = 170;
    const coreR = 175, coreG = 105, coreB = 230;

    for (let j = 1; j < this.GY; j++) {
      const y = j * cellH;
      for (let i = 0; i < this.GX; i++) {
        const a = Math.min((d[ix(i, j - 1)] + d[ix(i, j)]) * 0.45, maxA);
        if (a <= thr) continue;
        const x1 = i * cellW, x2 = (i + 1) * cellW;

        ctx.beginPath();
        ctx.strokeStyle = `rgba(${glowR},${glowG},${glowB},${(a * 0.3).toFixed(3)})`;
        ctx.lineWidth = 2.5;
        ctx.moveTo(x1, y); ctx.lineTo(x2, y); ctx.stroke();

        ctx.beginPath();
        ctx.strokeStyle = `rgba(${coreR},${coreG},${coreB},${a.toFixed(3)})`;
        ctx.lineWidth = 0.7;
        ctx.moveTo(x1, y); ctx.lineTo(x2, y); ctx.stroke();
      }
    }

    for (let i = 1; i < this.GX; i++) {
      const x = i * cellW;
      for (let j = 0; j < this.GY; j++) {
        const a = Math.min((d[ix(i - 1, j)] + d[ix(i, j)]) * 0.45, maxA);
        if (a <= thr) continue;
        const y1 = j * cellH, y2 = (j + 1) * cellH;

        ctx.beginPath();
        ctx.strokeStyle = `rgba(${glowR},${glowG},${glowB},${(a * 0.3).toFixed(3)})`;
        ctx.lineWidth = 2.5;
        ctx.moveTo(x, y1); ctx.lineTo(x, y2); ctx.stroke();

        ctx.beginPath();
        ctx.strokeStyle = `rgba(${coreR},${coreG},${coreB},${a.toFixed(3)})`;
        ctx.lineWidth = 0.7;
        ctx.moveTo(x, y1); ctx.lineTo(x, y2); ctx.stroke();
      }
    }
  }

  _bindMouse() {
    document.addEventListener('mousemove', (e) => {
      this.pmx = this.mx; this.pmy = this.my;
      this.mx = e.clientX; this.my = e.clientY;
      if (this.pmx > -900) {
        const dx = this.mx - this.pmx, dy = this.my - this.pmy;
        const steps = Math.max(1, (Math.sqrt(dx * dx + dy * dy) / 6) | 0);
        for (let s = 0; s < steps; s++) {
          const t = s / steps;
          this._addSource(this.pmx + dx * t, this.pmy + dy * t, dx * 0.015, dy * 0.015);
        }
      }
    });
  }

  // ---- DPR-aware resize + grid tier switching ----
  resize() {
    const dpr = window.devicePixelRatio || 1;
    this.cw = window.innerWidth;
    this.ch = window.innerHeight;

    this.canvas.width  = this.cw * dpr;
    this.canvas.height = this.ch * dpr;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // switch grid density if crossing breakpoint
    if (this._allocGrid()) {
      this._allocArrays();
    }
  }
}

// ---------- Init ----------
function initFluid() {
  const canvas = document.getElementById('ink-canvas');
  if (!canvas) return;
  const sim = new FluidSim(canvas);
  (function loop() { sim.step(); sim.render(); requestAnimationFrame(loop); })();

  let resizeTimer;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => sim.resize(), 120);
  });
}

function initScrollReveal() {
  const obs = new IntersectionObserver(
    (e) => { e.forEach(en => { if (en.isIntersecting) en.target.classList.add('visible'); }); },
    { threshold: 0.12, rootMargin: '0px 0px -30px 0px' }
  );
  document.querySelectorAll('.reveal').forEach(el => obs.observe(el));
}

function initNav() {
  const nav = document.querySelector('.nav');
  const secs = document.querySelectorAll('section[id]');
  const links = document.querySelectorAll('.nav-links a');
  window.addEventListener('scroll', () => {
    nav.classList.toggle('scrolled', window.scrollY > 30);
    let cur = '';
    secs.forEach(s => { if (window.scrollY >= s.offsetTop - 80) cur = s.id; });
    links.forEach(l => l.classList.toggle('active', l.getAttribute('href') === '#' + cur));
  });
}

function initTilt() {
  document.querySelectorAll('.project-card, .timeline-card').forEach(card => {
    card.addEventListener('mousemove', (e) => {
      const r = card.getBoundingClientRect(), cx = r.width / 2, cy = r.height / 2;
      card.style.transform = `perspective(800px) rotateX(${((e.clientY-r.top-cy)/cy)*-2.5}deg) rotateY(${((e.clientX-r.left-cx)/cx)*2.5}deg)`;
    });
    card.addEventListener('mouseleave', () => { card.style.transform = ''; });
  });
}

document.addEventListener('DOMContentLoaded', () => {
  initFluid(); initScrollReveal(); initNav(); initTilt();
});

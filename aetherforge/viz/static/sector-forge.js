/**
 * AetherForge Sector Forge — visual timeline, readiness rings, dataset shards,
 * and orbital train animation for sequential MoE sector training.
 */
(function (global) {
  'use strict';

  function esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function stateClass(st) {
    const s = (st || '').toLowerCase();
    if (s === 'trained' || s === 'dry_run' || s === 'done' || s === 'pass') return 'ok';
    if (s === 'running' || s === 'start') return 'live';
    if (s === 'blocked' || s === 'error' || s === 'fail' || s === 'failed') return 'fail';
    if (s === 'warn' || s === 'warning') return 'warn';
    if (s === 'skipped' || s === 'skip') return 'dim';
    return 'dim';
  }

  function stateLabel(st) {
    const s = (st || 'pending').toLowerCase();
    if (s === 'dry_run') return 'DRY';
    return s.toUpperCase();
  }

  function readinessPill(r) {
    const cls = stateClass(r);
    return `<span class="pill ${cls === 'dim' ? '' : cls}">${esc(r || '—')}</span>`;
  }

  /**
   * Build HTML for the Sector Forge command panel.
   */
  function renderSectorForgePanel(b) {
    const sf = b.sector_forge;
    const live = (b.live && b.live.sectors) || {};
    const visual = (b.live && b.live.visual) || {};
    if (!sf && !(live.items && live.items.length)) {
      return `
        <div class="panel sector-forge-panel" id="sectorForgePanel">
          <span class="hud-corner hud-tl"></span><span class="hud-corner hud-tr"></span>
          <span class="hud-corner hud-bl"></span><span class="hud-corner hud-br"></span>
          <h3>Sector Forge <span class="tag">awaiting sequential ESFT</span></h3>
          <div class="sub">Run with <code>training.sector_mode: sequential</code> (default) to light this bay.
            Each sector is forensically assessed, dataset-bound, then trained alone.</div>
          <div class="forge-empty-orbit" aria-hidden="true"></div>
        </div>`;
    }

    const items = (sf && sf.items) || live.items || [];
    const mode = (sf && sf.mode) || (b.summary && b.summary.sector_mode) || 'sequential';
    const wf = (sf && sf.workflow) || {};
    const nTrained = wf.n_trained != null ? wf.n_trained : live.n_trained || 0;
    const nBlocked = wf.n_blocked != null ? wf.n_blocked : live.n_blocked || 0;
    const nSkipped = wf.n_skipped != null ? wf.n_skipped : live.n_skipped || 0;
    const nTotal = live.n_total || items.length || 0;
    const nDone = live.n_done != null ? live.n_done : items.filter(i =>
      ['trained', 'dry_run', 'blocked', 'skipped', 'error'].includes(i.state)
    ).length;
    const pct = nTotal ? Math.round((nDone / nTotal) * 100) : 0;
    const overall = (sf && sf.readiness_overall) || live.overall || '—';
    const hero = visual.hero_label || 'SECTOR FORGE';
    const narrative = (sf && sf.readiness_narrative) || (sf && sf.forensics_narrative) || '';

    const timeline = items.map((it, i) => {
      const st = it.state || 'pending';
      const cls = stateClass(st);
      const color = it.color || 'var(--cyan)';
      return `
        <div class="sector-node ${cls}" style="--i:${i};--node-color:${color}" data-gid="${esc(it.group_id || '')}">
          <div class="node-ring">
            <div class="node-core"></div>
            <div class="node-index">${i + 1}</div>
          </div>
          <div class="node-body">
            <div class="node-title">${esc(it.name || it.group_id || 'sector')}</div>
            <div class="node-meta">
              <span class="pill ${cls}">${stateLabel(st)}</span>
              ${readinessPill(it.readiness)}
              <span class="chip">${it.n_experts || 0} exp</span>
              <span class="chip">${it.n_samples || 0} samp</span>
              ${it.domain ? `<span class="chip">${esc(it.domain)}</span>` : ''}
            </div>
            ${it.forensics_summary
              ? `<div class="node-forensics">${esc(it.forensics_summary)}</div>`
              : ''}
            ${it.error ? `<div class="gate-fail" style="font-size:11px;margin-top:4px">${esc(it.error)}</div>` : ''}
          </div>
          ${i < items.length - 1 ? '<div class="node-connector"></div>' : ''}
        </div>`;
    }).join('');

    const shards = (sf && sf.datasets && sf.datasets.shards) || [];
    const shardHtml = shards.length
      ? shards.map((s, i) => {
          const n = s.n_train || 0;
          const maxN = Math.max(...shards.map(x => x.n_train || 0), 1);
          const w = Math.min(100, (n / maxN) * 100);
          return `<div class="shard-row" style="--i:${i}">
            <div class="shard-name">${esc(s.name || s.group_id)}</div>
            <div class="track"><div class="fill good" data-w="${w}" style="width:0"></div></div>
            <div class="shard-n mono">${n}</div>
          </div>`;
        }).join('')
      : '<div class="sub">No sector dataset shards yet</div>';

    const readyRows = (sf && sf.readiness) || [];
    const readyHtml = readyRows.length
      ? `<div class="ready-grid">${readyRows.map((r, i) => `
          <div class="ready-card ${stateClass(r.status)}" style="--i:${i}">
            <div class="ready-top">
              <span>${esc(r.name || r.group_id)}</span>
              ${readinessPill(r.status)}
            </div>
            <div class="ready-score mono">${(r.score != null ? Number(r.score).toFixed(2) : '—')}</div>
            <div class="sub" style="font-size:11px;margin-top:4px">${esc((r.reasons || []).slice(0, 2).join(' · ') || 'ok')}</div>
          </div>`).join('')}</div>`
      : '';

    return `
      <div class="panel sector-forge-panel lit" id="sectorForgePanel">
        <span class="hud-corner hud-tl"></span><span class="hud-corner hud-tr"></span>
        <span class="hud-corner hud-bl"></span><span class="hud-corner hud-br"></span>
        <div class="forge-hero">
          <div class="forge-hero-label mono">${esc(hero)}</div>
          <div class="forge-hero-stats">
            <span class="stat mini"><span class="v">${esc(mode)}</span><span class="l">Mode</span></span>
            <span class="stat mini"><span class="v">${nTrained}/${nTotal || '—'}</span><span class="l">Trained</span></span>
            <span class="stat mini"><span class="v">${nBlocked}</span><span class="l">Blocked</span></span>
            <span class="stat mini"><span class="v">${nSkipped}</span><span class="l">Skipped</span></span>
            <span class="stat mini"><span class="v">${pct}%</span><span class="l">Wave</span></span>
            <span class="stat mini"><span class="v">${esc(String(overall).toUpperCase())}</span><span class="l">Readiness</span></span>
          </div>
        </div>
        <div class="progress-track forge-wave"><i id="sectorWaveFill" style="width:${pct}%"></i></div>
        ${narrative ? `<div class="sub forge-narrative">${esc(narrative)}</div>` : ''}

        <div class="forge-layout">
          <div class="forge-timeline-wrap">
            <h3 style="margin-bottom:10px">Pre-train forensics → ESFT wave <span class="tag">${items.length} sectors</span></h3>
            <div class="sector-timeline" id="sectorTimeline">${timeline || '<div class="sub">No sector steps</div>'}</div>
          </div>
          <div class="forge-side">
            <h3>Sector orbit <span class="tag">live</span></h3>
            <canvas id="sectorOrbitCanvas" width="420" height="280"></canvas>
            <h3 style="margin-top:14px">Dataset shards</h3>
            <div class="shard-list" id="shardList">${shardHtml}</div>
          </div>
        </div>
        ${readyHtml ? `<h3 style="margin-top:16px">Readiness dossiers</h3>${readyHtml}` : ''}
      </div>`;
  }

  let _orbitRAF = null;

  function drawSectorOrbit(b) {
    const canvas = document.getElementById('sectorOrbitCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const sf = b.sector_forge || {};
    const items = sf.items || (b.live && b.live.sectors && b.live.sectors.items) || [];
    const groups = (b.expert_groups && b.expert_groups.summary && b.expert_groups.summary.groups) || [];
    const colorOf = {};
    groups.forEach(g => { colorOf[g.id] = g.color; });
    items.forEach(it => {
      if (it.color) colorOf[it.group_id] = it.color;
    });

    const cx = W / 2, cy = H / 2;
    const n = Math.max(items.length, groups.length, 1);
    const baseR = Math.min(W, H) * 0.32;

    function frame(t) {
      const theme = document.documentElement.getAttribute('data-theme') || 'nexus';
      let bg = '#050a14', grid = 'rgba(0,229,255,0.06)', ink = 'rgba(0,229,255,0.5)';
      if (theme === 'matrix') { bg = '#020a02'; grid = 'rgba(0,255,65,0.07)'; ink = 'rgba(0,255,65,0.55)'; }
      if (theme === 'plasma') { bg = '#12040e'; grid = 'rgba(255,43,214,0.08)'; ink = 'rgba(255,107,157,0.55)'; }
      if (theme === 'aurora') { bg = '#040814'; grid = 'rgba(100,255,200,0.07)'; ink = 'rgba(120,200,255,0.55)'; }

      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, W, H);

      // radial grid
      ctx.strokeStyle = grid;
      for (let r = 30; r < baseR + 40; r += 28) {
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.stroke();
      }
      for (let a = 0; a < Math.PI * 2; a += Math.PI / 6) {
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(cx + Math.cos(a) * (baseR + 50), cy + Math.sin(a) * (baseR + 50));
        ctx.stroke();
      }

      // core pulse
      const pulse = 0.5 + 0.5 * Math.sin(t * 0.003);
      const grd = ctx.createRadialGradient(cx, cy, 2, cx, cy, 40 + pulse * 12);
      grd.addColorStop(0, ink);
      grd.addColorStop(1, 'transparent');
      ctx.fillStyle = grd;
      ctx.beginPath();
      ctx.arc(cx, cy, 40 + pulse * 12, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = ink;
      ctx.font = '10px JetBrains Mono';
      ctx.textAlign = 'center';
      ctx.fillText('MOE', cx, cy + 3);

      const source = items.length ? items : groups.map((g, i) => ({
        group_id: g.id, name: g.name, state: g.train ? 'pending' : 'skipped',
        color: g.color, n_experts: g.n_cells, active_fire_ratio: g.active_fire_ratio,
      }));

      source.forEach((it, i) => {
        const ang = (i / Math.max(source.length, 1)) * Math.PI * 2 - Math.PI / 2 + t * 0.00015;
        const fire = it.active_fire_ratio || (it.n_experts ? 0.5 : 0.3);
        const rr = baseR * (0.75 + 0.25 * Math.min(1, fire));
        const x = cx + Math.cos(ang) * rr;
        const y = cy + Math.sin(ang) * rr;
        const col = colorOf[it.group_id] || it.color || '#00e5ff';
        const st = (it.state || '').toLowerCase();
        const running = st === 'running';
        const done = st === 'trained' || st === 'dry_run';
        const blocked = st === 'blocked' || st === 'error';
        const rad = 10 + (running ? 4 * pulse : done ? 2 : 0);

        // tether
        ctx.strokeStyle = col + '55';
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(x, y);
        ctx.stroke();

        // glow
        ctx.shadowColor = col;
        ctx.shadowBlur = running ? 18 : done ? 12 : 6;
        ctx.fillStyle = col;
        ctx.globalAlpha = blocked ? 0.35 : 0.9;
        ctx.beginPath();
        ctx.arc(x, y, rad, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha = 1;
        ctx.shadowBlur = 0;

        // ring for readiness
        if (it.readiness === 'warn') {
          ctx.strokeStyle = '#ffc14a';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(x, y, rad + 4, 0, Math.PI * 2);
          ctx.stroke();
          ctx.lineWidth = 1;
        } else if (it.readiness === 'block' || blocked) {
          ctx.strokeStyle = '#ff4466';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(x, y, rad + 4, 0, Math.PI * 2);
          ctx.stroke();
          ctx.lineWidth = 1;
        }

        ctx.fillStyle = 'rgba(255,255,255,0.85)';
        ctx.font = '9px JetBrains Mono';
        ctx.textAlign = 'center';
        const label = (it.name || '').slice(0, 10);
        ctx.fillText(label, x, y + rad + 12);
      });

      _orbitRAF = requestAnimationFrame(() => frame(performance.now()));
    }

    if (_orbitRAF) cancelAnimationFrame(_orbitRAF);
    frame(performance.now());
  }

  function bindSectorForge(b) {
    requestAnimationFrame(() => {
      document.querySelectorAll('#shardList .fill[data-w]').forEach(el => {
        el.style.width = el.getAttribute('data-w') + '%';
      });
    });
    drawSectorOrbit(b);
    // click sector node → select in studio if present
    document.querySelectorAll('.sector-node[data-gid]').forEach(el => {
      el.addEventListener('click', () => {
        const gid = el.getAttribute('data-gid');
        if (!gid) return;
        if (typeof global.selectedGroupId !== 'undefined') {
          global.selectedGroupId = gid;
        }
        document.querySelectorAll('.sector-card').forEach(x => {
          x.classList.toggle('active', x.dataset.gid === gid);
        });
        if (typeof global.showGroupDetail === 'function') {
          global.showGroupDetail(gid);
        }
      });
    });
  }

  global.AetherSectorForge = {
    renderSectorForgePanel,
    bindSectorForge,
    drawSectorOrbit,
  };
})(window);

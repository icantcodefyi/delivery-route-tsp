// shortest-route-01 entry (Node 22, standard library only).
//
// For each instance /input/<id>.txt it writes /output/<id>.tour, a closed tour visiting every city once:
//   1. candidate lists: the K nearest cities of each city, found with a uniform grid;
//   2. a greedy-edge start (shortest candidate edges first, no city of degree 3, no early cycle), the fragments
//      joined nearest-endpoint first;
//   3. 2-opt + Or-opt local search on an array tour with neighbour lists and a don't-look queue;
//   4. an iterated local search: a local double-bridge kick (swap two short adjacent segments), local search from
//      the touched cities, keep the result when it is no longer, undo it otherwise.
// Everything is bounded by iteration counts and seeded by constants, so two runs write the same bytes.
//
// Outside the grader, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in for /input and /output.

import { readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const K = 10; // candidate neighbours per city
const KICKS_PER_CITY = 100; // iterated-local-search kicks per city (about 16 s for all eight sets on a laptop core)
const SEG_MAX = 50; // longest segment a kick moves

function readInstance(path) {
  const text = readFileSync(path, 'utf8');
  const lines = text.split('\n');
  const n = Number(lines[0]);
  const xs = new Float64Array(n);
  const ys = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    const line = lines[k + 1];
    const sp = line.indexOf(' ');
    xs[k] = Number(line.slice(0, sp));
    ys[k] = Number(line.slice(sp + 1));
  }
  return { n, xs, ys };
}

function solve(n, X, Y) {
  const dist = (a, b) => {
    const dx = X[a] - X[b];
    const dy = Y[a] - Y[b];
    return Math.round(Math.sqrt(dx * dx + dy * dy));
  };

  // ---- 1. K nearest neighbours with a grid ----
  const KK = Math.min(K, n - 1);
  const nb = new Int32Array(n * KK);
  const nbd = new Float64Array(n * KK);
  {
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (let i = 0; i < n; i++) {
      if (X[i] < minX) minX = X[i];
      if (X[i] > maxX) maxX = X[i];
      if (Y[i] < minY) minY = Y[i];
      if (Y[i] > maxY) maxY = Y[i];
    }
    const G = Math.max(1, Math.floor(Math.sqrt(n / 2)));
    const cw = (maxX - minX + 1) / G;
    const ch = (maxY - minY + 1) / G;
    const cellOf = new Int32Array(n);
    const cellStart = new Int32Array(G * G + 1);
    for (let i = 0; i < n; i++) {
      let cx = Math.floor((X[i] - minX) / cw);
      let cy = Math.floor((Y[i] - minY) / ch);
      if (cx >= G) cx = G - 1;
      if (cy >= G) cy = G - 1;
      const c = cy * G + cx;
      cellOf[i] = c;
      cellStart[c + 1]++;
    }
    for (let c = 0; c < G * G; c++) cellStart[c + 1] += cellStart[c];
    const fill = cellStart.slice(0, G * G);
    const items = new Int32Array(n);
    for (let i = 0; i < n; i++) items[fill[cellOf[i]]++] = i;
    const bd = new Float64Array(KK + 1);
    const bi = new Int32Array(KK + 1);
    const cmin = Math.min(cw, ch);
    for (let i = 0; i < n; i++) {
      const cx = cellOf[i] % G;
      const cy = (cellOf[i] - cx) / G;
      let cnt = 0;
      const xi = X[i], yi = Y[i];
      for (let r = 0; r <= G; r++) {
        const y0 = cy - r, y1 = cy + r, x0 = cx - r, x1 = cx + r;
        for (let yy = y0; yy <= y1; yy++) {
          if (yy < 0 || yy >= G) continue;
          const edge = yy === y0 || yy === y1;
          for (let xx = x0; xx <= x1; xx += edge ? 1 : x1 - x0 || 1) {
            if (xx >= 0 && xx < G) {
              const c = yy * G + xx;
              for (let t = cellStart[c]; t < cellStart[c + 1]; t++) {
                const j = items[t];
                if (j === i) continue;
                const dx = X[j] - xi, dy = Y[j] - yi;
                const d2 = dx * dx + dy * dy;
                if (cnt < KK || d2 < bd[cnt - 1]) {
                  let p = cnt < KK ? cnt++ : cnt - 1;
                  while (p > 0 && bd[p - 1] > d2) {
                    bd[p] = bd[p - 1];
                    bi[p] = bi[p - 1];
                    p--;
                  }
                  bd[p] = d2;
                  bi[p] = j;
                }
              }
            }
            if (x1 === x0) break;
          }
        }
        const reach = r * cmin;
        if (cnt >= KK && bd[cnt - 1] <= reach * reach) break;
      }
      for (let k = 0; k < KK; k++) {
        nb[i * KK + k] = bi[k];
        nbd[i * KK + k] = dist(i, bi[k]);
      }
    }
  }

  // ---- 2. greedy-edge start ----
  const tour = new Int32Array(n);
  const pos = new Int32Array(n);
  {
    const eu = [], ev = [], ew = [];
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < KK; k++) {
        const j = nb[i * KK + k];
        let keep = i < j;
        if (!keep) {
          keep = true;
          for (let t = 0; t < KK; t++) if (nb[j * KK + t] === i) { keep = false; break; }
        }
        if (keep) {
          eu.push(i < j ? i : j);
          ev.push(i < j ? j : i);
          ew.push(nbd[i * KK + k]);
        }
      }
    }
    const order = Array.from({ length: eu.length }, (_, e) => e);
    order.sort((a, b) => ew[a] - ew[b] || eu[a] - eu[b] || ev[a] - ev[b]);
    const deg = new Uint8Array(n);
    const link = new Int32Array(2 * n).fill(-1);
    const parent = new Int32Array(n);
    for (let i = 0; i < n; i++) parent[i] = i;
    const find = (x) => {
      while (parent[x] !== x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
      }
      return x;
    };
    for (const e of order) {
      const u = eu[e], v = ev[e];
      if (deg[u] >= 2 || deg[v] >= 2) continue;
      const ru = find(u), rv = find(v);
      if (ru === rv) continue;
      parent[ru] = rv;
      link[2 * u + deg[u]++] = v;
      link[2 * v + deg[v]++] = u;
    }
    const ends = [];
    for (let i = 0; i < n; i++) if (deg[i] < 2) ends.push(i);
    const seen = new Uint8Array(n);
    let len = 0;
    let cur = ends.length > 0 ? ends[0] : 0;
    while (true) {
      let prev = -1;
      let x = cur;
      while (x !== -1 && !seen[x]) {
        seen[x] = 1;
        tour[len++] = x;
        const a0 = link[2 * x], a1 = link[2 * x + 1];
        const nxt = a0 === prev ? a1 : a0;
        prev = x;
        x = nxt;
      }
      const last = tour[len - 1];
      let best = -1, bestD = Infinity;
      for (const e of ends) {
        if (seen[e]) continue;
        const dx = X[e] - X[last], dy = Y[e] - Y[last];
        const d2 = dx * dx + dy * dy;
        if (d2 < bestD) {
          bestD = d2;
          best = e;
        }
      }
      if (best === -1) break;
      cur = best;
    }
    if (len !== n) {
      // cannot happen; fall back to the identity order so the tour stays valid
      for (let i = 0; i < n; i++) tour[i] = i;
    }
    for (let i = 0; i < n; i++) pos[tour[i]] = i;
  }

  // ---- 3. local search machinery ----
  const succ = (c) => tour[pos[c] + 1 === n ? 0 : pos[c] + 1];
  const pred = (c) => tour[pos[c] === 0 ? n - 1 : pos[c] - 1];

  // undo log: every change rewrites one cyclic range of positions; its old contents are saved first
  let logging = false;
  let overflow = false;
  const OPS_CAP = 1 << 16;
  const BUF_CAP = 1 << 22;
  const logStart = new Int32Array(OPS_CAP);
  const logLen = new Int32Array(OPS_CAP);
  const buf = new Int32Array(BUF_CAP);
  let nOps = 0, bufTop = 0;
  const saveRange = (start, len) => {
    if (!logging || overflow) return;
    if (nOps >= OPS_CAP || bufTop + len > BUF_CAP) {
      overflow = true;
      return;
    }
    logStart[nOps] = start;
    logLen[nOps] = len;
    nOps++;
    let p = start;
    for (let t = 0; t < len; t++) {
      buf[bufTop++] = tour[p];
      if (++p === n) p = 0;
    }
  };
  const undoAll = () => {
    for (let o = nOps - 1; o >= 0; o--) {
      const len = logLen[o];
      let p = logStart[o];
      bufTop -= len;
      let b = bufTop;
      for (let t = 0; t < len; t++) {
        const c = buf[b++];
        tour[p] = c;
        pos[c] = p;
        if (++p === n) p = 0;
      }
    }
    nOps = 0;
    bufTop = 0;
  };

  // reverse the path of positions i..j (forward, cyclic), or its complement when that is shorter
  const reversePath = (i, j) => {
    let len = j - i;
    if (len < 0) len += n;
    len += 1;
    if (2 * len > n) {
      const ni = j + 1 === n ? 0 : j + 1;
      const nj = i === 0 ? n - 1 : i - 1;
      i = ni;
      j = nj;
      len = n - len;
    }
    saveRange(i, len);
    let a = i, b = j;
    for (let t = len >> 1; t > 0; t--) {
      const ca = tour[a], cb = tour[b];
      tour[a] = cb;
      pos[cb] = a;
      tour[b] = ca;
      pos[ca] = b;
      if (++a === n) a = 0;
      if (--b < 0) b = n - 1;
    }
  };

  // move the segment at positions i..i+L-1 into the gap after position k, reversed if rev
  const segTmp = new Int32Array(SEG_MAX + 8);
  const moveSegment = (i, L, k, rev) => {
    const j = (i + L - 1) % n;
    let df = k - j;
    if (df < 0) df += n;
    let db = i - 1 - k;
    if (db < 0) db += n;
    for (let t = 0; t < L; t++) {
      let q = i + (rev ? L - 1 - t : t);
      if (q >= n) q -= n;
      segTmp[t] = tour[q];
    }
    if (df <= db) {
      saveRange(i, df + L);
      let p = i;
      let q = j + 1 === n ? 0 : j + 1;
      for (let t = 0; t < df; t++) {
        const c = tour[q];
        tour[p] = c;
        pos[c] = p;
        if (++p === n) p = 0;
        if (++q === n) q = 0;
      }
      for (let t = 0; t < L; t++) {
        const c = segTmp[t];
        tour[p] = c;
        pos[c] = p;
        if (++p === n) p = 0;
      }
    } else {
      const start = k + 1 === n ? 0 : k + 1;
      saveRange(start, db + L);
      let p = j;
      let q = i === 0 ? n - 1 : i - 1;
      for (let t = 0; t < db; t++) {
        const c = tour[q];
        tour[p] = c;
        pos[c] = p;
        if (--p < 0) p = n - 1;
        if (--q < 0) q = n - 1;
      }
      p = start;
      for (let t = 0; t < L; t++) {
        const c = segTmp[t];
        tour[p] = c;
        pos[c] = p;
        if (++p === n) p = 0;
      }
    }
  };

  // don't-look queue
  const queue = new Int32Array(n);
  const inQ = new Uint8Array(n);
  let qh = 0, qt = 0, qs = 0;
  const push = (c) => {
    if (inQ[c]) return;
    inQ[c] = 1;
    queue[qt] = c;
    if (++qt === n) qt = 0;
    qs++;
  };

  // try to improve around city a; apply the move and return its (negative) delta, or 0
  const improveCity = (a) => {
    const pa = pos[a];
    const an = tour[pa + 1 === n ? 0 : pa + 1];
    const ap = tour[pa === 0 ? n - 1 : pa - 1];
    const dn = dist(a, an);
    const dp = dist(a, ap);
    // 2-opt, best over a's candidates
    let best = 0, bc = -1, btype = 0;
    const base = a * KK;
    for (let k = 0; k < KK; k++) {
      const dc = nbd[base + k];
      if (dc >= dn && dc >= dp) break;
      const c = nb[base + k];
      if (dc < dn) {
        const cn = succ(c);
        if (c !== an && cn !== a) {
          const delta = dc + dist(an, cn) - dn - dist(c, cn);
          if (delta < best) {
            best = delta;
            bc = c;
            btype = 1;
          }
        }
      }
      if (dc < dp) {
        const cp = pred(c);
        if (c !== ap && cp !== a) {
          const delta = dc + dist(ap, cp) - dp - dist(cp, c);
          if (delta < best) {
            best = delta;
            bc = c;
            btype = 2;
          }
        }
      }
    }
    if (btype === 1) {
      const cn = succ(bc);
      reversePath(pos[an], pos[bc]);
      push(a); push(an); push(bc); push(cn);
      return best;
    }
    if (btype === 2) {
      const cp = pred(bc);
      reversePath(pos[bc], pos[ap]);
      push(a); push(ap); push(bc); push(cp);
      return best;
    }
    // Or-opt: segments of 1..3 cities with a at one end
    for (let L = 1; L <= 3; L++) {
      for (let side = 0; side < (L === 1 ? 1 : 2); side++) {
        let i = side === 0 ? pa : pa - (L - 1);
        if (i < 0) i += n;
        let j = i + L - 1;
        if (j >= n) j -= n;
        const s1 = tour[i], s2 = tour[j];
        const p = tour[i === 0 ? n - 1 : i - 1];
        const nx = tour[j + 1 === n ? 0 : j + 1];
        const G = dist(p, s1) + dist(s2, nx) - dist(p, nx);
        if (G <= 0) continue;
        for (let end = 0; end < 2; end++) {
          const s = end === 0 ? s1 : s2;
          const o = end === 0 ? s2 : s1; // the segment's other end
          const sb = s * KK;
          for (let k = 0; k < KK; k++) {
            const dc = nbd[sb + k];
            if (dc >= G) break;
            const c = nb[sb + k];
            let r = pos[c] - i;
            if (r < 0) r += n;
            if (r < L) continue;
            // gap (c, succ c): c - s ... o - e
            const e = succ(c);
            let re = pos[e] - i;
            if (re < 0) re += n;
            if (re >= L) {
              const delta = dc + dist(o, e) - dist(c, e) - G;
              if (delta < 0) {
                moveSegment(i, L, pos[c], end === 1);
                push(p); push(nx); push(s1); push(s2); push(c); push(e);
                return delta;
              }
            }
            // gap (pred c, c): f - o ... s - c
            const f = pred(c);
            let rf = pos[f] - i;
            if (rf < 0) rf += n;
            if (rf >= L) {
              const delta = dc + dist(o, f) - dist(f, c) - G;
              if (delta < 0) {
                moveSegment(i, L, pos[f], end === 0);
                push(p); push(nx); push(s1); push(s2); push(c); push(f);
                return delta;
              }
            }
          }
        }
      }
    }
    return 0;
  };

  const localSearch = () => {
    let total = 0;
    while (qs > 0) {
      const a = queue[qh];
      if (++qh === n) qh = 0;
      qs--;
      inQ[a] = 0;
      total += improveCity(a);
    }
    return total;
  };

  const tourLength = () => {
    let s = 0;
    for (let k = 0; k < n; k++) s += dist(tour[k], tour[k + 1 === n ? 0 : k + 1]);
    return s;
  };

  for (let k = 0; k < n; k++) push(tour[k]);
  localSearch();

  // ---- 4. iterated local search ----
  let rs = (0x9e3779b9 ^ (n * 2654435761)) | 0;
  if (rs === 0) rs = 1;
  const rnd = () => {
    rs ^= rs << 13;
    rs ^= rs >>> 17;
    rs ^= rs << 5;
    return rs >>> 0;
  };
  const segMax = Math.max(1, Math.min(SEG_MAX, Math.floor(n / 8)));
  const kicks = KICKS_PER_CITY * n;
  let cur = tourLength();
  for (let it = 0; it < kicks; it++) {
    const p0 = rnd() % n;
    const l1 = 1 + (rnd() % segMax);
    const l2 = 1 + (rnd() % segMax);
    const i = p0 + 1 === n ? 0 : p0 + 1;
    const pb = (p0 + l1) % n;
    const pc = (pb + 1) % n;
    const k = (p0 + l1 + l2) % n;
    const kn = k + 1 === n ? 0 : k + 1;
    const aE = tour[p0], bS = tour[i], bE = tour[pb], cS = tour[pc], cE = tour[k], dS = tour[kn];
    let delta = dist(aE, cS) + dist(cE, bS) + dist(bE, dS) - dist(aE, bS) - dist(bE, cS) - dist(cE, dS);
    logging = true;
    overflow = false;
    nOps = 0;
    bufTop = 0;
    moveSegment(i, l1, k, false);
    push(aE); push(bS); push(bE); push(cS); push(cE); push(dS);
    delta += localSearch();
    if (delta <= 0 || overflow) cur += delta;
    else undoAll();
    logging = false;
  }

  return { tour, length: cur, check: tourLength() };
}

function validTour(t, n) {
  if (t.length !== n) return false;
  const seen = new Uint8Array(n);
  for (let k = 0; k < n; k++) {
    const c = t[k];
    if (c < 0 || c >= n || seen[c]) return false;
    seen[c] = 1;
  }
  return true;
}

for (const name of readdirSync(INPUT_DIR).filter((f) => f.endsWith('.txt')).sort()) {
  const { n, xs, ys } = readInstance(join(INPUT_DIR, name));
  let tour;
  if (n <= 8) {
    tour = Int32Array.from({ length: n }, (_, k) => k);
  } else {
    tour = solve(n, xs, ys).tour;
  }
  if (!validTour(tour, n)) tour = Int32Array.from({ length: n }, (_, k) => k);
  writeFileSync(join(OUTPUT_DIR, name.replace(/\.txt$/, '.tour')), `${Array.from(tour).join('\n')}\n`);
}

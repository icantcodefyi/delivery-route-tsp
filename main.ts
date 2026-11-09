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

const RESTARTS = [[12, 2], [10, 2], [14, 2]]; // runs: [nearest candidates, candidates per quadrant]
// The grader's core is about 6x slower than a laptop core: these budgets take about 13 s here, about 80 s there.
const RESTART_KICKS = 10; // kicks per city in each restart
const POLISH_KICKS = 32; // kicks per city polishing the best restart
const SMALL_BOOST = 1; // kick multiplier for sets of at most 2,000 cities
const UNIFORM_SINGLE = false; // uniform sets: one run with the whole budget instead of restarts (worse: off)
const SEG_MAX = 50; // longest segment a position-local kick moves
const SPATIAL_KICK = false; // kick around a random city and its near neighbours instead of a window of positions
const QUAD = 2; // candidates taken from each quadrant around a city
const LONG_M = 100; // the long-edge pass looks at this many longest edges
const LONG_ROUNDS = 40; // and applies at most this many moves per pass
const LONG_PASSES = 4; // long-edge passes spread over the kicks
const KICK_NB = 6; // the kick's two other cities come from this many nearest neighbours

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

// One full run: candidate lists with Kc nearest and Qc per quadrant, a start (greedy edge, or initTour), local
// search, long-edge passes and kicksPerCity * n kicks seeded by seed.
function solve(n, X, Y, Kc, Qc, kicksPerCity, seed, initTour) {
  const dist = (a, b) => {
    const dx = X[a] - X[b];
    const dy = Y[a] - Y[b];
    return Math.round(Math.sqrt(dx * dx + dy * dy));
  };

  // ---- 1. K nearest neighbours with a grid ----
  const KK = Math.min(Kc, n - 1);
  const KN = KK; // nearest cities gathered per city
  const QN = Qc; // nearest cities gathered per quadrant
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
    const bd = new Float64Array(KN + 1);
    const bi = new Int32Array(KN + 1);
    // per quadrant, the QN nearest cities, so that every direction (other clusters too) has a candidate
    const qd = new Float64Array(4 * QN);
    const qi = new Int32Array(4 * QN);
    const qc = new Int32Array(4);
    const cmin = Math.min(cw, ch);
    for (let i = 0; i < n; i++) {
      const cx = cellOf[i] % G;
      const cy = (cellOf[i] - cx) / G;
      let cnt = 0;
      qc[0] = qc[1] = qc[2] = qc[3] = 0;
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
                if (cnt < KN || d2 < bd[cnt - 1]) {
                  let p = cnt < KN ? cnt++ : cnt - 1;
                  while (p > 0 && bd[p - 1] > d2) {
                    bd[p] = bd[p - 1];
                    bi[p] = bi[p - 1];
                    p--;
                  }
                  bd[p] = d2;
                  bi[p] = j;
                }
                const q = (dx >= 0 ? 0 : 1) + (dy >= 0 ? 0 : 2);
                const qb = q * QN;
                const m = qc[q];
                if (m < QN || d2 < qd[qb + m - 1]) {
                  let p = m < QN ? qc[q]++ : m - 1;
                  while (p > 0 && qd[qb + p - 1] > d2) {
                    qd[qb + p] = qd[qb + p - 1];
                    qi[qb + p] = qi[qb + p - 1];
                    p--;
                  }
                  qd[qb + p] = d2;
                  qi[qb + p] = j;
                }
              }
            }
            if (x1 === x0) break;
          }
        }
        const reach = r * cmin;
        const r2 = reach * reach;
        if (cnt >= KN && bd[cnt - 1] <= r2) {
          let done = true;
          for (let q = 0; q < 4 && done; q++) if (qc[q] < QN || qd[q * QN + QN - 1] > r2) done = false;
          if (done) break;
        }
      }
      // the candidate list: the quadrant picks, then the nearest cities, KK distinct, sorted by distance
      let m = 0;
      const cand = [];
      for (let q = 0; q < 4; q++) for (let t = 0; t < qc[q]; t++) cand.push(qi[q * QN + t]);
      for (let t = 0; t < cnt; t++) cand.push(bi[t]);
      const pick = [];
      for (const c of cand) {
        if (m >= KK) break;
        if (pick.indexOf(c) >= 0) continue;
        pick.push(c);
        m++;
      }
      const pd = pick.map((c) => (X[c] - xi) * (X[c] - xi) + (Y[c] - yi) * (Y[c] - yi));
      const ord = pick.map((_, t) => t).sort((a, b) => pd[a] - pd[b] || pick[a] - pick[b]);
      for (let k = 0; k < KK; k++) {
        const c = pick[ord[k]];
        nb[i * KK + k] = c;
        nbd[i * KK + k] = dist(i, c);
      }
    }
  }

  // ---- 2. greedy-edge start ----
  const tour = new Int32Array(n);
  const pos = new Int32Array(n);
  if (initTour) {
    tour.set(initTour);
    for (let i = 0; i < n; i++) pos[tour[i]] = i;
  } else {
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
  // a reversal is its own inverse: log only its range (as a negative length), undo it by reversing again
  const logReversal = (start, len) => {
    if (!logging || overflow) return;
    if (nOps >= OPS_CAP) {
      overflow = true;
      return;
    }
    logStart[nOps] = start;
    logLen[nOps] = -len;
    nOps++;
  };
  const rawReverse = (i, len) => {
    let a = i, b = i + len - 1;
    if (b >= n) b -= n;
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
  const undoAll = () => {
    for (let o = nOps - 1; o >= 0; o--) {
      const len = logLen[o];
      let p = logStart[o];
      if (len < 0) {
        rawReverse(p, -len);
        continue;
      }
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
    logReversal(i, len);
    rawReverse(i, len);
  };

  // move the segment at positions i..i+L-1 into the gap after position k, reversed if rev
  const segTmp = new Int32Array(n + 8);
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

  // Long-edge pass: the longest tour edges are mostly the links between clusters. Try every 3-opt reconnection
  // of every triple of them (A B C -> A C B, A C B', A C' B, A B' C'), apply the best, repair locally, repeat.
  const el = new Float64Array(n);
  const keys = new Float64Array(n);
  const longEdgePass = () => {
    // big clustered sets have more links between clusters: look at 1.5% of the edges, at least LONG_M
    const M = Math.min(Math.max(LONG_M, Math.round(n * 0.015)), n - 1);
    const P = new Int32Array(M);
    for (let round = 0; round < LONG_ROUNDS; round++) {
      for (let p = 0; p < n; p++) {
        el[p] = dist(tour[p], tour[p + 1 === n ? 0 : p + 1]);
        keys[p] = -(el[p] * 16384 + p);
      }
      const sorted = keys.slice().sort();
      for (let t = 0; t < M; t++) P[t] = (-sorted[t]) % 16384;
      P.sort();
      let best = 0, bi = -1, bj = -1, bk = -1, btype = 0;
      for (let x = 0; x < M; x++) {
        const p1 = P[x];
        const a = tour[p1], b1 = tour[p1 + 1 === n ? 0 : p1 + 1];
        for (let y = x + 1; y < M; y++) {
          const p2 = P[y];
          const b2 = tour[p2], c1 = tour[p2 + 1 === n ? 0 : p2 + 1];
          const r12 = el[p1] + el[p2];
          const dac1 = dist(a, c1);
          const dab2 = dist(a, b2);
          // 2-opt between the two long edges: A B' C
          const g2 = dab2 + dist(b1, c1) - r12;
          if (g2 < best) { best = g2; bi = p1; bj = p2; bk = -1; btype = 6; }
          for (let z = y + 1; z < M; z++) {
            const p3 = P[z];
            const c2 = tour[p3], d = tour[p3 + 1 === n ? 0 : p3 + 1];
            const removed = r12 + el[p3];
            // A C B
            let g = dac1 + dist(c2, b1) + dist(b2, d) - removed;
            if (g < best) { best = g; bi = p1; bj = p2; bk = p3; btype = 1; }
            // A C B'
            g = dac1 + dist(c2, b2) + dist(b1, d) - removed;
            if (g < best) { best = g; bi = p1; bj = p2; bk = p3; btype = 2; }
            // A C' B
            g = dist(a, c2) + dist(c1, b1) + dist(b2, d) - removed;
            if (g < best) { best = g; bi = p1; bj = p2; bk = p3; btype = 3; }
            // A B' C'
            g = dab2 + dist(b1, c2) + dist(c1, d) - removed;
            if (g < best) { best = g; bi = p1; bj = p2; bk = p3; btype = 4; }
          }
        }
      }
      // Or-opt of a whole stretch between two long edges (often one visit of a cluster), into the gap next to a
      // candidate neighbour of one of its ends: this merges two visits of one cluster into one
      let mi = -1, ml = 0, mk = 0, mrev = false;
      let u0 = 0, u1 = 0, u2 = 0, u3 = 0, u4 = 0, u5 = 0;
      for (let x = 0; x < M; x++) {
        for (let y = x + 1; y < M; y++) {
          const p1 = P[x], p2 = P[y];
          for (let side = 0; side < 2; side++) {
            const i = side === 0 ? (p1 + 1) % n : (p2 + 1) % n;
            const len = side === 0 ? p2 - p1 : n - (p2 - p1);
            const j = side === 0 ? p2 : p1;
            if (len < 2 || len > n - 3) continue;
            const s1 = tour[i], s2 = tour[j];
            const o1 = tour[i === 0 ? n - 1 : i - 1], o2 = tour[j + 1 === n ? 0 : j + 1];
            const G = el[p1] + el[p2] - dist(o1, o2);
            if (G <= 0) continue;
            for (let end = 0; end < 2; end++) {
              const s = end === 0 ? s1 : s2;
              const o = end === 0 ? s2 : s1;
              const sb = s * KK;
              for (let k = 0; k < KK; k++) {
                const dc = nbd[sb + k];
                if (dc >= G) break;
                const c = nb[sb + k];
                let r = pos[c] - i;
                if (r < 0) r += n;
                if (r < len) continue;
                const e = succ(c);
                let re = pos[e] - i;
                if (re < 0) re += n;
                if (re >= len) {
                  const g = dc + dist(o, e) - dist(c, e) - G;
                  if (g < best) {
                    best = g; btype = 5; mi = i; ml = len; mk = pos[c]; mrev = end === 1;
                    u0 = s1; u1 = s2; u2 = o1; u3 = o2; u4 = c; u5 = e;
                  }
                }
                const f = pred(c);
                let rf = pos[f] - i;
                if (rf < 0) rf += n;
                if (rf >= len) {
                  const g = dc + dist(o, f) - dist(f, c) - G;
                  if (g < best) {
                    best = g; btype = 5; mi = i; ml = len; mk = pos[f]; mrev = end === 0;
                    u0 = s1; u1 = s2; u2 = o1; u3 = o2; u4 = c; u5 = f;
                  }
                }
              }
            }
          }
        }
      }
      if (btype === 0) return;
      if (btype === 6) {
        const a = tour[bi], b1 = tour[(bi + 1) % n], b2 = tour[bj], c1 = tour[(bj + 1) % n];
        reversePath((bi + 1) % n, bj);
        push(a); push(b1); push(b2); push(c1);
        localSearch();
        continue;
      }
      if (btype === 5) {
        moveSegment(mi, ml, mk, mrev);
        push(u0); push(u1); push(u2); push(u3); push(u4); push(u5);
        localSearch();
        continue;
      }
      const p1 = bi, p2 = bj, p3 = bk;
      const a = tour[p1], b1 = tour[(p1 + 1) % n], b2 = tour[p2], c1 = tour[(p2 + 1) % n], c2 = tour[p3], d = tour[(p3 + 1) % n];
      const lb = p2 - p1, lc = p3 - p2;
      if (btype === 1) {
        if (lb <= lc) moveSegment((p1 + 1) % n, lb, p3, false);
        else moveSegment((p2 + 1) % n, lc, p1, false);
      } else if (btype === 2) {
        moveSegment((p1 + 1) % n, lb, p3, true);
      } else if (btype === 3) {
        moveSegment((p2 + 1) % n, lc, p1, true);
      } else {
        reversePath((p1 + 1) % n, p2);
        let r = pos[c2] - pos[c1];
        if (r < 0) r += n;
        if (r === lc - 1) reversePath(pos[c1], pos[c2]);
        else reversePath(pos[c2], pos[c1]);
      }
      push(a); push(b1); push(b2); push(c1); push(c2); push(d);
      localSearch();
    }
  };
  longEdgePass();

  // ---- 4. iterated local search ----
  let rs = (0x9e3779b9 ^ (n * 2654435761) ^ Math.imul(seed + 1, 0x85ebca6b)) | 0;
  if (rs === 0) rs = 1;
  const rnd = () => {
    rs ^= rs << 13;
    rs ^= rs >>> 17;
    rs ^= rs << 5;
    return rs >>> 0;
  };
  const segMax = Math.max(1, Math.min(SEG_MAX, Math.floor(n / 8)));
  const dbTmp = new Int32Array(3 * segMax + 8);
  const kicks = Math.floor(kicksPerCity * n);
  let cur = tourLength();
  const longEvery = Math.max(1, Math.floor(kicks / LONG_PASSES));
  for (let it = 0; it < kicks; it++) {
    if (it > 0 && it % longEvery === 0) longEdgePass();
    // three cut positions p0 < pb < k: swap the segments B = p0+1..pb and C = pb+1..k (a double bridge)
    let p0, pb, k;
    if (SPATIAL_KICK) {
      // cut right after a random city and two of its near neighbours, so the three new edges are short
      const v = rnd() % n;
      const w1 = nb[v * KK + (rnd() % KICK_NB)];
      const w2 = nb[v * KK + (rnd() % KICK_NB)];
      const q0 = pos[v], q1 = pos[w1], q2 = pos[w2];
      if (q0 === q1 || q0 === q2 || q1 === q2) continue;
      p0 = Math.min(q0, q1, q2);
      k = Math.max(q0, q1, q2);
      pb = q0 + q1 + q2 - p0 - k;
      if (p0 === 0 && k === n - 1) continue;
    } else {
      // a true double bridge on three short consecutive segments: A B C D -> A D C B (four edges change, and no
      // sequential 2-opt/3-opt move undoes it)
      p0 = rnd() % n;
      const l1 = 1 + (rnd() % segMax);
      const l2 = 1 + (rnd() % segMax);
      const l3 = 1 + (rnd() % segMax);
      const pB = (p0 + l1) % n, pC = (p0 + l1 + l2) % n, pD = (p0 + l1 + l2 + l3) % n;
      const a = tour[p0], b1 = tour[(p0 + 1) % n], b2 = tour[pB], c1 = tour[(pB + 1) % n];
      const c2 = tour[pC], d1 = tour[(pC + 1) % n], d2 = tour[pD], e = tour[(pD + 1) % n];
      const dlt = dist(a, d1) + dist(d2, c1) + dist(c2, b1) + dist(b2, e)
        - dist(a, b1) - dist(b2, c1) - dist(c2, d1) - dist(d2, e);
      logging = true;
      overflow = false;
      nOps = 0;
      bufTop = 0;
      const start = (p0 + 1) % n;
      const W = l1 + l2 + l3;
      saveRange(start, W);
      let q = start;
      for (let t = 0; t < W; t++) {
        dbTmp[t] = tour[q];
        if (++q === n) q = 0;
      }
      let p = start;
      for (let t = l1 + l2; t < W; t++) { const c = dbTmp[t]; tour[p] = c; pos[c] = p; if (++p === n) p = 0; }
      for (let t = l1; t < l1 + l2; t++) { const c = dbTmp[t]; tour[p] = c; pos[c] = p; if (++p === n) p = 0; }
      for (let t = 0; t < l1; t++) { const c = dbTmp[t]; tour[p] = c; pos[c] = p; if (++p === n) p = 0; }
      push(a); push(b1); push(b2); push(c1); push(c2); push(d1); push(d2); push(e);
      const total = dlt + localSearch();
      if (total <= 0 || overflow) cur += total;
      else undoAll();
      logging = false;
      continue;
    }
    const i = (p0 + 1) % n;
    pb %= n;
    k %= n;
    const pc = (pb + 1) % n;
    const kn = k + 1 === n ? 0 : k + 1;
    const aE = tour[p0], bS = tour[i], bE = tour[pb], cS = tour[pc], cE = tour[k], dS = tour[kn];
    let delta = dist(aE, cS) + dist(cE, bS) + dist(bE, dS) - dist(aE, bS) - dist(bE, cS) - dist(cE, dS);
    logging = true;
    overflow = false;
    nOps = 0;
    bufTop = 0;
    let lb = pb - p0;
    if (lb < 0) lb += n;
    let lc = k - pb;
    if (lc < 0) lc += n;
    // move the shorter of the two segments
    if (lb <= lc) moveSegment(i, lb, k, false);
    else moveSegment(pc, lc, p0, false);
    push(aE); push(bS); push(bE); push(cS); push(cE); push(dS);
    delta += localSearch();
    if (delta <= 0 || overflow) cur += delta;
    else undoAll();
    logging = false;
  }

  longEdgePass();
  return { tour, length: cur, check: tourLength() };
}

// Clustered or uniform, from the data: with about two cities per grid cell, a uniform set leaves about e^-2 = 14%
// of the cells empty, a clustered one most of them.
function isClustered(n, X, Y) {
  const G = Math.max(1, Math.floor(Math.sqrt(n / 2)));
  const seen = new Uint8Array(G * G);
  for (let i = 0; i < n; i++) {
    let cx = Math.floor((X[i] * G) / 1000001);
    let cy = Math.floor((Y[i] * G) / 1000001);
    if (cx >= G) cx = G - 1;
    if (cy >= G) cy = G - 1;
    seen[cy * G + cx] = 1;
  }
  let empty = 0;
  for (let c = 0; c < G * G; c++) if (!seen[c]) empty++;
  return empty > 0.4 * G * G;
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
  } else if (!UNIFORM_SINGLE || isClustered(n, xs, ys)) {
    // restarts with different candidate lists (clustered sets are chaotic: the cluster order a run settles on
    // varies), keep the shortest, then polish it with more kicks
    let best = null;
    let bestCfg = 0;
    // small sets are cheap: give them more kicks per city
    const boost = n <= 2000 ? SMALL_BOOST : 1;
    const runs = RESTARTS;
    for (let r = 0; r < runs.length; r++) {
      const res = solve(n, xs, ys, runs[r][0], runs[r][1], RESTART_KICKS * boost, r, null);
      if (best === null || res.check < best.check) {
        best = res;
        bestCfg = r;
      }
    }
    const fin = solve(n, xs, ys, runs[bestCfg][0], runs[bestCfg][1], POLISH_KICKS * boost, 99, best.tour);
    tour = fin.check <= best.check ? fin.tour : best.tour;
  } else {
    // a uniform set has no cluster order to get wrong: one run gets the whole kick budget
    const total = RESTARTS.length * RESTART_KICKS + POLISH_KICKS;
    tour = solve(n, xs, ys, RESTARTS[0][0], RESTARTS[0][1], total, 0, null).tour;
  }
  if (!validTour(tour, n)) tour = Int32Array.from({ length: n }, (_, k) => k);
  writeFileSync(join(OUTPUT_DIR, name.replace(/\.txt$/, '.tour')), `${Array.from(tour).join('\n')}\n`);
}

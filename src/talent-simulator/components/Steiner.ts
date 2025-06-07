/* --------------------------------------------------------------------- */
/*  Types and helper functions                                            */
/* --------------------------------------------------------------------- */

export interface NodeType {
	id: number;
	points: number;
	linkedNodesIndexes: number[];
}

export interface SteinerResult {
	steinerCost: number;
	steinerNodes: Set<number>;
	steinerEdges: [number, number][];
}

interface Arc {
	to: number;
	w: number;
}

function splitIn(v: number): number {
	return v << 1;
}

function splitOut(v: number): number {
	return (v << 1) | 1;
}

function origId(splitId: number): number {
	return splitId >> 1;
}

function buildSplitGraph(nodes: NodeType[]): Arc[][] {
	const N = nodes.length * 2;
	const g: Arc[][] = Array(N);
	for (let i = 0; i < N; i++) g[i] = [];

	for (const v of nodes) {
		const inId = splitIn(v.id);
		const outId = splitOut(v.id);

		/* pay cost once when entering v_out */
		g[inId].push({ to: outId, w: v.points });

		/* zero-cost traverse to neighbours (out → nb-in) */
		for (const nb of v.linkedNodesIndexes) {
			g[outId].push({ to: splitIn(nb), w: 0 });
		}
	}
	return g;
}

class MinHeap {
	private key: number[] = [];
	private val: number[] = [];
	private pos: Int32Array;

	constructor(n: number) {
		this.pos = new Int32Array(n).fill(-1);
	}

	private swap(i: number, j: number) {
		[this.key[i], this.key[j]] = [this.key[j], this.key[i]];
		[this.val[i], this.val[j]] = [this.val[j], this.val[i]];
		this.pos[this.key[i]] = i;
		this.pos[this.key[j]] = j;
	}

	private up(i: number) {
		while (i && this.val[i] < this.val[(i - 1) >> 1]) {
			const p = (i - 1) >> 1;
			this.swap(i, p);
			i = p;
		}
	}

	private down(i: number) {
		const n = this.key.length;
		while (true) {
			let l = (i << 1) + 1;
			if (l >= n) break;
			const r = l + 1,
				s = r < n && this.val[r] < this.val[l] ? r : l;
			if (this.val[i] <= this.val[s]) break;
			this.swap(i, s);
			i = s;
		}
	}

	pushOrDecrease(k: number, v: number) {
		const i = this.pos[k];
		if (i === -1) {
			const idx = this.key.length;
			this.key.push(k);
			this.val.push(v);
			this.pos[k] = idx;
			this.up(idx);
		} else if (v < this.val[i]) {
			this.val[i] = v;
			this.up(i);
		}
	}

	pop(): number | null {
		if (!this.key.length) return null;
		const k = this.key[0];
		const lk = this.key.pop()!,
			lv = this.val.pop()!;
		if (this.key.length) {
			this.key[0] = lk;
			this.val[0] = lv;
			this.pos[lk] = 0;
			this.down(0);
		}
		this.pos[k] = -1;
		return k;
	}
}

function dijkstra(
	g: Arc[][],
	src: number,
): { dist: Float64Array; parent: Int32Array } {
	const N = g.length,
		INF = 1e15;
	const dist = new Float64Array(N);
	dist.fill(INF);
	dist[src] = 0;
	const parent = new Int32Array(N);
	parent.fill(-1);
	const pq = new MinHeap(N);
	pq.pushOrDecrease(src, 0);

	while (true) {
		const u = pq.pop();
		if (u === null) break;
		const du = dist[u];
		for (const e of g[u]) {
			const alt = du + e.w;
			if (alt < dist[e.to]) {
				dist[e.to] = alt;
				parent[e.to] = u;
				pq.pushOrDecrease(e.to, alt);
			}
		}
	}
	return { dist, parent };
}

/* --------------------------------------------------------------------- */
/*  Exact Steiner (Dreyfus–Wagner) with rock-solid reconstruction        */
/* --------------------------------------------------------------------- */
export function exactSteiner(
	nodes: NodeType[],
	keyIds: number[], // list of terminal node IDs
): SteinerResult {
	const g = buildSplitGraph(nodes);
	const N = g.length; // ≤ 600
	const K = keyIds.length; // ≤ 20
	if (!K) {
		return {
			steinerCost: 0,
			steinerNodes: new Set<number>(),
			steinerEdges: [],
		};
	}
	const FULL = (1 << K) - 1;

	/* 1.  terminal split-IDs and one Dijkstra per terminal */
	const termSplit: number[] = [];
	const distT: Float64Array[] = [];
	const parT: Int16Array[] = []; // parent per terminal

	for (let i = 0; i < K; i++) {
		const src = splitOut(keyIds[i]);
		termSplit[i] = src;
		const { dist, parent } = dijkstra(g, src);
		distT[i] = dist;
		const p16 = new Int16Array(parent.length);
		for (let j = 0; j < parent.length; j++) p16[j] = parent[j];
		parT[i] = p16;
	}

	/* 2.  DP tables */
	const INF = 1e15;
	const dp: Float64Array[] = Array(1 << K);
	const pred: Int16Array[] = Array(1 << K); // predecessor vertex
	const split: Int32Array[] = Array(1 << K); // split mask  A ⊂ S

	for (let S = 0; S <= FULL; S++) {
		dp[S] = new Float64Array(N);
		pred[S] = new Int16Array(N);
		split[S] = new Int32Array(N);
		for (let v = 0; v < N; v++) {
			dp[S][v] = INF;
			pred[S][v] = -1;
			split[S][v] = 0;
		}
	}

	/* 2a.  singletons */
	for (let i = 0; i < K; i++) {
		const m = 1 << i;
		const d = distT[i];
		for (let v = 0; v < N; v++) dp[m][v] = d[v];
		split[m].fill(-1); // mark "terminal source"
	}

	/* 3.  iterate over all non-empty subsets */
	for (let S = 1; S <= FULL; S++) {
		if ((S & (S - 1)) === 0) continue; // singleton already done

		/* 3a.  subset merge */
		const low = S & -S; // ensure A ≤ S\A only once
		for (let A = (S - 1) & S; A; A = (A - 1) & S) {
			if (!(A & low)) continue;
			const B = S ^ A;
			const dpA = dp[A],
				dpB = dp[B],
				dpS = dp[S];
			const prS = pred[S];

			for (let v = 0; v < N; v++) {
				const cand = dpA[v] + dpB[v];
				if (cand < dpS[v]) {
					dpS[v] = cand;
					split[S][v] = A; // remember which subset A was used
					prS[v] = -1; // no predecessor edge at this vertex
				}
			}
		}

		/* 3b.  multi-source Dijkstra relaxation */
		const pq = new MinHeap(N);
		for (let v = 0; v < N; v++)
			if (dp[S][v] < INF) pq.pushOrDecrease(v, dp[S][v]);

		while (true) {
			const u = pq.pop();
			if (u === null) break;
			const du = dp[S][u];
			for (let k = 0; k < g[u].length; k++) {
				const e = g[u][k];
				const alt = du + e.w;
				if (alt < dp[S][e.to]) {
					dp[S][e.to] = alt;
					pred[S][e.to] = u; // predecessor via edge
					split[S][e.to] = 0; // 0  = came from an edge, not a split
					pq.pushOrDecrease(e.to, alt);
				}
			}
		}
	}

	/* 4.  best root */
	let root = 0,
		bestCost = dp[FULL][0];
	for (let v = 1; v < N; v++)
		if (dp[FULL][v] < bestCost) {
			bestCost = dp[FULL][v];
			root = v;
		}

	/* 5.  reconstruction ---------------------------------------------------- */

	const nodesUsed = new Set<number>();
	const edgeSet = new Set<string>();
	const edgeList: [number, number][] = [];

	function addEdge(u0: number, v0: number) {
		if (u0 === v0) return;
		const a = u0 < v0 ? u0 : v0;
		const b = u0 < v0 ? v0 : u0;
		const key = a + "-" + b;
		if (!edgeSet.has(key)) {
			edgeSet.add(key);
			edgeList.push([a, b]);
		}
	}

	function tracePathFromTerminal(termIdx: number, v: number) {
		let cur = v;
		const par = parT[termIdx];
		while (cur !== termSplit[termIdx]) {
			const p = par[cur];
			if (p === -1) break; // safety
			/* only out→in arcs correspond to original edges */
			if ((p & 1) === 1 && (cur & 1) === 0) addEdge(origId(p), origId(cur));
			nodesUsed.add(origId(cur));
			cur = p;
		}
		nodesUsed.add(keyIds[termIdx]);
	}

	function rebuild(S: number, v: number) {
		if (split[S][v] === -1) {
			// singleton
			/* which terminal is it? */
			for (let i = 0; i < K; i++)
				if (S & (1 << i)) {
					tracePathFromTerminal(i, v);
					break;
				}
			return;
		}

		if (split[S][v] > 0) {
			// merged A + B at same vertex
			const A = split[S][v];
			const B = S ^ A;
			rebuild(A, v);
			rebuild(B, v);
			return;
		}

		/* came via predecessor edge */
		const u = pred[S][v];
		if (u === -1) throw new Error("Trace broken");
		rebuild(S, u);
		if ((u & 1) === 1 && (v & 1) === 0) addEdge(origId(u), origId(v));
		nodesUsed.add(origId(v));
	}

	rebuild(FULL, root);

	return {
		steinerCost: bestCost,
		steinerNodes: nodesUsed,
		steinerEdges: edgeList,
	};
}

<script lang="ts">
	/**
	 * The centerpiece: the RLRP -> PATT -> SIM cycle, showing where the
	 * algorithm currently is.
	 *
	 * Layout is a triangle so both feedback edges (SIM->PATT and PATT->RLRP)
	 * can be drawn as returning arcs without crossing the forward path.
	 */
	import type { EdgeDef, Progress, RoundState, StageDef, StageKey } from '$lib/types';

	let {
		progress,
		round,
		selectedEdge = $bindable(null)
	}: {
		progress: Progress;
		round: RoundState | null;
		selectedEdge?: string | null;
	} = $props();

	// Wide aspect on purpose: the whole cycle has to fit on screen in one go,
	// because this is the figure that ends up in the report.
	const W = 900;
	const H = 380;

	// triangle: RLRP top-left, PATT top-right, SIM bottom-centre
	const POS: Record<StageKey, { x: number; y: number }> = {
		rlrp: { x: 172, y: 104 },
		patt: { x: 728, y: 104 },
		sim: { x: 450, y: 284 }
	};
	const NODE_W = 214;
	const NODE_H = 100;

	const COLOR: Record<StageKey, string> = {
		rlrp: 'var(--rlrp)',
		patt: 'var(--patt)',
		sim: 'var(--sim)'
	};

	const isLive = $derived(progress.status === 'running');
	const activeStage = $derived(isLive ? progress.current_stage : null);
	const activeEdge = $derived(round?.outcome?.edge_id ?? (isLive ? progress.current_edge : null));

	function stageState(key: StageKey) {
		return round?.stages?.[key];
	}

	function stageStatus(key: StageKey): string {
		if (activeStage === key) return 'running';
		return stageState(key)?.status ?? 'pending';
	}

	/** Forward edges are "done" once their source stage finished in this round. */
	function edgeTraversed(edge: EdgeDef): boolean {
		if (edge.kind === 'feedback') return activeEdge === edge.id;
		const source = stageStatus(edge.source);
		return source === 'completed' || source === 'reused';
	}

	function nodeOpacity(key: StageKey): number {
		const status = stageStatus(key);
		if (status === 'pending') return 0.4;
		if (status === 'blocked') return 0.72;
		return 1;
	}

	/** Straight forward edge between two node borders. */
	function forwardPath(edge: EdgeDef): string {
		const a = POS[edge.source];
		const b = POS[edge.target];
		const [ax, ay] = borderPoint(a, b);
		const [bx, by] = borderPoint(b, a);
		return `M ${ax} ${ay} L ${bx} ${by}`;
	}

	/** Feedback edges bow outward so they read as a return path. */
	function feedbackPath(edge: EdgeDef): string {
		const a = POS[edge.source];
		const b = POS[edge.target];
		const [ax, ay] = borderPoint(a, b, 0.62);
		const [bx, by] = borderPoint(b, a, 0.62);
		const mx = (ax + bx) / 2;
		const my = (ay + by) / 2;
		// push the control point away from the triangle centre
		const cx = (POS.rlrp.x + POS.patt.x + POS.sim.x) / 3;
		const cy = (POS.rlrp.y + POS.patt.y + POS.sim.y) / 3;
		const dx = mx - cx;
		const dy = my - cy;
		const len = Math.hypot(dx, dy) || 1;
		const bow = 78;
		return `M ${ax} ${ay} Q ${mx + (dx / len) * bow} ${my + (dy / len) * bow} ${bx} ${by}`;
	}

	/** Where a line from `from` towards `to` leaves the node's box. */
	function borderPoint(
		from: { x: number; y: number },
		to: { x: number; y: number },
		scale = 1
	): [number, number] {
		const dx = to.x - from.x;
		const dy = to.y - from.y;
		const hw = (NODE_W / 2 + 12) * scale;
		const hh = (NODE_H / 2 + 12) * scale;
		if (dx === 0 && dy === 0) return [from.x, from.y];
		const tx = dx === 0 ? Infinity : hw / Math.abs(dx);
		const ty = dy === 0 ? Infinity : hh / Math.abs(dy);
		const t = Math.min(tx, ty);
		return [from.x + dx * t, from.y + dy * t];
	}

	/** Labels sit clear of their own line: handoffs are pushed perpendicular
	 *  towards the inside of the triangle, feedback labels sit past the apex of
	 *  their bow. Without this the two labels near SIM overlap. */
	function labelPoint(edge: EdgeDef): { x: number; y: number } {
		const a = POS[edge.source];
		const b = POS[edge.target];
		const cx = (POS.rlrp.x + POS.patt.x + POS.sim.x) / 3;
		const cy = (POS.rlrp.y + POS.patt.y + POS.sim.y) / 3;
		const mx = (a.x + b.x) / 2;
		const my = (a.y + b.y) / 2;

		if (edge.kind === 'handoff') {
			// perpendicular to the edge, on the side facing the triangle centre
			const dx = b.x - a.x;
			const dy = b.y - a.y;
			const len = Math.hypot(dx, dy) || 1;
			let px = -dy / len;
			let py = dx / len;
			if ((mx + px - cx) ** 2 + (my + py - cy) ** 2 > (mx - cx) ** 2 + (my - cy) ** 2) {
				px = -px;
				py = -py;
			}
			return { x: mx + px * 26, y: my + py * 26 - 3 };
		}

		const dx = mx - cx;
		const dy = my - cy;
		const len = Math.hypot(dx, dy) || 1;
		return { x: mx + (dx / len) * 96, y: my + (dy / len) * 96 };
	}

	function shortLabel(edge: EdgeDef): string {
		if (edge.id === 'rlrp->patt') return 'assignment + capacity';
		if (edge.id === 'patt->sim') return 'patterns + routes';
		if (edge.id === 'patt->rlrp') return 'capacity shortfall → demand ↑';
		return 'KPI miss → λ ↓';
	}

	function toggleEdge(id: string) {
		selectedEdge = selectedEdge === id ? null : id;
	}

	const stageDefs = $derived(
		progress.stages.reduce<Record<string, StageDef>>((acc, s) => ((acc[s.key] = s), acc), {})
	);
</script>

<div class="diagram-wrap">
	<svg viewBox="0 0 {W} {H}" role="img" aria-label="RLRP, PATT and SIM pipeline loop">
		<defs>
			{#each ['rlrp', 'patt', 'sim', 'feedback', 'idle'] as tone}
				<marker
					id="arrow-{tone}"
					viewBox="0 0 10 10"
					refX="9"
					refY="5"
					markerWidth="7"
					markerHeight="7"
					orient="auto-start-reverse"
				>
					<path
						d="M 0 0 L 10 5 L 0 10 z"
						fill={tone === 'idle'
							? 'var(--border-strong)'
							: tone === 'feedback'
								? 'var(--feedback)'
								: `var(--${tone})`}
					/>
				</marker>
			{/each}
		</defs>

		<!-- edges -->
		{#each progress.edges as edge (edge.id)}
			{@const traversed = edgeTraversed(edge)}
			{@const active = activeEdge === edge.id}
			{@const tone = edge.kind === 'feedback' ? 'feedback' : traversed ? edge.source : 'idle'}
			{@const path = edge.kind === 'feedback' ? feedbackPath(edge) : forwardPath(edge)}
			{@const point = labelPoint(edge)}
			<g
				class="edge"
				class:selected={selectedEdge === edge.id}
				onclick={() => toggleEdge(edge.id)}
				onkeydown={(e) => e.key === 'Enter' && toggleEdge(edge.id)}
				role="button"
				tabindex="0"
			>
				<path d={path} class="edge-hit" />
				<path
					d={path}
					class="edge-line"
					class:active
					class:dim={!traversed && !active}
					stroke={tone === 'idle'
						? 'var(--border-strong)'
						: tone === 'feedback'
							? 'var(--feedback)'
							: `var(--${tone})`}
					stroke-dasharray={edge.kind === 'feedback' ? '7 5' : undefined}
					marker-end="url(#arrow-{tone})"
				/>
				<text
					x={point.x}
					y={point.y}
					class="edge-label"
					class:dim={!traversed && !active}
					fill={edge.kind === 'feedback' ? 'var(--feedback)' : 'var(--text-muted)'}
				>
					{shortLabel(edge)}
				</text>
			</g>
		{/each}

		<!-- nodes -->
		{#each progress.stages as stage (stage.key)}
			{@const p = POS[stage.key]}
			{@const status = stageStatus(stage.key)}
			{@const state = stageState(stage.key)}
			<g class="node" opacity={nodeOpacity(stage.key)}>
				{#if status === 'running'}
					<rect
						x={p.x - NODE_W / 2 - 5}
						y={p.y - NODE_H / 2 - 5}
						width={NODE_W + 10}
						height={NODE_H + 10}
						rx="14"
						fill="none"
						stroke={COLOR[stage.key]}
						stroke-width="2"
						class="halo"
					/>
				{/if}
				<rect
					x={p.x - NODE_W / 2}
					y={p.y - NODE_H / 2}
					width={NODE_W}
					height={NODE_H}
					rx="11"
					fill="var(--surface)"
					stroke={status === 'pending' ? 'var(--border)' : COLOR[stage.key]}
					stroke-width={status === 'running' ? 2.5 : 1.5}
				/>
				<rect
					x={p.x - NODE_W / 2}
					y={p.y - NODE_H / 2}
					width="5"
					height={NODE_H}
					rx="2.5"
					fill={COLOR[stage.key]}
				/>
				<text x={p.x - NODE_W / 2 + 18} y={p.y - NODE_H / 2 + 25} class="node-title">
					{stage.title}
				</text>
				<text x={p.x - NODE_W / 2 + 18} y={p.y - NODE_H / 2 + 42} class="node-sub">
					{stage.subtitle}
				</text>
				<foreignObject
					x={p.x - NODE_W / 2 + 16}
					y={p.y - NODE_H / 2 + 50}
					width={NODE_W - 32}
					height={NODE_H - 58}
				>
					<div class="node-body">
						{#if state?.headline}
							<span class="headline">{state.headline}</span>
						{:else}
							<span class="decides">{stage.decides}</span>
						{/if}
					</div>
				</foreignObject>
				<g transform="translate({p.x + NODE_W / 2 - 12}, {p.y - NODE_H / 2 + 18})">
					<circle
						r="5"
						fill={status === 'completed'
							? 'var(--ok)'
							: status === 'running'
								? 'var(--warn)'
								: status === 'blocked'
									? 'var(--bad)'
									: status === 'reused'
										? 'var(--text-faint)'
										: 'var(--border-strong)'}
						class:pulse={status === 'running'}
					/>
				</g>
			</g>
		{/each}
	</svg>

	{#if selectedEdge}
		{@const edge = progress.edges.find((e) => e.id === selectedEdge)}
		{#if edge}
			<div class="edge-detail">
				<div class="row">
					<strong>{stageDefs[edge.source]?.title} → {stageDefs[edge.target]?.title}</strong>
					<span class="pill" class:feedback={edge.kind === 'feedback'}>{edge.kind}</span>
					<span class="spacer"></span>
					<button onclick={() => (selectedEdge = null)}>Close</button>
				</div>
				<p class="muted" style="margin:6px 0 0">{edge.label}</p>
				<p class="faint" style="margin:2px 0 0; font-size:0.8rem">{edge.detail}</p>
			</div>
		{/if}
	{/if}
</div>

<style>
	.diagram-wrap {
		position: relative;
	}

	svg {
		width: 100%;
		max-width: 1080px;
		height: auto;
		display: block;
		margin: 0 auto;
	}

	.node-title {
		font-size: 15px;
		font-weight: 700;
		fill: var(--text);
	}

	.node-sub {
		font-size: 10.5px;
		fill: var(--text-faint);
		text-transform: uppercase;
		letter-spacing: 0.06em;
	}

	.node-body {
		font-size: 11.5px;
		line-height: 1.35;
		color: var(--text-muted);
		height: 100%;
		display: flex;
		align-items: center;
	}

	.node-body .headline {
		color: var(--text);
		font-weight: 600;
	}

	.edge-line {
		fill: none;
		stroke-width: 2;
		transition: stroke 0.25s ease;
	}

	.edge-line.dim {
		opacity: 0.35;
	}

	.edge-line.active {
		stroke-width: 3;
		stroke-dasharray: 8 6;
		animation: flow 0.9s linear infinite;
	}

	@keyframes flow {
		to {
			stroke-dashoffset: -28;
		}
	}

	.edge-hit {
		fill: none;
		stroke: transparent;
		stroke-width: 22;
		cursor: pointer;
	}

	.edge-label {
		font-size: 10.5px;
		text-anchor: middle;
		pointer-events: none;
		font-weight: 550;
	}

	.edge-label.dim {
		opacity: 0.45;
	}

	.edge.selected .edge-line {
		stroke-width: 3.5;
	}

	.edge:focus {
		outline: none;
	}

	.halo {
		animation: halo 1.6s ease-in-out infinite;
	}

	@keyframes halo {
		0%,
		100% {
			opacity: 0.55;
		}
		50% {
			opacity: 0.12;
		}
	}

	.pulse {
		animation: halo 1.1s ease-in-out infinite;
	}

	.edge-detail {
		margin-top: 4px;
		border: 1px solid var(--border);
		border-radius: 8px;
		padding: 10px 13px;
		background: var(--surface-2);
	}

	@media (prefers-reduced-motion: reduce) {
		.edge-line.active,
		.halo,
		.pulse {
			animation: none;
		}
	}
</style>

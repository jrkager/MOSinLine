<script lang="ts">
	/** Plain XY network map. Our instances use Solomon Euclidean coordinates, not
	 *  lat/lon, so this is an equal-scale scatter plot rather than a geographic
	 *  projection.
	 *
	 *  Draws, in back-to-front order: assignment spokes, routes, then nodes.
	 *  Depot squares scale with warehouse size and store dots with demand, so the
	 *  model's decision is legible without reading the tables. */
	type Node = {
		id: number;
		kind: 'depot' | 'store';
		x: number;
		y: number;
		label?: string;
		/** relative magnitude (demand for stores, size for depots) */
		value?: number;
		open?: boolean;
	};
	type Route = { coords: [number, number][]; color?: string; label?: string; dashed?: boolean };
	type Spoke = { from: [number, number]; to: [number, number]; color?: string };
	type LegendItem = { label: string; color?: string; shape?: 'square' | 'dot' | 'line' | 'dash' };

	let {
		nodes,
		routes = [],
		spokes = [],
		highlight = new Set<number>(),
		colorBy = null,
		legend = [],
		height = 320,
		showLabels = true,
		arrows = false
	}: {
		nodes: Node[];
		routes?: Route[];
		spokes?: Spoke[];
		highlight?: Set<number>;
		colorBy?: Map<number, string> | null;
		legend?: LegendItem[];
		height?: number;
		showLabels?: boolean;
		arrows?: boolean;
	} = $props();

	const W = 640;
	const PAD = 30;

	const pts = $derived([
		...nodes.map((n) => [n.x, n.y] as [number, number]),
		...routes.flatMap((r) => r.coords)
	]);
	const xMin = $derived(pts.length ? Math.min(...pts.map((p) => p[0])) : 0);
	const xMax = $derived(pts.length ? Math.max(...pts.map((p) => p[0])) : 1);
	const yMin = $derived(pts.length ? Math.min(...pts.map((p) => p[1])) : 0);
	const yMax = $derived(pts.length ? Math.max(...pts.map((p) => p[1])) : 1);

	// equal scale on both axes so distances stay visually truthful
	const span = $derived(Math.max(xMax - xMin, yMax - yMin) || 1);
	const scale = $derived(Math.min((W - 2 * PAD) / span, (height - 2 * PAD) / span));
	const offX = $derived(PAD + (W - 2 * PAD - (xMax - xMin) * scale) / 2);
	const offY = $derived(PAD + (height - 2 * PAD - (yMax - yMin) * scale) / 2);

	const sx = $derived((x: number) => offX + (x - xMin) * scale);
	// flip y so the plot reads like a map rather than screen coordinates
	const sy = $derived((y: number) => height - (offY + (y - yMin) * scale));

	const storeMax = $derived(
		Math.max(1e-9, ...nodes.filter((n) => n.kind === 'store').map((n) => n.value ?? 0))
	);
	const depotMax = $derived(
		Math.max(1e-9, ...nodes.filter((n) => n.kind === 'depot').map((n) => n.value ?? 0))
	);

	function storeR(n: Node): number {
		if (n.value === undefined) return 4.5;
		return 3.2 + 5.2 * Math.sqrt(Math.max(0, n.value) / storeMax);
	}

	function depotHalf(n: Node): number {
		if (n.value === undefined) return 6;
		return 5 + 5 * Math.sqrt(Math.max(0, n.value) / depotMax);
	}

	function routePath(coords: [number, number][]): string {
		return coords.map((c, i) => `${i ? 'L' : 'M'} ${sx(c[0])} ${sy(c[1])}`).join(' ');
	}

	const depots = $derived(nodes.filter((n) => n.kind === 'depot'));
	const stores = $derived(nodes.filter((n) => n.kind === 'store'));
</script>

<svg viewBox="0 0 {W} {height}" role="img" aria-label="network map">
	<defs>
		<marker
			id="map-arrow"
			viewBox="0 0 10 10"
			refX="8"
			refY="5"
			markerWidth="5"
			markerHeight="5"
			orient="auto-start-reverse"
		>
			<path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor" />
		</marker>
	</defs>

	{#each spokes as spoke, i (i)}
		<line
			x1={sx(spoke.from[0])}
			y1={sy(spoke.from[1])}
			x2={sx(spoke.to[0])}
			y2={sy(spoke.to[1])}
			stroke={spoke.color ?? 'var(--border-strong)'}
			stroke-width="1"
			opacity="0.42"
		/>
	{/each}

	{#each routes as route, i (i)}
		<path
			d={routePath(route.coords)}
			fill="none"
			stroke={route.color ?? 'var(--patt)'}
			stroke-width="2"
			stroke-linejoin="round"
			stroke-dasharray={route.dashed ? '6 4' : undefined}
			opacity="0.9"
			style="color: {route.color ?? 'var(--patt)'}"
			marker-end={arrows ? 'url(#map-arrow)' : undefined}
		/>
	{/each}

	{#each stores as node (node.id)}
		<circle
			cx={sx(node.x)}
			cy={sy(node.y)}
			r={storeR(node)}
			fill={colorBy?.get(node.id) ?? 'var(--text-muted)'}
			stroke="var(--surface)"
			stroke-width="1.3"
		/>
		{#if showLabels}
			<text x={sx(node.x) + storeR(node) + 3} y={sy(node.y) + 3.5} class="lbl">
				{node.label ?? node.id}
			</text>
		{/if}
	{/each}

	{#each depots as node (node.id)}
		{@const h = depotHalf(node)}
		{@const isOpen = node.open ?? highlight.has(node.id)}
		<rect
			x={sx(node.x) - h}
			y={sy(node.y) - h}
			width={h * 2}
			height={h * 2}
			rx="2.5"
			fill={isOpen ? 'var(--rlrp)' : 'var(--surface)'}
			stroke="var(--rlrp)"
			stroke-width="2"
			stroke-dasharray={isOpen ? undefined : '3 2'}
			opacity={isOpen ? 1 : 0.55}
		/>
		{#if showLabels}
			<text x={sx(node.x)} y={sy(node.y) - h - 5} class="lbl depot" class:closed={!isOpen}>
				{node.label ?? `D${Math.abs(node.id)}`}
			</text>
		{/if}
	{/each}
</svg>

{#if legend.length}
	<div class="legend">
		{#each legend as item (item.label)}
			<span class="item">
				{#if item.shape === 'square'}
					<span class="sw square" style="border-color:{item.color ?? 'var(--rlrp)'}"></span>
				{:else if item.shape === 'line'}
					<span class="sw line" style="background:{item.color ?? 'var(--patt)'}"></span>
				{:else if item.shape === 'dash'}
					<span class="sw dash" style="border-color:{item.color ?? 'var(--border-strong)'}"></span>
				{:else}
					<span class="sw dot" style="background:{item.color ?? 'var(--text-muted)'}"></span>
				{/if}
				{item.label}
			</span>
		{/each}
	</div>
{/if}

<style>
	svg {
		width: 100%;
		height: auto;
		background: var(--surface-2);
		border-radius: 8px;
		border: 1px solid var(--border);
	}

	.lbl {
		font-size: 9px;
		fill: var(--text-faint);
	}

	.lbl.depot {
		text-anchor: middle;
		fill: var(--rlrp);
		font-weight: 700;
		font-size: 10px;
	}

	.lbl.depot.closed {
		fill: var(--text-faint);
		font-weight: 500;
	}

	.legend {
		display: flex;
		gap: 13px;
		flex-wrap: wrap;
		margin-top: 7px;
		font-size: 0.74rem;
		color: var(--text-muted);
	}

	.item {
		display: inline-flex;
		align-items: center;
		gap: 5px;
	}

	.sw {
		display: inline-block;
	}

	.sw.dot {
		width: 8px;
		height: 8px;
		border-radius: 50%;
	}

	.sw.square {
		width: 9px;
		height: 9px;
		border: 2px solid;
		border-radius: 2px;
	}

	.sw.line {
		width: 14px;
		height: 2.5px;
		border-radius: 2px;
	}

	.sw.dash {
		width: 14px;
		height: 0;
		border-top: 2px dashed;
	}
</style>

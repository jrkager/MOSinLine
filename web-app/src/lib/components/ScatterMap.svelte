<script lang="ts">
	/** Plain XY map. Our instances use Solomon Euclidean coordinates, not
	 *  lat/lon, so this is a scatter plot with equal axis scaling rather than a
	 *  geographic projection. */
	type Node = { id: number; kind: 'depot' | 'store'; x: number; y: number; label?: string };
	type Route = { coords: [number, number][]; color?: string; label?: string };

	let {
		nodes,
		routes = [],
		highlight = new Set<number>(),
		colorBy = null,
		height = 320
	}: {
		nodes: Node[];
		routes?: Route[];
		highlight?: Set<number>;
		colorBy?: Map<number, string> | null;
		height?: number;
	} = $props();

	const W = 640;
	const PAD = 26;

	const pts = $derived([...nodes.map((n) => [n.x, n.y] as [number, number]), ...routes.flatMap((r) => r.coords)]);
	const xMin = $derived(pts.length ? Math.min(...pts.map((p) => p[0])) : 0);
	const xMax = $derived(pts.length ? Math.max(...pts.map((p) => p[0])) : 1);
	const yMin = $derived(pts.length ? Math.min(...pts.map((p) => p[1])) : 0);
	const yMax = $derived(pts.length ? Math.max(...pts.map((p) => p[1])) : 1);

	// equal scale on both axes so distances stay visually truthful
	const span = $derived(Math.max(xMax - xMin, yMax - yMin) || 1);
	const scale = $derived(Math.min((W - 2 * PAD) / span, (height - 2 * PAD) / span));
	const offX = $derived(PAD + ((W - 2 * PAD) - (xMax - xMin) * scale) / 2);
	const offY = $derived(PAD + ((height - 2 * PAD) - (yMax - yMin) * scale) / 2);

	const sx = $derived((x: number) => offX + (x - xMin) * scale);
	// flip y so the plot reads like a map rather than screen coordinates
	const sy = $derived((y: number) => height - (offY + (y - yMin) * scale));

	function routePath(coords: [number, number][]): string {
		return coords.map((c, i) => `${i ? 'L' : 'M'} ${sx(c[0])} ${sy(c[1])}`).join(' ');
	}
</script>

<svg viewBox="0 0 {W} {height}" role="img" aria-label="instance map">
	{#each routes as route, i (i)}
		<path
			d={routePath(route.coords)}
			fill="none"
			stroke={route.color ?? 'var(--patt)'}
			stroke-width="1.8"
			stroke-linejoin="round"
			opacity="0.85"
		/>
	{/each}

	{#each nodes as node (node.id)}
		{#if node.kind === 'depot'}
			<rect
				x={sx(node.x) - 6}
				y={sy(node.y) - 6}
				width="12"
				height="12"
				rx="2.5"
				fill={highlight.has(node.id) ? 'var(--rlrp)' : 'var(--surface)'}
				stroke="var(--rlrp)"
				stroke-width="2"
			/>
			<text x={sx(node.x)} y={sy(node.y) - 11} class="lbl depot">
				{node.label ?? `D${Math.abs(node.id)}`}
			</text>
		{:else}
			<circle
				cx={sx(node.x)}
				cy={sy(node.y)}
				r="4.5"
				fill={colorBy?.get(node.id) ?? 'var(--text-muted)'}
				stroke="var(--surface)"
				stroke-width="1.2"
			/>
			<text x={sx(node.x) + 7} y={sy(node.y) + 3.5} class="lbl">{node.label ?? node.id}</text>
		{/if}
	{/each}
</svg>

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
</style>

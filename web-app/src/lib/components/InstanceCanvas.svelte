<script lang="ts">
	/** Interactive XY canvas for the instance builder: click empty space to add a
	 *  node, click a node to select it, drag to move it.
	 *
	 *  The coordinate system is the model's own (Solomon-style plain XY, no
	 *  projection). The view keeps a fixed world window rather than fitting to the
	 *  data, because a fit that changes while you place nodes makes editing
	 *  disorienting. */
	type Store = { store_id: number; x: number; y: number; demand_t_per_day: number };
	type Depot = { depot_id: number; x: number; y: number; max_size: number };

	let {
		stores,
		depots,
		selected = $bindable<{ kind: 'store' | 'depot'; id: number } | null>(null),
		mode = 'select',
		world = { minX: 0, minY: 0, maxX: 100, maxY: 100 },
		onadd,
		onmove
	}: {
		stores: Store[];
		depots: Depot[];
		selected?: { kind: 'store' | 'depot'; id: number } | null;
		mode?: 'select' | 'store' | 'depot';
		world?: { minX: number; minY: number; maxX: number; maxY: number };
		onadd?: (kind: 'store' | 'depot', x: number, y: number) => void;
		onmove?: (kind: 'store' | 'depot', id: number, x: number, y: number) => void;
	} = $props();

	const W = 640;
	const PAD = 22;

	let svgEl: SVGSVGElement | null = $state(null);
	let dragging = $state<{ kind: 'store' | 'depot'; id: number } | null>(null);
	let moved = $state(false);

	const spanX = $derived(Math.max(1e-9, world.maxX - world.minX));
	const spanY = $derived(Math.max(1e-9, world.maxY - world.minY));
	// Match the canvas aspect to the world, so the editable area fills the canvas
	// and there is no dead margin that looks clickable but places nodes off-grid.
	const height = $derived(Math.round((W - 2 * PAD) * (spanY / spanX)) + 2 * PAD);
	// equal scale, so distances stay visually truthful
	const scale = $derived(Math.min((W - 2 * PAD) / spanX, (height - 2 * PAD) / spanY));
	const offX = $derived(PAD + (W - 2 * PAD - spanX * scale) / 2);
	const offY = $derived(PAD + (height - 2 * PAD - spanY * scale) / 2);

	const sx = $derived((x: number) => offX + (x - world.minX) * scale);
	const sy = $derived((y: number) => height - (offY + (y - world.minY) * scale));
	const ix = $derived((px: number) => (px - offX) / scale + world.minX);
	const iy = $derived((py: number) => (height - py - offY) / scale + world.minY);

	const storeMax = $derived(Math.max(1e-9, ...stores.map((s) => s.demand_t_per_day)));
	const depotMax = $derived(Math.max(1e-9, ...depots.map((d) => d.max_size)));

	function storeR(s: Store): number {
		return 4 + 6 * Math.sqrt(Math.max(0, s.demand_t_per_day) / storeMax);
	}
	function depotHalf(d: Depot): number {
		return 6 + 5 * Math.sqrt(Math.max(0, d.max_size) / depotMax);
	}

	/** pointer position in SVG user units, accounting for the viewBox scaling */
	function toLocal(event: PointerEvent): { px: number; py: number } | null {
		if (!svgEl) return null;
		const rect = svgEl.getBoundingClientRect();
		return {
			px: ((event.clientX - rect.left) / rect.width) * W,
			py: ((event.clientY - rect.top) / rect.height) * height
		};
	}

	function clampWorld(x: number, y: number): [number, number] {
		return [
			Math.min(world.maxX, Math.max(world.minX, x)),
			Math.min(world.maxY, Math.max(world.minY, y))
		];
	}

	function onCanvasPointerDown(event: PointerEvent) {
		if (mode === 'select') return;
		const p = toLocal(event);
		if (!p) return;
		const [x, y] = clampWorld(ix(p.px), iy(p.py));
		onadd?.(mode, Math.round(x * 10) / 10, Math.round(y * 10) / 10);
	}

	function onNodePointerDown(event: PointerEvent, kind: 'store' | 'depot', id: number) {
		// in placement mode a click on a node should still place, not drag
		if (mode !== 'select') return;
		event.stopPropagation();
		// Select first: pointer capture is only a nicety for dragging past the
		// element's edge, and it throws when the pointerId is not active — which
		// would otherwise abort this handler and lose the selection entirely.
		selected = { kind, id };
		dragging = { kind, id };
		moved = false;
		try {
			(event.target as Element).setPointerCapture?.(event.pointerId);
		} catch {
			/* no capture available; dragging still works while inside the svg */
		}
	}

	function onPointerMove(event: PointerEvent) {
		if (!dragging) return;
		const p = toLocal(event);
		if (!p) return;
		const [x, y] = clampWorld(ix(p.px), iy(p.py));
		moved = true;
		onmove?.(dragging.kind, dragging.id, Math.round(x * 10) / 10, Math.round(y * 10) / 10);
	}

	function onPointerUp() {
		dragging = null;
	}

	function isSel(kind: 'store' | 'depot', id: number): boolean {
		return selected?.kind === kind && selected.id === id;
	}

	const gridStep = $derived(Math.max(1, Math.round(spanX / 10)));
	const gridXs = $derived(
		Array.from({ length: Math.floor(spanX / gridStep) + 1 }, (_, i) => world.minX + i * gridStep)
	);
	const gridYs = $derived(
		Array.from({ length: Math.floor(spanY / gridStep) + 1 }, (_, i) => world.minY + i * gridStep)
	);
</script>

<svg
	bind:this={svgEl}
	viewBox="0 0 {W} {height}"
	class:placing={mode !== 'select'}
	role="application"
	aria-label="instance editor canvas"
	onpointerdown={onCanvasPointerDown}
	onpointermove={onPointerMove}
	onpointerup={onPointerUp}
	onpointerleave={onPointerUp}
>
	<!-- grid -->
	{#each gridXs as gx}
		<line x1={sx(gx)} y1={sy(world.minY)} x2={sx(gx)} y2={sy(world.maxY)} class="grid" />
	{/each}
	{#each gridYs as gy}
		<line x1={sx(world.minX)} y1={sy(gy)} x2={sx(world.maxX)} y2={sy(gy)} class="grid" />
	{/each}
	<rect
		x={sx(world.minX)}
		y={sy(world.maxY)}
		width={spanX * scale}
		height={spanY * scale}
		class="frame"
	/>

	{#each stores as s (s.store_id)}
		<circle
			cx={sx(s.x)}
			cy={sy(s.y)}
			r={storeR(s)}
			class="store"
			class:sel={isSel('store', s.store_id)}
			onpointerdown={(e) => onNodePointerDown(e, 'store', s.store_id)}
			role="button"
			tabindex="-1"
		/>
		<text x={sx(s.x) + storeR(s) + 3} y={sy(s.y) + 3.5} class="lbl">{s.store_id}</text>
	{/each}

	{#each depots as d (d.depot_id)}
		{@const h = depotHalf(d)}
		<rect
			x={sx(d.x) - h}
			y={sy(d.y) - h}
			width={h * 2}
			height={h * 2}
			rx="2.5"
			class="depot"
			class:sel={isSel('depot', d.depot_id)}
			onpointerdown={(e) => onNodePointerDown(e, 'depot', d.depot_id)}
			role="button"
			tabindex="-1"
		/>
		<text x={sx(d.x)} y={sy(d.y) - h - 5} class="lbl depot-lbl">D{Math.abs(d.depot_id)}</text>
	{/each}

	{#if !stores.length && !depots.length}
		<text x={W / 2} y={height / 2} class="placeholder">
			Pick “Add store” or “Add depot”, then click on the canvas
		</text>
	{/if}
</svg>

<style>
	svg {
		width: 100%;
		height: auto;
		background: var(--surface-2);
		border: 1px solid var(--border);
		border-radius: 8px;
		touch-action: none;
	}

	svg.placing {
		cursor: crosshair;
	}

	.grid {
		stroke: var(--border);
		stroke-width: 0.6;
		opacity: 0.7;
	}

	.frame {
		fill: none;
		stroke: var(--border-strong);
		stroke-width: 1;
	}

	.store {
		fill: var(--sim);
		stroke: var(--surface);
		stroke-width: 1.4;
		cursor: grab;
	}

	.store.sel {
		stroke: var(--text);
		stroke-width: 2.4;
	}

	.depot {
		fill: var(--rlrp);
		stroke: var(--surface);
		stroke-width: 1.6;
		cursor: grab;
	}

	.depot.sel {
		stroke: var(--text);
		stroke-width: 2.4;
	}

	.lbl {
		font-size: 9px;
		fill: var(--text-faint);
		pointer-events: none;
	}

	.depot-lbl {
		text-anchor: middle;
		fill: var(--rlrp);
		font-weight: 700;
		font-size: 10px;
	}

	.placeholder {
		text-anchor: middle;
		fill: var(--text-faint);
		font-size: 13px;
		pointer-events: none;
	}
</style>

<script lang="ts">
	/** Minimal multi-series line chart. Hand-rolled SVG: the charts here are
	 *  simple enough that a plotting library would be more code, not less. */
	type Series = { label: string; color?: string; points: [number, number][]; dim?: boolean };

	let {
		series,
		xLabel = '',
		yLabel = '',
		height = 220
	}: { series: Series[]; xLabel?: string; yLabel?: string; height?: number } = $props();

	const W = 640;
	const PAD = { top: 12, right: 14, bottom: 34, left: 58 };

	const all = $derived(series.flatMap((s) => s.points));
	const xs = $derived(all.map((p) => p[0]));
	const ys = $derived(all.map((p) => p[1]));

	const xMin = $derived(xs.length ? Math.min(...xs) : 0);
	const xMax = $derived(xs.length ? Math.max(...xs) : 1);
	const yMinRaw = $derived(ys.length ? Math.min(...ys) : 0);
	const yMaxRaw = $derived(ys.length ? Math.max(...ys) : 1);
	const pad = $derived((yMaxRaw - yMinRaw) * 0.08 || Math.abs(yMaxRaw) * 0.08 || 1);
	const yMin = $derived(yMinRaw - pad);
	const yMax = $derived(yMaxRaw + pad);

	const sx = $derived((x: number) =>
		PAD.left + ((x - xMin) / (xMax - xMin || 1)) * (W - PAD.left - PAD.right)
	);
	const sy = $derived((y: number) =>
		PAD.top + (1 - (y - yMin) / (yMax - yMin || 1)) * (height - PAD.top - PAD.bottom)
	);

	function path(points: [number, number][]): string {
		if (!points.length) return '';
		return points.map((p, i) => `${i ? 'L' : 'M'} ${sx(p[0])} ${sy(p[1])}`).join(' ');
	}

	function ticks(min: number, max: number, count = 4): number[] {
		if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) return [min];
		const step = (max - min) / count;
		return Array.from({ length: count + 1 }, (_, i) => min + step * i);
	}

	function fmt(v: number): string {
		const abs = Math.abs(v);
		if (abs >= 10000) return v.toExponential(1);
		if (abs >= 100) return v.toFixed(0);
		if (abs >= 1) return v.toFixed(1);
		return v.toFixed(2);
	}
</script>

{#if all.length === 0}
	<div class="empty">no data yet</div>
{:else}
	<svg viewBox="0 0 {W} {height}" role="img" aria-label={yLabel || 'chart'}>
		{#each ticks(yMin, yMax) as t}
			<line x1={PAD.left} x2={W - PAD.right} y1={sy(t)} y2={sy(t)} class="grid" />
			<text x={PAD.left - 8} y={sy(t) + 4} class="tick" text-anchor="end">{fmt(t)}</text>
		{/each}
		{#each ticks(xMin, xMax, 4) as t}
			<text x={sx(t)} y={height - PAD.bottom + 17} class="tick" text-anchor="middle">{fmt(t)}</text>
		{/each}

		{#each series as s (s.label)}
			<path
				d={path(s.points)}
				fill="none"
				stroke={s.color ?? 'var(--patt)'}
				stroke-width={s.dim ? 1.4 : 2.2}
				opacity={s.dim ? 0.35 : 1}
				stroke-linejoin="round"
			/>
			{#if s.points.length && !s.dim}
				<circle
					cx={sx(s.points[s.points.length - 1][0])}
					cy={sy(s.points[s.points.length - 1][1])}
					r="3.5"
					fill={s.color ?? 'var(--patt)'}
				/>
			{/if}
		{/each}

		{#if yLabel}
			<text
				class="axis"
				transform="rotate(-90)"
				x={-(height - PAD.bottom) / 2 - PAD.top}
				y="13"
				text-anchor="middle">{yLabel}</text
			>
		{/if}
		{#if xLabel}
			<text class="axis" x={(W + PAD.left) / 2} y={height - 3} text-anchor="middle">{xLabel}</text>
		{/if}
	</svg>

	{#if series.length > 1}
		<div class="legend">
			{#each series as s (s.label)}
				<span class="item" class:dim={s.dim}>
					<span class="swatch" style="background:{s.color ?? 'var(--patt)'}"></span>{s.label}
				</span>
			{/each}
		</div>
	{/if}
{/if}

<style>
	svg {
		width: 100%;
		height: auto;
	}

	.grid {
		stroke: var(--border);
		stroke-width: 1;
	}

	.tick {
		font-size: 10px;
		fill: var(--text-faint);
	}

	.axis {
		font-size: 11px;
		fill: var(--text-muted);
		font-weight: 550;
	}

	.legend {
		display: flex;
		gap: 12px;
		flex-wrap: wrap;
		margin-top: 6px;
		font-size: 0.76rem;
		color: var(--text-muted);
	}

	.item {
		display: inline-flex;
		align-items: center;
		gap: 5px;
	}

	.item.dim {
		opacity: 0.5;
	}

	.swatch {
		width: 10px;
		height: 3px;
		border-radius: 2px;
	}
</style>

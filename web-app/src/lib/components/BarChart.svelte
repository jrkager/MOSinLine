<script lang="ts">
	/** Horizontal bars with an optional reference line (e.g. vehicle capacity)
	 *  and optional stacked segments. Used for the weekday load profile, the
	 *  delivery-frequency distribution and depot utilisation. */
	type Bar = {
		label: string;
		value: number;
		/** stacked parts; when given, `value` is only used for the readout */
		parts?: { value: number; color: string; label: string }[];
		color?: string;
		note?: string;
	};

	let {
		bars,
		reference = null,
		referenceLabel = '',
		unit = '',
		digits = 1,
		max = null
	}: {
		bars: Bar[];
		reference?: number | null;
		referenceLabel?: string;
		unit?: string;
		digits?: number;
		max?: number | null;
	} = $props();

	const dataMax = $derived(Math.max(1e-9, ...bars.map((b) => b.value)));
	// Leave headroom past the reference line so it sits visibly inside the track
	// rather than flush against the right edge.
	const upper = $derived(
		max ?? (reference !== null ? Math.max(dataMax, reference * 1.08) : dataMax)
	);

	function pctOf(v: number): number {
		return Math.max(0, Math.min(100, (v / upper) * 100));
	}

	function fmt(v: number): string {
		return v.toLocaleString(undefined, {
			minimumFractionDigits: digits,
			maximumFractionDigits: digits
		});
	}
</script>

<div class="chart">
	{#each bars as bar (bar.label)}
		<div class="row">
			<span class="label">{bar.label}</span>
			<span class="track">
				{#if reference !== null}
					<span class="ref" style="left: {pctOf(reference)}%" title={referenceLabel}></span>
				{/if}
				{#if bar.parts?.length}
					<span class="stack">
						{#each bar.parts as part (part.label)}
							<span
								class="seg"
								style="width: {pctOf(part.value)}%; background: {part.color}"
								title="{part.label}: {fmt(part.value)} {unit}"
							></span>
						{/each}
					</span>
				{:else}
					<span
						class="fill"
						style="width: {pctOf(bar.value)}%; background: {bar.color ?? 'var(--patt)'}"
					></span>
				{/if}
			</span>
			<span class="value">{fmt(bar.value)}{unit ? ` ${unit}` : ''}</span>
		</div>
		{#if bar.note}
			<div class="note">{bar.note}</div>
		{/if}
	{/each}
	{#if reference !== null && referenceLabel}
		<div class="ref-legend">
			<span class="ref-mark"></span>{referenceLabel} = {fmt(reference)}{unit ? ` ${unit}` : ''}
		</div>
	{/if}
</div>

<style>
	.chart {
		display: flex;
		flex-direction: column;
		gap: 5px;
	}

	.row {
		display: grid;
		grid-template-columns: 58px 1fr 96px;
		gap: 9px;
		align-items: center;
		font-size: 0.8rem;
	}

	.label {
		color: var(--text-muted);
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.track {
		position: relative;
		background: var(--surface-2);
		border: 1px solid var(--border);
		border-radius: 4px;
		height: 15px;
		overflow: hidden;
	}

	.fill,
	.stack {
		display: flex;
		height: 100%;
	}

	.fill {
		display: block;
	}

	.seg {
		display: block;
		height: 100%;
	}

	.ref {
		position: absolute;
		top: 0;
		bottom: 0;
		width: 0;
		border-left: 2px dashed var(--bad);
		z-index: 2;
	}

	.value {
		text-align: right;
		font-variant-numeric: tabular-nums;
		color: var(--text);
	}

	.note {
		grid-column: 2 / -1;
		font-size: 0.72rem;
		color: var(--text-faint);
		margin: -2px 0 3px 67px;
	}

	.ref-legend {
		margin-top: 4px;
		font-size: 0.72rem;
		color: var(--text-faint);
		display: flex;
		align-items: center;
		gap: 6px;
	}

	.ref-mark {
		display: inline-block;
		width: 13px;
		height: 0;
		border-top: 2px dashed var(--bad);
	}
</style>

<script lang="ts">
	/** stores x Mon..Sat delivery grid — the clearest single view of a PATT
	 *  solution. Each row carries a pattern ribbon (the 6 bits as blocks) and
	 *  cells shaded by delivered tonnage, optionally split by product segment. */
	import { num } from '$lib/format';

	type Store = {
		store_id: number;
		pattern: number[];
		frequency: number;
		delivery_t: number[];
		weekly_t: number;
		delivery_by_segment_t?: Record<string, number[]>;
	};

	let {
		stores,
		days,
		mode = $bindable<'quantity' | 'segments'>('quantity')
	}: { stores: Store[]; days: string[]; mode?: 'quantity' | 'segments' } = $props();

	const SEG_COLORS: Record<string, string> = {
		fresh: 'var(--sim)',
		dry: 'var(--patt)',
		frozen: 'var(--rlrp)'
	};
	const SEG_ORDER = ['fresh', 'dry', 'frozen'];

	const maxQty = $derived(
		Math.max(1e-9, ...stores.flatMap((s) => s.delivery_t.filter((_, i) => s.pattern[i] === 1)))
	);

	const hasSegments = $derived(stores.some((s) => s.delivery_by_segment_t));

	function intensity(store: Store, day: number): number {
		if (!store.pattern[day]) return 0;
		return 0.16 + 0.74 * (store.delivery_t[day] / maxQty);
	}

	function segParts(store: Store, day: number) {
		const by = store.delivery_by_segment_t;
		if (!by) return [];
		return SEG_ORDER.filter((s) => by[s]).map((s) => ({
			seg: s,
			value: by[s][day] ?? 0,
			color: SEG_COLORS[s]
		}));
	}

	/** cell width fraction for a stacked segment bar, relative to the day max */
	function segWidth(value: number): number {
		return Math.max(0, Math.min(100, (value / maxQty) * 100));
	}

	const sorted = $derived([...stores].sort((a, b) => b.frequency - a.frequency || a.store_id - b.store_id));
</script>

<div class="head-row">
	<div class="hint">
		{#if mode === 'quantity'}
			Shading is the delivered tonnage; a dot means no delivery that day.
		{:else}
			Each cell stacks the delivered tonnage by product segment.
		{/if}
	</div>
	{#if hasSegments}
		<div class="row">
			<button class:primary={mode === 'quantity'} onclick={() => (mode = 'quantity')}>
				Quantity
			</button>
			<button class:primary={mode === 'segments'} onclick={() => (mode = 'segments')}>
				Segments
			</button>
		</div>
	{/if}
</div>

{#if mode === 'segments'}
	<div class="seg-legend">
		{#each SEG_ORDER as s}
			<span class="li"><span class="sw" style="background:{SEG_COLORS[s]}"></span>{s}</span>
		{/each}
	</div>
{/if}

<div class="table-wrap">
	<table>
		<thead>
			<tr>
				<th>Store</th>
				<th>Pattern</th>
				{#each days as d}<th>{d}</th>{/each}
				<th>Freq</th>
				<th>Weekly t</th>
			</tr>
		</thead>
		<tbody>
			{#each sorted as s (s.store_id)}
				<tr>
					<td class="mono">{s.store_id}</td>
					<td>
						<span class="ribbon" title={s.pattern.join('')}>
							{#each s.pattern as bit, i}
								<span class="bit" class:on={bit === 1} title={days[i]}></span>
							{/each}
						</span>
					</td>
					{#each days as _, day}
						<td class="cell">
							{#if s.pattern[day]}
								{#if mode === 'segments'}
									<span class="stack" title="{num(s.delivery_t[day])} t total">
										{#each segParts(s, day) as part (part.seg)}
											{#if part.value > 1e-9}
												<span
													class="seg"
													style="width:{segWidth(part.value)}%; background:{part.color}"
													title="{part.seg}: {num(part.value)} t"
												></span>
											{/if}
										{/each}
									</span>
								{:else}
									<span
										class="chip"
										style="background: color-mix(in srgb, var(--patt) {intensity(s, day) *
											100}%, transparent)"
										title="{num(s.delivery_t[day])} t"
									>
										{num(s.delivery_t[day], 1)}
									</span>
								{/if}
							{:else}
								<span class="off">·</span>
							{/if}
						</td>
					{/each}
					<td>{s.frequency}</td>
					<td>{num(s.weekly_t)}</td>
				</tr>
			{/each}
		</tbody>
	</table>
</div>

<style>
	.head-row {
		display: flex;
		justify-content: space-between;
		align-items: center;
		gap: 12px;
		margin-bottom: 8px;
		flex-wrap: wrap;
	}

	.cell {
		text-align: center;
		padding: 3px 5px;
		min-width: 62px;
	}

	.chip {
		display: inline-block;
		min-width: 42px;
		padding: 2px 6px;
		border-radius: 5px;
		font-size: 0.74rem;
		font-variant-numeric: tabular-nums;
		border: 1px solid color-mix(in srgb, var(--patt) 35%, transparent);
	}

	.stack {
		display: flex;
		height: 13px;
		width: 100%;
		background: var(--surface-2);
		border: 1px solid var(--border);
		border-radius: 4px;
		overflow: hidden;
	}

	.seg {
		display: block;
		height: 100%;
	}

	.off {
		color: var(--text-faint);
	}

	.ribbon {
		display: inline-flex;
		gap: 2px;
	}

	.bit {
		width: 8px;
		height: 13px;
		border-radius: 2px;
		background: var(--surface-2);
		border: 1px solid var(--border);
	}

	.bit.on {
		background: var(--patt);
		border-color: var(--patt);
	}

	.seg-legend {
		display: flex;
		gap: 12px;
		font-size: 0.74rem;
		color: var(--text-muted);
		margin-bottom: 7px;
	}

	.li {
		display: inline-flex;
		align-items: center;
		gap: 5px;
	}

	.sw {
		width: 9px;
		height: 9px;
		border-radius: 2px;
		display: inline-block;
	}
</style>

<script lang="ts">
	/** stores x Mon..Sat grid of delivery flags and quantities - the most
	 *  communicative single view of a PATT solution. */
	import { num } from '$lib/format';

	type Store = {
		store_id: number;
		pattern: number[];
		frequency: number;
		delivery_t: number[];
		weekly_t: number;
	};

	let { stores, days }: { stores: Store[]; days: string[] } = $props();

	const maxQty = $derived(
		Math.max(1e-9, ...stores.flatMap((s) => s.delivery_t.filter((_, i) => s.pattern[i] === 1)))
	);

	function intensity(store: Store, day: number): number {
		if (!store.pattern[day]) return 0;
		return 0.18 + 0.72 * (store.delivery_t[day] / maxQty);
	}
</script>

<div class="table-wrap">
	<table>
		<thead>
			<tr>
				<th>Store</th>
				{#each days as d}<th>{d}</th>{/each}
				<th>Freq</th>
				<th>Weekly t</th>
			</tr>
		</thead>
		<tbody>
			{#each stores as s (s.store_id)}
				<tr>
					<td class="mono">{s.store_id}</td>
					{#each days as _, day}
						<td class="cell">
							{#if s.pattern[day]}
								<span
									class="chip"
									style="background: color-mix(in srgb, var(--patt) {intensity(s, day) * 100}%, transparent)"
									title="{num(s.delivery_t[day])} t"
								>
									{num(s.delivery_t[day], 1)}
								</span>
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
	.cell {
		text-align: center;
		padding: 3px 5px;
	}

	.chip {
		display: inline-block;
		min-width: 40px;
		padding: 2px 6px;
		border-radius: 5px;
		font-size: 0.74rem;
		font-variant-numeric: tabular-nums;
		border: 1px solid color-mix(in srgb, var(--patt) 35%, transparent);
	}

	.off {
		color: var(--text-faint);
	}
</style>

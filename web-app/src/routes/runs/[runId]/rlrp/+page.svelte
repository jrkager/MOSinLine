<script lang="ts">
	import { onMount } from 'svelte';
	import { page } from '$app/state';
	import { createRunContext } from '$lib/run-context.svelte';
	import { num } from '$lib/format';
	import MetricTile from '$lib/components/MetricTile.svelte';
	import ScatterMap from '$lib/components/ScatterMap.svelte';
	import StageNav from '$lib/components/StageNav.svelte';

	const ctx = createRunContext(
		() => page.params.runId!,
		() => page.url.searchParams.get('round')
	);

	let rlrp = $state<any>(null);
	let loadError = $state<string | null>(null);
	let scenario = $state<number | null>(null);

	const DEPOT_COLORS = ['var(--rlrp)', 'var(--patt)', 'var(--sim)', 'var(--feedback)', '#db2777'];

	async function loadRound(round: number | null) {
		if (round === null) return;
		try {
			rlrp = await ctx.artifact(`rounds/${round}/rlrp.json`);
			loadError = null;
			if (scenario === null) scenario = rlrp?.scenarios?.[0]?.scenario_id ?? null;
		} catch (e) {
			rlrp = null;
			loadError = e instanceof Error ? e.message : String(e);
		}
	}

	onMount(async () => {
		await ctx.load();
		await loadRound(ctx.round);
	});

	let lastRound: number | null = null;
	$effect(() => {
		if (ctx.round !== lastRound) {
			lastRound = ctx.round;
			loadRound(ctx.round);
		}
	});

	const current = $derived(rlrp?.scenarios?.find((s: any) => s.scenario_id === scenario) ?? null);

	// colour each store by the depot serving it in the selected scenario
	const colorBy = $derived.by(() => {
		const map = new Map<number, string>();
		current?.depots?.forEach((d: any, i: number) => {
			for (const store of d.stores) map.set(store, DEPOT_COLORS[i % DEPOT_COLORS.length]);
		});
		return map;
	});

	const openDepots = $derived(
		new Set<number>((current?.depots ?? []).filter((d: any) => d.open).map((d: any) => d.depot_id))
	);

	const nodes = $derived(ctx.instance?.nodes ?? []);
</script>

<div class="card">
	<div class="card-header">
		<div>
			<h1>RLRP — where the depots go</h1>
			<div class="card-sub">
				Robust location-routing decides which depot candidates open, how big they are, and which
				stores each one serves — separately per demand scenario.
			</div>
		</div>
	</div>
	<StageNav runId={ctx.runId} round={ctx.round} />
</div>

{#if ctx.error || loadError}
	<div class="error-banner">{ctx.error ?? loadError}</div>
{:else if !rlrp}
	<div class="empty">{ctx.loading ? 'loading…' : 'no RLRP results for this round yet'}</div>
{:else}
	<div class="card">
		<div class="card-header">
			<h2>Round {rlrp.round} solve</h2>
			{#if rlrp.reused}<span class="pill">reused from the previous round</span>{/if}
		</div>
		<div class="tiles">
			<MetricTile k="Objective" v={num(rlrp.cost, 1)} />
			<MetricTile
				k="Gap reached"
				v={rlrp.reached_gap === null ? '—' : `${num(rlrp.reached_gap * 100, 2)}%`}
			/>
			<MetricTile k="Iterations" v={rlrp.iterations ?? '—'} />
			<MetricTile k="Runtime" v={`${num(rlrp.runtime_sec, 1)} s`} />
			<MetricTile k="Master time" v={`${num(rlrp.time_master_sec, 2)} s`} />
			<MetricTile k="2nd stage time" v={`${num(rlrp.time_second_stage_sec, 2)} s`} />
		</div>
	</div>

	<div class="card">
		<div class="card-header">
			<h2>Depot decisions</h2>
			<div class="row">
				{#each rlrp.scenarios as s (s.scenario_id)}
					<button class:primary={scenario === s.scenario_id} onclick={() => (scenario = s.scenario_id)}>
						Scenario {s.scenario_id}
					</button>
				{/each}
			</div>
		</div>

		{#if current}
			<div class="split">
				<div>
					<ScatterMap {nodes} highlight={openDepots} colorBy={colorBy} height={340} />
					<div class="hint" style="margin-top:6px">
						Squares are depot candidates (filled = opened); dots are stores, coloured by the
						depot serving them. Coordinates are Solomon Euclidean, not geographic.
					</div>
				</div>
				<div class="table-wrap">
					<table>
						<thead>
							<tr>
								<th>Depot</th>
								<th>Status</th>
								<th>Size t/day</th>
								<th>Stores</th>
							</tr>
						</thead>
						<tbody>
							{#each current.depots as d (d.depot_id)}
								<tr class:highlight={d.open}>
									<td class="mono">D{Math.abs(d.depot_id)}</td>
									<td>{d.open ? 'open' : 'closed'}</td>
									<td>{d.open ? num(d.size_t_per_day) : '—'}</td>
									<td>{d.n_stores || '—'}</td>
								</tr>
							{/each}
						</tbody>
					</table>
					{#each current.depots.filter((d: any) => d.open) as d (d.depot_id)}
						<div class="assign">
							<strong class="mono">D{Math.abs(d.depot_id)}</strong>
							<span class="faint mono">{d.stores.join(', ')}</span>
						</div>
					{/each}
				</div>
			</div>
		{/if}
	</div>
{/if}

<style>
	.split {
		display: grid;
		grid-template-columns: minmax(300px, 1.3fr) minmax(260px, 1fr);
		gap: 16px;
		align-items: start;
	}

	@media (max-width: 900px) {
		.split {
			grid-template-columns: 1fr;
		}
	}

	.assign {
		margin-top: 8px;
		font-size: 0.78rem;
		display: flex;
		gap: 8px;
		align-items: baseline;
	}
</style>

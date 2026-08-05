<script lang="ts">
	import { onMount } from 'svelte';
	import { page } from '$app/state';
	import { createRunContext } from '$lib/run-context.svelte';
	import { num, pct } from '$lib/format';
	import LineChart from '$lib/components/LineChart.svelte';
	import MetricTile from '$lib/components/MetricTile.svelte';
	import PatternCalendar from '$lib/components/PatternCalendar.svelte';
	import ScatterMap from '$lib/components/ScatterMap.svelte';
	import StageNav from '$lib/components/StageNav.svelte';

	const ctx = createRunContext(
		() => page.params.runId!,
		() => page.url.searchParams.get('round')
	);

	let index = $state<any>(null);
	let unit = $state<any>(null);
	let unitId = $state<string | null>(null);
	let day = $state(0);
	let loadError = $state<string | null>(null);

	const DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
	const ROUTE_COLORS = ['var(--patt)', 'var(--rlrp)', 'var(--sim)', 'var(--feedback)', '#db2777', '#0891b2'];

	async function loadIndex(round: number | null) {
		if (round === null) return;
		try {
			index = await ctx.artifact(`rounds/${round}/patt/index.json`);
			loadError = null;
			unitId = index?.units?.[0]?.id ?? null;
			await loadUnit();
		} catch (e) {
			index = null;
			loadError = e instanceof Error ? e.message : String(e);
		}
	}

	async function loadUnit() {
		if (!unitId || ctx.round === null) return;
		try {
			unit = await ctx.artifact(`rounds/${ctx.round}/patt/${unitId}.json`);
		} catch {
			unit = null;
		}
	}

	onMount(async () => {
		await ctx.load();
		await loadIndex(ctx.round);
	});

	let lastRound: number | null = null;
	$effect(() => {
		if (ctx.round !== lastRound) {
			lastRound = ctx.round;
			loadIndex(ctx.round);
		}
	});

	async function select(id: string) {
		unitId = id;
		await loadUnit();
	}

	const dayData = $derived(unit?.routes_by_day?.[day] ?? null);

	const mapNodes = $derived.by(() => {
		if (!unit) return [];
		return [
			{ id: 0, kind: 'depot' as const, x: unit.depot.x, y: unit.depot.y, label: 'DC' },
			...unit.stores.map((s: any) => ({
				id: s.store_id,
				kind: 'store' as const,
				x: s.x,
				y: s.y,
				label: String(s.store_id)
			}))
		];
	});

	const mapRoutes = $derived(
		(dayData?.routes ?? []).map((r: any, i: number) => ({
			coords: r.coords,
			color: ROUTE_COLORS[i % ROUTE_COLORS.length]
		}))
	);

	const convergence = $derived(
		unit?.convergence?.length
			? [
					{
						label: unit.id,
						color: 'var(--patt)',
						points: unit.convergence.map((p: any) => [p.iteration, p.cost] as [number, number])
					}
				]
			: []
	);
</script>

<div class="card">
	<div class="card-header">
		<div>
			<h1>PATT — the delivery plan</h1>
			<div class="card-sub">
				For each depot RLRP opened, the ALNS picks one weekly delivery pattern per store and the
				routes that serve them.
			</div>
		</div>
	</div>
	<StageNav runId={ctx.runId} round={ctx.round} />
</div>

{#if ctx.error || loadError}
	<div class="error-banner">{ctx.error ?? loadError}</div>
{:else if !index}
	<div class="empty">{ctx.loading ? 'loading…' : 'no PATT results for this round yet'}</div>
{:else}
	<div class="card">
		<div class="card-header">
			<h2>Scenario / depot</h2>
			<div class="row">
				{#each index.units as u (u.id)}
					<button class:primary={unitId === u.id} onclick={() => select(u.id)}>
						s{u.scenario_id} · D{Math.abs(u.depot_id)}
						<span class="faint"> ({u.n_stores})</span>
					</button>
				{/each}
			</div>
		</div>

		{#if !unit}
			<div class="empty">this unit was not solved in this round</div>
		{:else}
			<div class="tiles">
				<MetricTile k="Objective" v={num(unit.objective, 1)} />
				<MetricTile k="Pattern cost" v={num(unit.pattern_cost, 1)} />
				<MetricTile k="Routing cost" v={num(unit.routing_cost, 1)} />
				<MetricTile k="lambda" v={num(unit.lambda, 2)} />
				<MetricTile k="Runtime" v={`${num(unit.runtime_sec, 1)} s`} />
				<MetricTile k="Feasible" v={unit.feasible ? 'yes' : 'violations'} />
			</div>
		{/if}
	</div>

	{#if unit}
		<div class="card">
			<div class="card-header">
				<div>
					<h2>Predicted KPIs</h2>
					<div class="card-sub">What the PATT model expects; the SIM stage checks these.</div>
				</div>
			</div>
			<div class="tiles">
				<MetricTile k="Weekly demand" v={`${num(unit.predicted.demand_t)} t`} />
				<MetricTile k="Waste" v={pct(unit.predicted.waste_pct)} />
				<MetricTile k="Stockout" v={pct(unit.predicted.stockout_pct)} />
				<MetricTile k="Food-waste CO2" v={`${num(unit.predicted.fw_co2_kg_per_week, 0)} kg/wk`} />
				<MetricTile
					k="Transport CO2"
					v={`${num(unit.predicted.transport_co2_kg_per_week, 0)} kg/wk`}
				/>
				<MetricTile k="Distance" v={`${num(unit.predicted.km_per_week, 0)} km/wk`} />
			</div>
		</div>

		<div class="card">
			<div class="card-header">
				<div>
					<h2>Delivery pattern per store</h2>
					<div class="card-sub">
						Shading is the delivered tonnage on that weekday; a dot means no delivery.
					</div>
				</div>
				<div class="row">
					{#each Object.entries(unit.frequency_histogram) as [freq, count]}
						<span class="pill">{count}× freq {freq}</span>
					{/each}
				</div>
			</div>
			<PatternCalendar stores={unit.stores} days={DAYS} />
		</div>

		<div class="card">
			<div class="card-header">
				<div>
					<h2>Routes</h2>
					<div class="card-sub">One vehicle per colour; the square is the depot.</div>
				</div>
				<div class="row">
					{#each DAYS as d, i}
						<button class:primary={day === i} onclick={() => (day = i)}>{d}</button>
					{/each}
				</div>
			</div>
			<div class="split">
				<ScatterMap nodes={mapNodes} routes={mapRoutes} highlight={new Set([0])} height={340} />
				<div>
					<div class="tiles" style="grid-template-columns: 1fr 1fr">
						<MetricTile k="Vehicles" v={dayData?.n_vehicles ?? 0} />
						<MetricTile k="Distance" v={`${num(dayData?.distance_km, 1)} km`} />
						<MetricTile k="Delivered" v={`${num(dayData?.delivered_t)} t`} />
					</div>
					{#if dayData?.routes?.length}
						<div class="table-wrap" style="margin-top:10px">
							<table>
								<thead>
									<tr><th>Vehicle</th><th>Stops</th><th>Load t</th><th>km</th></tr>
								</thead>
								<tbody>
									{#each dayData.routes as r, i (r.vehicle_id)}
										<tr>
											<td>
												<span class="swatch" style="background:{ROUTE_COLORS[i % ROUTE_COLORS.length]}"
												></span>
												{r.vehicle_id}
											</td>
											<td class="mono faint">
												{r.stop_store_ids.map((s: number | null) => s ?? 'DC').join(' → ')}
											</td>
											<td>{num(r.departure_load_t)}</td>
											<td>{num(r.distance_km, 1)}</td>
										</tr>
									{/each}
								</tbody>
							</table>
						</div>
					{:else}
						<div class="empty" style="margin-top:10px">no deliveries on this day</div>
					{/if}
				</div>
			</div>
		</div>

		{#if convergence.length}
			<div class="card">
				<div class="card-header"><h2>ALNS convergence</h2></div>
				<LineChart series={convergence} xLabel="iteration" yLabel="objective" />
			</div>
		{/if}

		{#if unit.operator_performance?.length}
			<div class="card">
				<div class="card-header">
					<div>
						<h2>Operator performance</h2>
						<div class="card-sub">Which ALNS pattern operators actually found improvements.</div>
					</div>
				</div>
				<div class="table-wrap">
					<table>
						<thead>
							<tr>
								{#each Object.keys(unit.operator_performance[0]) as col}
									<th>{col}</th>
								{/each}
							</tr>
						</thead>
						<tbody>
							{#each unit.operator_performance as row, i (i)}
								<tr>
									{#each Object.values(row) as value}
										<td>{typeof value === 'number' ? num(value, 2) : value}</td>
									{/each}
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			</div>
		{/if}
	{/if}
{/if}

<style>
	.split {
		display: grid;
		grid-template-columns: minmax(300px, 1.1fr) minmax(280px, 1fr);
		gap: 16px;
		align-items: start;
	}

	@media (max-width: 900px) {
		.split {
			grid-template-columns: 1fr;
		}
	}

	.swatch {
		display: inline-block;
		width: 9px;
		height: 3px;
		border-radius: 2px;
		margin-right: 5px;
		vertical-align: middle;
	}
</style>

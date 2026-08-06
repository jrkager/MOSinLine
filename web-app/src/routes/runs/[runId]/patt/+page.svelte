<script lang="ts">
	import { onMount } from 'svelte';
	import { page } from '$app/state';
	import { createRunContext } from '$lib/run-context.svelte';
	import { num, pct } from '$lib/format';
	import BarChart from '$lib/components/BarChart.svelte';
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
	let calendarMode = $state<'quantity' | 'segments'>('quantity');

	const DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
	const ROUTE_COLORS = [
		'var(--patt)',
		'var(--rlrp)',
		'var(--sim)',
		'var(--feedback)',
		'#db2777',
		'#0891b2'
	];
	const SEG_COLORS: Record<string, string> = {
		fresh: 'var(--sim)',
		dry: 'var(--patt)',
		frozen: 'var(--rlrp)'
	};

	async function loadIndex(round: number | null) {
		if (round === null) return;
		try {
			index = await ctx.artifact(`rounds/${round}/patt/index.json`);
			loadError = null;
			if (!unitId || !index?.units?.some((u: any) => u.id === unitId)) {
				unitId = index?.units?.[0]?.id ?? null;
			}
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

	function mapNodesFor(d: number) {
		if (!unit) return [];
		const servedToday = new Set<number>(
			unit.stores.filter((s: any) => s.pattern[d] === 1).map((s: any) => s.store_id)
		);
		return [
			{
				id: 0,
				kind: 'depot' as const,
				x: unit.depot.x,
				y: unit.depot.y,
				label: 'DC',
				open: true
			},
			...unit.stores.map((s: any) => ({
				id: s.store_id,
				kind: 'store' as const,
				x: s.x,
				y: s.y,
				label: String(s.store_id),
				value: s.weekly_t,
				served: servedToday.has(s.store_id)
			}))
		];
	}

	/** stores not visited on this day are greyed out */
	function colorForDay(d: number) {
		const m = new Map<number, string>();
		for (const s of unit?.stores ?? []) {
			m.set(s.store_id, s.pattern[d] === 1 ? 'var(--sim)' : 'var(--border-strong)');
		}
		return m;
	}

	function routesFor(d: number) {
		const dd = unit?.routes_by_day?.[d];
		return (dd?.routes ?? []).map((r: any, i: number) => ({
			coords: r.coords,
			color: ROUTE_COLORS[i % ROUTE_COLORS.length]
		}));
	}

	const mapNodes = $derived(mapNodesFor(day));
	const mapRoutes = $derived(routesFor(day));
	const dayColors = $derived(colorForDay(day));

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

	/** delivered tonnage per weekday, split by segment */
	const loadBars = $derived(
		(unit?.routes_by_day ?? []).map((d: any) => {
			const parts = ['fresh', 'dry', 'frozen'].map((seg) => ({
				label: seg,
				color: SEG_COLORS[seg],
				value: (unit?.stores ?? []).reduce(
					(acc: number, s: any) => acc + (s.delivery_by_segment_t?.[seg]?.[d.day] ?? 0),
					0
				)
			}));
			return {
				label: d.day_name,
				value: d.delivered_t,
				parts: parts.filter((p) => p.value > 1e-9),
				note: `${d.n_vehicles} vehicle(s) · ${num(d.distance_km, 1)} km${
					d.n_vehicles ? ` · largest route ${num(d.max_route_load_t)} t` : ''
				}`
			};
		})
	);

	const freqBars = $derived(
		Object.entries(unit?.frequency_histogram ?? {}).map(([freq, count]) => ({
			label: `${freq}×/wk`,
			value: Number(count),
			color: 'var(--patt)'
		}))
	);

	const segTotals = $derived(
		['fresh', 'dry', 'frozen'].map((seg) => ({
			label: seg,
			color: SEG_COLORS[seg],
			value: (unit?.stores ?? []).reduce(
				(acc: number, s: any) => acc + (s.weekly_by_segment_t?.[seg] ?? 0),
				0
			)
		}))
	);
</script>

<div class="card">
	<div class="card-header">
		<div>
			<h1>PATT — the delivery plan</h1>
			<div class="card-sub">
				For each depot the RLRP opened, the ALNS picks one weekly delivery pattern per store and
				the vehicle tours that serve them.
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
				<MetricTile k="Waste" v={pct(unit.predicted.waste_pct)} />
				<MetricTile k="Stockout" v={pct(unit.predicted.stockout_pct)} />
			</div>
		{/if}
	</div>

	{#if unit}
		<div class="card">
			<div class="card-header">
				<div>
					<h2>Delivery pattern per store</h2>
					<div class="card-sub">
						One pattern per store, held for the whole week. The ribbon is the 6-bit pattern;
						rows are sorted by delivery frequency.
					</div>
				</div>
			</div>
			<PatternCalendar stores={unit.stores} days={DAYS} bind:mode={calendarMode} />
		</div>

		<div class="split">
			<div class="card">
				<div class="card-header">
					<div>
						<h2>Delivery frequency</h2>
						<div class="card-sub">How often stores get visited, across the depot.</div>
					</div>
				</div>
				<BarChart bars={freqBars} unit="stores" digits={0} />
			</div>

			<div class="card">
				<div class="card-header">
					<div>
						<h2>Weekly tonnage by segment</h2>
						<div class="card-sub">
							Every segment shares one delivery pattern per store (co-delivery), so these travel
							together.
						</div>
					</div>
				</div>
				<BarChart bars={segTotals} unit="t/wk" digits={2} />
			</div>
		</div>

		<div class="card">
			<div class="card-header">
				<div>
					<h2>Load per weekday</h2>
					<div class="card-sub">
						Stacked by segment. The dashed line is one vehicle's capacity Q — bars beyond it need
						more than one truck.
					</div>
				</div>
			</div>
			<BarChart
				bars={loadBars}
				reference={ctx.instance?.vehicle_capacity_t ?? null}
				referenceLabel="vehicle capacity Q"
				unit="t"
				digits={2}
			/>
			<div class="seg-legend">
				{#each ['fresh', 'dry', 'frozen'] as seg}
					<span class="li"><span class="sw" style="background:{SEG_COLORS[seg]}"></span>{seg}</span>
				{/each}
			</div>
		</div>

		<div class="card">
			<div class="card-header">
				<div>
					<h2>Routes on {DAYS[day]}</h2>
					<div class="card-sub">
						One colour per vehicle; the square is the depot. Grey stores are not served on this
						day.
					</div>
				</div>
				<div class="row">
					{#each DAYS as d, i}
						<button class:primary={day === i} onclick={() => (day = i)}>{d}</button>
					{/each}
				</div>
			</div>
			<div class="split">
				<ScatterMap
					nodes={mapNodes}
					routes={mapRoutes}
					colorBy={dayColors}
					highlight={new Set([0])}
					height={360}
					arrows={true}
					legend={[
						{ label: 'depot', shape: 'square', color: 'var(--rlrp)' },
						{ label: 'served today', shape: 'dot', color: 'var(--sim)' },
						{ label: 'not served today', shape: 'dot', color: 'var(--border-strong)' }
					]}
				/>
				<div>
					<div class="tiles" style="grid-template-columns: 1fr 1fr">
						<MetricTile k="Vehicles" v={dayData?.n_vehicles ?? 0} />
						<MetricTile k="Distance" v={`${num(dayData?.distance_km, 1)} km`} />
						<MetricTile k="Delivered" v={`${num(dayData?.delivered_t)} t`} />
						<MetricTile
							k="Stores served"
							v={unit.stores.filter((s: any) => s.pattern[day] === 1).length}
						/>
					</div>
					{#if dayData?.routes?.length}
						<div class="table-wrap" style="margin-top:10px">
							<table>
								<thead>
									<tr><th>Vehicle</th><th>Stops</th><th>Load t</th><th>Fill</th><th>km</th></tr>
								</thead>
								<tbody>
									{#each dayData.routes as r, i (r.vehicle_id)}
										<tr>
											<td>
												<span
													class="sw line"
													style="background:{ROUTE_COLORS[i % ROUTE_COLORS.length]}"
												></span>
												{r.vehicle_id}
											</td>
											<td class="mono faint">
												{r.stop_store_ids.map((s: number | null) => s ?? 'DC').join(' → ')}
											</td>
											<td>{num(r.departure_load_t)}</td>
											<td>
												{ctx.instance?.vehicle_capacity_t
													? pct((r.departure_load_t / ctx.instance.vehicle_capacity_t) * 100, 0)
													: '—'}
											</td>
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

		<div class="card">
			<div class="card-header">
				<div>
					<h2>The whole week</h2>
					<div class="card-sub">
						All six days at once — this is what the pattern decision looks like geographically.
					</div>
				</div>
			</div>
			<div class="week-grid">
				{#each DAYS as d, i}
					<div class="mini" class:active={day === i}>
						<button class="mini-head" onclick={() => (day = i)}>
							{d}
							<span class="faint">
								· {unit.routes_by_day[i].n_vehicles} veh ·
								{num(unit.routes_by_day[i].delivered_t, 1)} t
							</span>
						</button>
						<ScatterMap
							nodes={mapNodesFor(i)}
							routes={routesFor(i)}
							colorBy={colorForDay(i)}
							highlight={new Set([0])}
							height={180}
							showLabels={false}
						/>
					</div>
				{/each}
			</div>
		</div>

		<div class="card">
			<div class="card-header">
				<div>
					<h2>Predicted KPIs</h2>
					<div class="card-sub">What the PATT model expects; the SIM stage checks these.</div>
				</div>
			</div>
			<div class="tiles">
				<MetricTile k="Weekly demand" v={`${num(unit.predicted.demand_t)} t`} />
				<MetricTile k="Delivered" v={`${num(unit.predicted.delivered_t)} t`} hint="demand − stockout + waste" />
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
		grid-template-columns: repeat(auto-fit, minmax(330px, 1fr));
		gap: 16px;
		align-items: start;
	}

	.week-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
		gap: 11px;
	}

	.mini {
		border: 1px solid var(--border);
		border-radius: 9px;
		padding: 7px;
	}

	.mini.active {
		border-color: var(--patt);
		box-shadow: 0 0 0 2px color-mix(in srgb, var(--patt) 18%, transparent);
	}

	.mini-head {
		width: 100%;
		text-align: left;
		border: none;
		background: none;
		padding: 0 0 5px;
		font-size: 0.82rem;
		font-weight: 650;
		cursor: pointer;
	}

	.sw {
		display: inline-block;
		width: 9px;
		height: 9px;
		border-radius: 2px;
		margin-right: 5px;
		vertical-align: middle;
	}

	.sw.line {
		width: 12px;
		height: 3px;
	}

	.seg-legend {
		display: flex;
		gap: 12px;
		font-size: 0.74rem;
		color: var(--text-muted);
		margin-top: 8px;
	}

	.li {
		display: inline-flex;
		align-items: center;
		gap: 5px;
	}
</style>

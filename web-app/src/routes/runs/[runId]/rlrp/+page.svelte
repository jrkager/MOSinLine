<script lang="ts">
	import { onMount } from 'svelte';
	import { page } from '$app/state';
	import { createRunContext } from '$lib/run-context.svelte';
	import { num, pct } from '$lib/format';
	import BarChart from '$lib/components/BarChart.svelte';
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
	let showSpokes = $state(true);
	let showTours = $state(true);

	const DEPOT_COLORS = ['var(--rlrp)', 'var(--patt)', 'var(--sim)', 'var(--feedback)', '#db2777'];

	async function loadRound(round: number | null) {
		if (round === null) return;
		try {
			rlrp = await ctx.artifact(`rounds/${round}/rlrp.json`);
			loadError = null;
			if (scenario === null || !rlrp?.scenarios?.some((s: any) => s.scenario_id === scenario)) {
				scenario = rlrp?.scenarios?.[0]?.scenario_id ?? null;
			}
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
	const coords = $derived.by(() => {
		const m = new Map<number, [number, number]>();
		for (const n of ctx.instance?.nodes ?? []) m.set(n.id, [n.x, n.y]);
		return m;
	});

	/** colour per depot, stable across scenarios so comparison works */
	const depotColor = $derived.by(() => {
		const m = new Map<number, string>();
		(ctx.instance?.nodes ?? [])
			.filter((n: any) => n.kind === 'depot')
			.forEach((d: any, i: number) => m.set(d.id, DEPOT_COLORS[i % DEPOT_COLORS.length]));
		return m;
	});

	function buildScenarioView(sc: any) {
		if (!sc || !ctx.instance) return null;
		const storeToDepot = new Map<number, number>();
		for (const d of sc.depots) for (const st of d.stores) storeToDepot.set(st, d.depot_id);

		const openIds = new Set<number>(
			sc.depots.filter((d: any) => d.open).map((d: any) => d.depot_id)
		);

		const nodes = (ctx.instance.nodes ?? []).map((n: any) => {
			if (n.kind === 'depot') {
				const d = sc.depots.find((x: any) => x.depot_id === n.id);
				return { ...n, value: d?.size_t_per_day ?? 0, open: !!d?.open };
			}
			return { ...n, value: sc.store_demand_t_per_day?.[String(n.id)] ?? 0 };
		});

		const colorBy = new Map<number, string>();
		for (const [store, depot] of storeToDepot) {
			colorBy.set(store, depotColor.get(depot) ?? 'var(--text-muted)');
		}

		const spokes = showSpokes
			? [...storeToDepot.entries()]
					.map(([store, depot]) => {
						const a = coords.get(store);
						const b = coords.get(depot);
						if (!a || !b) return null;
						return { from: a, to: b, color: depotColor.get(depot) };
					})
					.filter((s): s is NonNullable<typeof s> => s !== null)
			: [];

		const routes = showTours
			? (sc.tours ?? []).map((t: any) => ({
					coords: t.coords,
					color: depotColor.get(t.depot_id) ?? 'var(--patt)',
					dashed: true
				}))
			: [];

		return { nodes, colorBy, spokes, routes, openIds };
	}

	const view = $derived(buildScenarioView(current));

	const legend = $derived([
		{ label: 'depot opened', shape: 'square' as const, color: 'var(--rlrp)' },
		{ label: 'depot closed', shape: 'dash' as const, color: 'var(--rlrp)' },
		{ label: 'store (size = demand)', shape: 'dot' as const, color: 'var(--text-muted)' },
		...(showTours ? [{ label: 'second-stage tour', shape: 'dash' as const, color: 'var(--patt)' }] : [])
	]);

	const utilBars = $derived(
		(current?.depots ?? [])
			.filter((d: any) => d.open)
			.map((d: any) => ({
				label: `D${Math.abs(d.depot_id)}`,
				value: d.assigned_demand_t_per_day,
				color: depotColor.get(d.depot_id) ?? 'var(--rlrp)',
				note: `built ${num(d.size_t_per_day)} t/day of a possible ${num(d.max_size_t_per_day)} · ${
					d.utilisation === null ? '—' : pct(d.utilisation * 100, 0)
				} used by the aggregate demand`
			}))
	);

	const maxDepotSize = $derived(
		Math.max(1e-9, ...(current?.depots ?? []).map((d: any) => d.size_t_per_day))
	);

	const tourBars = $derived(
		(current?.tours ?? []).map((t: any, i: number) => ({
			label: `T${i + 1}`,
			value: t.departure_load_t,
			color: depotColor.get(t.depot_id) ?? 'var(--patt)',
			note: `${t.n_stores} stores · ${num(t.distance_km, 1)} km · from D${Math.abs(t.depot_id)}`
		}))
	);
</script>

<div class="card">
	<div class="card-header">
		<div>
			<h1>RLRP — the network</h1>
			<div class="card-sub">
				Which depot candidates open, how big they are built, and which stores each one serves —
				decided separately per demand scenario.
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
			<div>
				<h2>Network for scenario {scenario}</h2>
				<div class="card-sub">
					Depot squares scale with the capacity built; store dots with their aggregate daily
					demand. Coordinates are Solomon Euclidean, not geographic.
				</div>
			</div>
			<div class="row">
				{#each rlrp.scenarios as s (s.scenario_id)}
					<button
						class:primary={scenario === s.scenario_id}
						onclick={() => (scenario = s.scenario_id)}
					>
						Scenario {s.scenario_id}
					</button>
				{/each}
			</div>
		</div>

		{#if view && current}
			<div class="row" style="margin-bottom:10px">
				<label class="toggle">
					<input type="checkbox" bind:checked={showSpokes} /> store → depot assignment
				</label>
				<label class="toggle">
					<input type="checkbox" bind:checked={showTours} /> second-stage tours
					{#if !(current.tours ?? []).length}
						<span class="faint">(none reported)</span>
					{/if}
				</label>
			</div>

			<ScatterMap
				nodes={view.nodes}
				routes={view.routes}
				spokes={view.spokes}
				colorBy={view.colorBy}
				highlight={view.openIds}
				{legend}
				height={420}
			/>

			<div class="tiles" style="margin-top:12px">
				<MetricTile
					k="Depots opened"
					v={`${current.depots.filter((d: any) => d.open).length} / ${current.depots.length}`}
				/>
				<MetricTile
					k="Capacity built"
					v={`${num(current.depots.reduce((s: number, d: any) => s + d.size_t_per_day, 0))} t/day`}
				/>
				<MetricTile
					k="Demand served"
					v={`${num(current.depots.reduce((s: number, d: any) => s + d.assigned_demand_t_per_day, 0))} t/day`}
				/>
				<MetricTile k="Tours" v={(current.tours ?? []).length || '—'} />
				<MetricTile
					k="Tour distance"
					v={(current.tours ?? []).length
						? `${num((current.tours ?? []).reduce((s: number, t: any) => s + t.distance_km, 0), 0)} km`
						: '—'}
				/>
			</div>
		{/if}
	</div>

	{#if current}
		<div class="split">
			<div class="card">
				<div class="card-header">
					<div>
						<h2>Capacity built vs demand assigned</h2>
						<div class="card-sub">
							The RLRP sizes a depot for the aggregate demand it serves — extra capacity costs
							money, so there is little headroom by design. That is exactly why PATT can fail
							the capacity check.
						</div>
					</div>
				</div>
				{#if utilBars.length}
					<BarChart
						bars={utilBars}
						reference={maxDepotSize}
						referenceLabel="capacity built (largest depot)"
						unit="t/day"
						digits={2}
					/>
				{:else}
					<div class="empty">no depot opened in this scenario</div>
				{/if}
			</div>

			<div class="card">
				<div class="card-header">
					<div>
						<h2>Depot decisions</h2>
						<div class="card-sub">Per candidate site, in this scenario.</div>
					</div>
				</div>
				<div class="table-wrap">
					<table>
						<thead>
							<tr>
								<th>Depot</th>
								<th>Status</th>
								<th>Built t/day</th>
								<th>Cap t/day</th>
								<th>Stores</th>
							</tr>
						</thead>
						<tbody>
							{#each current.depots as d (d.depot_id)}
								<tr class:highlight={d.open}>
									<td>
										<span class="sw" style="background:{depotColor.get(d.depot_id)}"></span>
										<span class="mono">D{Math.abs(d.depot_id)}</span>
									</td>
									<td>{d.open ? 'open' : 'closed'}</td>
									<td>{d.open ? num(d.size_t_per_day) : '—'}</td>
									<td class="faint">{num(d.max_size_t_per_day, 0)}</td>
									<td>{d.n_stores || '—'}</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
				{#each current.depots.filter((d: any) => d.open) as d (d.depot_id)}
					<div class="assign">
						<strong class="mono">D{Math.abs(d.depot_id)}</strong>
						<span class="faint mono">{d.stores.join(', ')}</span>
					</div>
				{/each}
			</div>
		</div>

		{#if (current.tours ?? []).length}
			<div class="card">
				<div class="card-header">
					<div>
						<h2>Second-stage tours</h2>
						<div class="card-sub">
							The RLRP is a location-<em>routing</em> problem: its second stage already builds
							tours to cost the assignment. These use aggregate daily demand, so they are not
							PATT's weekday routes — they are how the RLRP priced this network.
						</div>
					</div>
				</div>
				<BarChart bars={tourBars} reference={ctx.instance?.vehicle_capacity_t ?? null}
					referenceLabel="vehicle capacity Q" unit="t" digits={2} />
				<div class="table-wrap" style="margin-top:10px">
					<table>
						<thead>
							<tr><th>Tour</th><th>Depot</th><th>Stops</th><th>Load t</th><th>km</th></tr>
						</thead>
						<tbody>
							{#each current.tours as t, i (i)}
								<tr>
									<td class="mono">T{i + 1}</td>
									<td class="mono">D{Math.abs(t.depot_id)}</td>
									<td class="mono faint">
										{t.stops
											.map((n: number) => (n < 0 ? `D${Math.abs(n)}` : n))
											.join(' → ')}
									</td>
									<td>{num(t.departure_load_t)}</td>
									<td>{num(t.distance_km, 1)}</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			</div>
		{/if}

		<div class="card">
			<div class="card-header">
				<div>
					<h2>All scenarios side by side</h2>
					<div class="card-sub">
						The point of solving robustly: how the network changes when demand does.
					</div>
				</div>
			</div>
			<div class="strip">
				{#each rlrp.scenarios as s (s.scenario_id)}
					{@const v = buildScenarioView(s)}
					<div class="mini" class:active={scenario === s.scenario_id}>
						<button class="mini-head" onclick={() => (scenario = s.scenario_id)}>
							Scenario {s.scenario_id}
							<span class="faint">
								· {s.depots.filter((d: any) => d.open).length} open ·
								{num(s.depots.reduce((acc: number, d: any) => acc + d.size_t_per_day, 0))} t/day
							</span>
						</button>
						{#if v}
							<ScatterMap
								nodes={v.nodes}
								routes={v.routes}
								spokes={v.spokes}
								colorBy={v.colorBy}
								highlight={v.openIds}
								height={210}
								showLabels={false}
							/>
						{/if}
					</div>
				{/each}
			</div>
		</div>

		<div class="card">
			<div class="card-header"><h2>Round {rlrp.round} solve</h2>
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
	{/if}
{/if}

<style>
	.split {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(330px, 1fr));
		gap: 16px;
		align-items: start;
	}

	.assign {
		margin-top: 8px;
		font-size: 0.78rem;
		display: flex;
		gap: 8px;
		align-items: baseline;
	}

	.toggle {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		font-size: 0.8rem;
		color: var(--text-muted);
		cursor: pointer;
	}

	.sw {
		display: inline-block;
		width: 9px;
		height: 9px;
		border-radius: 2px;
		margin-right: 6px;
		vertical-align: middle;
	}

	.strip {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
		gap: 12px;
	}

	.mini {
		border: 1px solid var(--border);
		border-radius: 9px;
		padding: 8px;
	}

	.mini.active {
		border-color: var(--rlrp);
		box-shadow: 0 0 0 2px color-mix(in srgb, var(--rlrp) 18%, transparent);
	}

	.mini-head {
		width: 100%;
		text-align: left;
		border: none;
		background: none;
		padding: 0 0 6px;
		font-size: 0.82rem;
		font-weight: 650;
		cursor: pointer;
	}
</style>

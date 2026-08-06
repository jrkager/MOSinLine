<script lang="ts">
	import { onMount } from 'svelte';
	import { page } from '$app/state';
	import { createRunContext } from '$lib/run-context.svelte';
	import { num, pct } from '$lib/format';
	import MetricTile from '$lib/components/MetricTile.svelte';
	import StageNav from '$lib/components/StageNav.svelte';

	const ctx = createRunContext(
		() => page.params.runId!,
		() => page.url.searchParams.get('round')
	);

	let sim = $state<any>(null);
	let unitId = $state<string | null>(null);
	let loadError = $state<string | null>(null);

	const COLUMNS = [
		{ key: 'demand_u', label: 'Demand t/wk', fmt: (v: any) => num(v, 1) },
		{ key: 'delivered_u', label: 'Delivered t/wk', fmt: (v: any) => num(v, 1) },
		{ key: 'waste%', label: 'Waste %', fmt: (v: any) => pct(v) },
		{ key: 'stockout%', label: 'Stockout %', fmt: (v: any) => pct(v) },
		{ key: 'FW_CO2_kg/wk', label: 'FW CO₂ kg/wk', fmt: (v: any) => num(v, 0) },
		{ key: 'TR_CO2_kg/wk', label: 'TR CO₂ kg/wk', fmt: (v: any) => num(v, 0) },
		{ key: 'TR_cost/wk', label: 'TR cost/wk', fmt: (v: any) => num(v, 0) },
		{ key: 'km/wk', label: 'km/wk', fmt: (v: any) => num(v, 0) },
		{ key: 'cancel/wk', label: 'Cancel/wk', fmt: (v: any) => num(v, 2) },
		{ key: 'drop/wk', label: 'Drop/wk', fmt: (v: any) => num(v, 2) },
		{ key: 'piggy_u/wk', label: 'Piggy u/wk', fmt: (v: any) => num(v, 2) }
	];

	async function loadRound(round: number | null) {
		if (round === null) return;
		try {
			sim = await ctx.artifact(`rounds/${round}/sim.json`);
			loadError = null;
			unitId = sim?.units?.[0]?.id ?? null;
		} catch (e) {
			sim = null;
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

	const unit = $derived(sim?.units?.find((u: any) => u.id === unitId) ?? null);
	const verdict = $derived(sim?.verdict ?? null);
	const refVariant = $derived(verdict?.reference_variant ?? 2);

	/** Relative bar width for the variant comparison, per column. */
	function barWidth(rows: any[], key: string, value: any): number {
		const values = rows.map((r) => Math.abs(Number(r[key]) || 0));
		const max = Math.max(...values, 1e-9);
		return (Math.abs(Number(value) || 0) / max) * 100;
	}
</script>

<div class="card">
	<div class="card-header">
		<div>
			<h1>SIM — what actually happens</h1>
			<div class="card-sub">
				The plan is executed in a discrete-event simulation for {sim?.weeks ?? '—'} weeks. The
				comparison against PATT's own prediction is what decides whether the loop closes.
			</div>
		</div>
	</div>
	<StageNav runId={ctx.runId} round={ctx.round} />
</div>

{#if ctx.error || loadError}
	<div class="error-banner">{ctx.error ?? loadError}</div>
{:else if !sim}
	<div class="empty">{ctx.loading ? 'loading…' : 'no simulation results for this round'}</div>
{:else}
	{#if verdict}
		<div class="card">
			<div class="card-header">
				<div>
					<h2>Verdict</h2>
					<div class="card-sub">{verdict.reason}</div>
				</div>
				<span class="pill" class:accepted={verdict.accepted} class:infeasible={!verdict.accepted}>
					{verdict.accepted ? 'accepted' : 'rejected'}
				</span>
			</div>
			<div class="tiles">
				<MetricTile k="Reference" v={`Variant ${verdict.reference_variant}`} />
				<MetricTile
					k="Worst delta waste"
					v={verdict.worst_delta_waste_pp === null ? '—' : `${num(verdict.worst_delta_waste_pp)} pp`}
					hint={`tolerance ${verdict.tolerances.waste_pp} pp`}
				/>
				<MetricTile
					k="Worst delta stockout"
					v={verdict.worst_delta_stockout_pp === null
						? '—'
						: `${num(verdict.worst_delta_stockout_pp)} pp`}
					hint={`tolerance ${verdict.tolerances.stockout_pp} pp`}
				/>
				<MetricTile k="Driver" v={verdict.driver ?? 'none'} />
			</div>
			<div class="hint" style="margin-top:8px">
				The acceptance thresholds are provisional and still need a modelling decision
				(WEBTOOL.md §9.4).
			</div>
		</div>
	{/if}

	<div class="card">
		<div class="card-header">
			<div>
				<h2>Predicted vs simulated</h2>
				<div class="card-sub">
					{sim.weeks} weeks × {sim.runs_per_variant} replication(s), first {sim.warmup_weeks} weeks
					warmup. All figures are per week. Variant {refVariant} executes the plan as written — it
					should track the PATT row. Delivered follows the conservation identity
					<span class="mono">demand − stockout + waste</span>.
				</div>
			</div>
			<div class="row">
				{#each sim.units as u (u.id)}
					<button class:primary={unitId === u.id} onclick={() => (unitId = u.id)}>
						s{u.scenario_id} · D{Math.abs(u.depot_id)}
					</button>
				{/each}
			</div>
		</div>

		{#if unit}
			<div class="table-wrap">
				<table>
					<thead>
						<tr>
							<th>Run</th>
							{#each COLUMNS as c}<th>{c.label}</th>{/each}
						</tr>
					</thead>
					<tbody>
						{#each unit.rows as row (row.run)}
							<tr class:highlight={row.run === 'PATT model' || row.variant === refVariant}>
								<td>
									{row.run}
									{#if row.variant === refVariant}<span class="pill" style="margin-left:6px"
											>reference</span
										>{/if}
								</td>
								{#each COLUMNS as c}<td>{c.fmt(row[c.key])}</td>{/each}
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{/if}
	</div>

	{#if unit}
		<div class="card">
			<div class="card-header">
				<div>
					<h2>Variant comparison</h2>
					<div class="card-sub">
						Each variant is a different execution rule for the same plan — how much the outcome
						depends on operating policy, not just on the optimization.
					</div>
				</div>
			</div>
			<div class="bars">
				{#each ['waste%', 'stockout%', 'TR_CO2_kg/wk', 'km/wk'] as key}
					<div class="bar-group">
						<div class="bar-title">{COLUMNS.find((c) => c.key === key)?.label ?? key}</div>
						{#each unit.rows as row (row.run)}
							<div class="bar-row">
								<span class="bar-label">{row.run}</span>
								<span class="bar-track">
									<span
										class="bar-fill"
										class:model={row.run === 'PATT model'}
										style="width: {barWidth(unit.rows, key, row[key])}%"
									></span>
								</span>
								<span class="bar-value">{COLUMNS.find((c) => c.key === key)?.fmt(row[key])}</span>
							</div>
						{/each}
					</div>
				{/each}
			</div>
		</div>
	{/if}
{/if}

<style>
	.bars {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
		gap: 18px;
	}

	.bar-title {
		font-size: 0.76rem;
		text-transform: uppercase;
		letter-spacing: 0.06em;
		color: var(--text-faint);
		margin-bottom: 6px;
	}

	.bar-row {
		display: grid;
		grid-template-columns: 82px 1fr 74px;
		gap: 8px;
		align-items: center;
		margin-bottom: 4px;
		font-size: 0.78rem;
	}

	.bar-label {
		color: var(--text-muted);
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.bar-track {
		background: var(--surface-2);
		border-radius: 4px;
		height: 12px;
		overflow: hidden;
		border: 1px solid var(--border);
	}

	.bar-fill {
		display: block;
		height: 100%;
		background: var(--sim);
	}

	.bar-fill.model {
		background: var(--patt);
	}

	.bar-value {
		text-align: right;
		font-variant-numeric: tabular-nums;
	}
</style>

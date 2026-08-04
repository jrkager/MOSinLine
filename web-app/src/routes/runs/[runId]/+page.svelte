<script lang="ts">
	/** The primary screen: the loop, and where the algorithm currently is. */
	import { onMount } from 'svelte';
	import { page } from '$app/state';
	import { api, runArtifact } from '$lib/api';
	import { duration, num, pct } from '$lib/format';
	import LoopDiagram from '$lib/components/LoopDiagram.svelte';
	import LineChart from '$lib/components/LineChart.svelte';
	import MetricTile from '$lib/components/MetricTile.svelte';
	import RoundTimeline from '$lib/components/RoundTimeline.svelte';
	import StatusPill from '$lib/components/StatusPill.svelte';
	import type { Progress, RoundState } from '$lib/types';

	const runId = $derived(page.params.runId!);

	let progress = $state<Progress | null>(null);
	let manifest = $state<any>(null);
	let instance = $state<any>(null);
	let error = $state<string | null>(null);
	let selectedRound = $state<number | null>(null);
	let selectedEdge = $state<string | null>(null);
	let pinnedRound = $state(false);
	let logLines = $state<string[]>([]);
	let showLog = $state(false);
	let sim = $state<any>(null);
	let busy = $state(false);

	const live = $derived(progress?.status === 'running' || manifest?.status === 'queued');

	const round = $derived<RoundState | null>(
		progress?.rounds.find((r) => r.round === selectedRound) ?? null
	);

	const isCurrentRound = $derived(selectedRound === progress?.current_round);

	async function refresh() {
		try {
			const data = await api.getRun(runId);
			manifest = data.manifest;
			progress = data.progress;
			instance = data.instance;
			error = null;
			if (progress && (!pinnedRound || selectedRound === null)) {
				selectedRound = progress.current_round ?? progress.rounds.at(-1)?.round ?? null;
			}
			await loadRoundExtras();
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		}
	}

	let loadedSimKey = '';
	async function loadRoundExtras() {
		const key = `${runId}/${selectedRound}`;
		const stage = round?.stages?.sim;
		if (!selectedRound || stage?.status !== 'completed') {
			if (key !== loadedSimKey) sim = null;
			loadedSimKey = key;
			return;
		}
		if (key === loadedSimKey && sim) return;
		try {
			sim = await runArtifact(runId, `rounds/${selectedRound}/sim.json`);
			loadedSimKey = key;
		} catch {
			sim = null;
		}
	}

	async function loadLog() {
		try {
			logLines = (await api.log(runId, 'pipeline.log', 200)).lines ?? [];
		} catch {
			logLines = [];
		}
	}

	async function stop() {
		busy = true;
		try {
			await api.stopRun(runId);
			await refresh();
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			busy = false;
		}
	}

	onMount(() => {
		refresh();
		const timer = setInterval(() => {
			refresh();
			if (showLog) loadLog();
		}, 1200);
		return () => clearInterval(timer);
	});

	$effect(() => {
		if (showLog) loadLog();
	});

	// --- PATT convergence series for the currently selected round ---
	const PALETTE = ['var(--patt)', 'var(--rlrp)', 'var(--sim)', 'var(--feedback)', '#db2777'];
	const pattSeries = $derived(
		(round?.stages?.patt?.units ?? [])
			.filter((u) => u.trajectory?.length)
			.map((u, i) => ({
				label: `s${u.scenario_id} · D${Math.abs(u.depot_id)}`,
				color: PALETTE[i % PALETTE.length],
				dim: live && u.status === 'completed',
				points: (u.trajectory ?? []).map(
					(p) => [p.iteration, p.cost] as [number, number]
				)
			}))
	);

	const verdict = $derived(sim?.verdict ?? null);
</script>

<div class="card">
	<div class="card-header">
		<div>
			<h1 class="mono">{runId}</h1>
			<div class="card-sub">
				{instance?.instance_name ?? manifest?.instance_kind ?? ''}
				{#if instance}
					· {instance.n_stores} stores · {instance.n_depots} depot candidates ·
					{instance.n_scenarios} scenarios
				{/if}
			</div>
		</div>
		<div class="row">
			<StatusPill status={progress?.status ?? manifest?.status} />
			{#if live}
				<button class="danger" onclick={stop} disabled={busy}>Stop run</button>
			{/if}
		</div>
	</div>

	{#if error}
		<div class="error-banner">{error}</div>
	{/if}

	<div class="tiles">
		<MetricTile k="Elapsed" v={duration(progress?.elapsed_sec)} />
		<MetricTile
			k="Round"
			v={`${progress?.current_round ?? progress?.rounds.length ?? 0} / ${manifest?.params?.feedback?.max_rounds ?? '—'}`}
		/>
		<MetricTile k="Stage" v={progress?.current_stage?.toUpperCase() ?? '—'} />
		<MetricTile k="Mode" v={progress?.mode ?? '—'} />
		<MetricTile k="lambda" v={num(manifest?.params?.transport?.lam, 2)} />
		<MetricTile k="ALNS iters" v={manifest?.params?.patt?.max_iterations ?? '—'} />
	</div>

	{#if progress?.current_detail}
		<div class="now">
			<span class="pill running"><span class="dot"></span>now</span>
			{progress.current_detail}
		</div>
	{:else if progress?.result?.reason}
		<div class="now">
			<StatusPill status={progress.status} />
			{progress.result.reason}
		</div>
	{/if}
</div>

{#if progress}
	<div class="card">
		<div class="card-header">
			<div>
				<h2>The loop</h2>
				<div class="card-sub">
					Three stages, two handoffs, two feedback edges. Click an edge to see what crosses it.
				</div>
			</div>
			{#if !isCurrentRound && live}
				<button onclick={() => { pinnedRound = false; selectedRound = progress!.current_round; }}>
					Follow live
				</button>
			{/if}
		</div>

		<LoopDiagram {progress} {round} bind:selectedEdge />

		{#if progress.rounds.length}
			<div style="margin-top: 14px">
				<div class="nav-label">Rounds</div>
				<RoundTimeline
					rounds={progress.rounds}
					bind:selected={
						() => selectedRound,
						(v) => {
							selectedRound = v;
							pinnedRound = true;
						}
					}
				/>
			</div>
		{/if}

		{#if round?.outcome}
			<div class="outcome" class:accepted={round.outcome.kind === 'accepted'}>
				<strong>Round {round.round} outcome:</strong>
				{round.outcome.reason}
			</div>
		{/if}
	</div>

	{#if round}
		<div class="card">
			<div class="card-header"><h2>Round {round.round} — stage detail</h2></div>
			<div class="stage-grid">
				{#each progress.stages as def (def.key)}
					{@const state = round.stages[def.key]}
					<div class="stage-box" style="border-top-color: var(--{def.key})">
						<div class="row">
							<strong>{def.title}</strong>
							<StatusPill status={state.status} />
						</div>
						<div class="faint" style="font-size:0.76rem">{def.engine}</div>
						<div style="margin-top:6px">
							{#if state.headline}
								{state.headline}
							{:else}
								<span class="muted">{def.decides}</span>
							{/if}
						</div>
						{#if def.key === 'patt' && state.units.length}
							<div class="units">
								{#each state.units as unit (unit.id)}
									<span class="unit" class:done={unit.status === 'completed'} class:run={unit.status === 'running'}>
										s{unit.scenario_id}·D{Math.abs(unit.depot_id)}
										{#if unit.best_cost}<span class="faint"> {num(unit.best_cost, 0)}</span>{/if}
									</span>
								{/each}
							</div>
						{/if}
					</div>
				{/each}
			</div>
		</div>

		{#if pattSeries.length}
			<div class="card">
				<div class="card-header">
					<div>
						<h2>PATT convergence</h2>
						<div class="card-sub">ALNS best objective per (scenario, depot)</div>
					</div>
					<a class="btn" href="/runs/{runId}/patt?round={round.round}">Open PATT results</a>
				</div>
				<LineChart series={pattSeries} xLabel="iteration" yLabel="objective" />
			</div>
		{/if}

		{#if verdict}
			<div class="card">
				<div class="card-header">
					<div>
						<h2>Simulation verdict</h2>
						<div class="card-sub">
							Does the executed plan match what PATT predicted? This is what decides the
							SIM → PATT feedback edge.
						</div>
					</div>
					<a class="btn" href="/runs/{runId}/sim?round={round.round}">Open SIM results</a>
				</div>
				<div class="tiles" style="margin-bottom:12px">
					<MetricTile k="Decision" v={verdict.accepted ? 'accepted' : 'rejected'} />
					<MetricTile k="Reference" v={`Variant ${verdict.reference_variant}`} />
					<MetricTile
						k="Worst Δ waste"
						v={verdict.worst_delta_waste_pp === null
							? '—'
							: `${num(verdict.worst_delta_waste_pp)} pp`}
						hint={`tol ${verdict.tolerances.waste_pp} pp`}
					/>
					<MetricTile
						k="Worst Δ stockout"
						v={verdict.worst_delta_stockout_pp === null
							? '—'
							: `${num(verdict.worst_delta_stockout_pp)} pp`}
						hint={`tol ${verdict.tolerances.stockout_pp} pp`}
					/>
				</div>
				<div class="table-wrap">
					<table>
						<thead>
							<tr>
								<th>Unit</th>
								<th>Waste PATT</th>
								<th>Waste SIM</th>
								<th>Δ pp</th>
								<th>Stockout PATT</th>
								<th>Stockout SIM</th>
								<th>Δ pp</th>
							</tr>
						</thead>
						<tbody>
							{#each verdict.checks as c (c.id)}
								<tr>
									<td class="mono">{c.id}</td>
									<td>{pct(c.predicted_waste_pct)}</td>
									<td>{pct(c.simulated_waste_pct)}</td>
									<td class:bad={!c.waste_ok}>{num(c.delta_waste_pp)}</td>
									<td>{pct(c.predicted_stockout_pct)}</td>
									<td>{pct(c.simulated_stockout_pct)}</td>
									<td class:bad={!c.stockout_ok}>{num(c.delta_stockout_pp)}</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			</div>
		{/if}
	{/if}

	<div class="card">
		<div class="card-header">
			<h2>Pipeline log</h2>
			<button onclick={() => (showLog = !showLog)}>{showLog ? 'Hide' : 'Show'}</button>
		</div>
		{#if showLog}
			<pre class="log">{logLines.join('\n') || 'no output yet'}</pre>
		{:else}
			<div class="hint">
				{progress.log.length} events. Latest: {progress.log.at(-1)?.message ?? '—'}
			</div>
		{/if}
	</div>
{:else if !error}
	<div class="empty">waiting for the run to start…</div>
{/if}

<style>
	.now {
		margin-top: 12px;
		padding: 8px 12px;
		border-radius: 8px;
		background: var(--surface-2);
		border: 1px solid var(--border);
		display: flex;
		align-items: center;
		gap: 9px;
		font-size: 0.86rem;
	}

	.outcome {
		margin-top: 12px;
		padding: 9px 13px;
		border-radius: 8px;
		border-left: 3px solid var(--feedback);
		background: color-mix(in srgb, var(--feedback) 8%, transparent);
		font-size: 0.86rem;
	}

	.outcome.accepted {
		border-left-color: var(--ok);
		background: color-mix(in srgb, var(--ok) 8%, transparent);
	}

	.stage-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
		gap: 12px;
	}

	.stage-box {
		border: 1px solid var(--border);
		border-top-width: 3px;
		border-radius: 9px;
		padding: 11px 13px;
		font-size: 0.85rem;
	}

	.units {
		display: flex;
		flex-wrap: wrap;
		gap: 5px;
		margin-top: 9px;
	}

	.unit {
		font-size: 0.72rem;
		padding: 2px 7px;
		border-radius: 5px;
		border: 1px solid var(--border);
		background: var(--surface-2);
		font-family: var(--mono);
	}

	.unit.done {
		border-color: color-mix(in srgb, var(--ok) 45%, transparent);
	}

	.unit.run {
		border-color: var(--warn);
		background: color-mix(in srgb, var(--warn) 12%, transparent);
	}

	td.bad {
		color: var(--bad);
		font-weight: 600;
	}
</style>

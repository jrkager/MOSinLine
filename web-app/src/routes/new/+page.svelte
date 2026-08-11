<script lang="ts">
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import { api } from '$lib/api';
	import { shortTime } from '$lib/format';

	let defaults = $state<any>(null);
	let params = $state<any>(null);
	let runName = $state('');
	let advanced = $state(false);
	let submitting = $state(false);
	let error = $state<string | null>(null);
	let saved = $state<any[]>([]);
	/** name of the saved instance selected in the "built" source */
	let savedName = $state('');

	const VARIANT_CHOICES = [1, 2, 3, 4, 5, 6, 7, 8];

	async function refreshSaved() {
		try {
			saved = (await api.listInstances()).instances ?? [];
		} catch {
			saved = [];
		}
	}

	onMount(async () => {
		try {
			defaults = await api.defaults();
			params = structuredClone($state.snapshot(defaults));
			await refreshSaved();
			// coming back from the builder: preselect what was just saved
			const justSaved = page.url.searchParams.get('instance');
			if (justSaved && saved.some((s) => s.name === justSaved)) {
				params.instance.kind = 'payload';
				savedName = justSaved;
			}
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		}
	});

	/** keep payload_path in step with the selected saved instance */
	$effect(() => {
		if (!params || params.instance.kind !== 'payload') return;
		const row = saved.find((s) => s.name === savedName);
		if (row) params.instance.payload_path = row.payload_path;
	});

	async function removeSaved(name: string) {
		if (!confirm(`Delete instance "${name}"?`)) return;
		try {
			await api.deleteInstance(name);
			if (savedName === name) savedName = '';
			await refreshSaved();
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		}
	}

	function toggleVariant(v: number) {
		const list: number[] = params.sim.variants;
		params.sim.variants = list.includes(v) ? list.filter((x) => x !== v) : [...list, v].sort();
	}

	function preset(kind: 'demo' | 'thorough') {
		if (kind === 'demo') {
			params.instance.k = 5;
			params.patt.max_iterations = 15;
			params.sim.weeks = 26;
			params.sim.runs = 1;
			params.sim.variants = [2, 1];
		} else {
			params.instance.k = 10;
			params.patt.max_iterations = 500;
			params.sim.weeks = 52;
			params.sim.runs = 3;
			params.sim.variants = [2, 1, 3, 4];
		}
	}

	async function submit() {
		if (params.instance.kind === 'payload' && !savedName) {
			error = 'Pick a saved instance, or build one in the instance builder.';
			return;
		}
		submitting = true;
		error = null;
		try {
			const body: any = { params: $state.snapshot(params) };
			if (runName.trim()) body.run_name = runName.trim();
			const created = await api.createRun(body);
			await goto(`/runs/${created.run.run_id}`);
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
			submitting = false;
		}
	}
</script>

<div class="card">
	<div class="card-header">
		<div>
			<h1>New run</h1>
			<div class="card-sub">
				The defaults are a fast smoke run. Raise the ALNS iterations for results worth quoting.
			</div>
		</div>
		<div class="row">
			<button onclick={() => preset('demo')}>Demo preset</button>
			<button onclick={() => preset('thorough')}>Thorough preset</button>
		</div>
	</div>

	{#if error}
		<div class="error-banner">{error}</div>
	{/if}

	{#if !params}
		<div class="empty">loading defaults…</div>
	{:else}
		<div class="row" style="justify-content: space-between; align-items: flex-end">
			<h3>Instance</h3>
			<a class="btn" href="/builder">Instance builder →</a>
		</div>
		<div class="form-grid" style="margin: 8px 0 8px">
			<label class="field">
				Source
				<select bind:value={params.instance.kind}>
					<option value="r101">Solomon R101 (predefined)</option>
					<option value="synthetic">Synthetic (5 stores)</option>
					<option value="payload">Built / saved instance</option>
				</select>
			</label>
			{#if params.instance.kind === 'r101'}
				<label class="field">
					Stores (k)
					<input type="number" min="3" max="100" bind:value={params.instance.k} />
				</label>
				<label class="field">
					Instance index (i)
					<input type="number" min="1" max="20" bind:value={params.instance.i} />
				</label>
			{/if}
			{#if params.instance.kind === 'payload'}
				<label class="field" style="grid-column: span 2">
					Saved instance
					<select bind:value={savedName}>
						<option value="">— select —</option>
						{#each saved as row (row.name)}
							<option value={row.name}>
								{row.name} · {row.n_stores} stores, {row.n_depots} sites
							</option>
						{/each}
					</select>
				</label>
			{/if}
			<label class="field">
				Run name (optional)
				<input type="text" bind:value={runName} placeholder="auto-generated" />
			</label>
		</div>

		{#if params.instance.kind === 'payload'}
			{#if saved.length === 0}
				<div class="empty" style="margin-bottom: 12px">
					No saved instances yet. Use the
					<a href="/builder" style="text-decoration:underline">instance builder</a> to draw one, or
					load a predefined set there and edit it.
				</div>
			{:else}
				<div class="table-wrap" style="margin-bottom: 12px">
					<table>
						<thead>
							<tr>
								<th>Instance</th><th>Stores</th><th>Sites</th><th>Scenarios</th>
								<th>Saved</th><th>Built from</th><th></th>
							</tr>
						</thead>
						<tbody>
							{#each saved as row (row.name)}
								<tr class:highlight={savedName === row.name}>
									<td>
										<button class="link" onclick={() => (savedName = row.name)}>{row.name}</button>
									</td>
									<td>{row.n_stores}</td>
									<td>{row.n_depots}</td>
									<td>{row.n_scenarios}</td>
									<td class="faint">{shortTime(row.saved_at)}</td>
									<td class="faint">{row.source ?? '—'}</td>
									<td>
										<a class="btn small" href="/builder?edit={encodeURIComponent(row.name)}">Edit</a>
										<button class="danger small" onclick={() => removeSaved(row.name)}>Delete</button>
									</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
				{#each saved.filter((r) => r.name === savedName && r.warnings?.length) as row (row.name)}
					<div class="warn-banner" style="margin-bottom:12px">
						<strong>{row.name} has feasibility warnings:</strong>
						<ul style="margin:4px 0 0 16px">
							{#each row.warnings as w}<li>{w}</li>{/each}
						</ul>
					</div>
				{/each}
			{/if}
		{/if}

		<h3>Objective &amp; loop</h3>
		<div class="form-grid" style="margin: 8px 0 4px">
			<label class="field">
				λ — economic vs environmental
				<input type="number" step="0.05" min="0" max="1" bind:value={params.transport.lam} />
			</label>
			<label class="field">
				Feedback mode
				<select bind:value={params.feedback.mode}>
					<option value="full">full loop (capacity + λ)</option>
					<option value="capacity">capacity feedback only</option>
					<option value="single">single pass</option>
				</select>
			</label>
			<label class="field">
				Max rounds
				<input type="number" min="1" max="20" bind:value={params.feedback.max_rounds} />
			</label>
		</div>
		<div class="hint" style="margin-bottom: 18px">
			λ is shared by RLRP and PATT — changing it here re-derives both sides together.
		</div>

		<h3>PATT</h3>
		<div class="form-grid" style="margin: 8px 0 4px">
			<label class="field">
				ALNS iterations
				<input type="number" min="1" max="5000" bind:value={params.patt.max_iterations} />
			</label>
			<label class="field">
				Time limit (s)
				<input type="number" min="10" bind:value={params.patt.time_limit} />
			</label>
		</div>
		<div class="hint" style="margin-bottom: 18px">
			25 iterations is a smoke run; 500–2000 is paper-grade and much slower.
		</div>

		<h3>Simulation</h3>
		<div class="form-grid" style="margin: 8px 0 6px">
			<label class="field">
				Weeks
				<input type="number" min="1" max="200" bind:value={params.sim.weeks} />
			</label>
			<label class="field">
				Replications
				<input type="number" min="1" max="20" bind:value={params.sim.runs} />
			</label>
			<label class="field">
				Warmup weeks
				<input type="number" min="0" max="20" bind:value={params.sim.warmup_weeks} />
			</label>
			<label class="field">
				Reference variant
				<input type="number" min="1" max="8" bind:value={params.feedback.reference_variant} />
			</label>
		</div>
		<div class="row" style="margin-bottom: 6px">
			<span class="hint">Variants:</span>
			{#each VARIANT_CHOICES as v}
				<button
					class:primary={params.sim.variants.includes(v)}
					onclick={() => toggleVariant(v)}
					style="padding: 3px 10px"
				>
					V{v}
				</button>
			{/each}
		</div>
		<div class="hint" style="margin-bottom: 18px">
			The reference variant is the one compared against PATT's own prediction to decide whether
			the plan is accepted.
		</div>

		<button onclick={() => (advanced = !advanced)} style="margin-bottom: 12px">
			{advanced ? 'Hide' : 'Show'} advanced parameters
		</button>

		{#if advanced}
			<h3>RLRP</h3>
			<div class="form-grid" style="margin: 8px 0 18px">
				<label class="field">
					MIP gap
					<input type="number" step="0.01" min="0" max="1" bind:value={params.rlrp.gap} />
				</label>
				<label class="field">
					Time limit (s)
					<input type="number" min="10" bind:value={params.rlrp.timelimit} />
				</label>
				<label class="field">
					Threads (0 = all)
					<input type="number" min="0" bind:value={params.rlrp.n_threads} />
				</label>
				<label class="field">
					Demand aggregation
					<select bind:value={params.rlrp.demand_aggregation}>
						<option value={1}>1 — average over days</option>
						<option value={2}>2 — max over days</option>
						<option value={3}>3 — n-th largest</option>
					</select>
				</label>
			</div>

			<h3>Transport</h3>
			<div class="form-grid" style="margin: 8px 0 18px">
				<label class="field">
					c_km (€/km)
					<input type="number" step="0.01" bind:value={params.transport.c_km} />
				</label>
				<label class="field">
					c_fuel (€/L)
					<input type="number" step="0.05" bind:value={params.transport.c_fuel} />
				</label>
				<label class="field">
					eta (L/t·km)
					<input type="number" step="0.005" bind:value={params.transport.eta} />
				</label>
				<label class="field">
					theta_TR (kg CO₂/L)
					<input type="number" step="0.1" bind:value={params.transport.theta_TR} />
				</label>
				<label class="field">
					Q (t)
					<input type="number" step="0.1" bind:value={params.transport.Q} />
				</label>
				<label class="field">
					W0 (t)
					<input type="number" step="0.1" bind:value={params.transport.W0} />
				</label>
			</div>

			<h3>Feedback thresholds</h3>
			<div class="form-grid" style="margin: 8px 0 6px">
				<label class="field">
					Waste tolerance (pp)
					<input type="number" step="0.5" bind:value={params.feedback.waste_tolerance_pp} />
				</label>
				<label class="field">
					Stockout tolerance (pp)
					<input type="number" step="0.5" bind:value={params.feedback.stockout_tolerance_pp} />
				</label>
				<label class="field">
					λ factor on failure
					<input type="number" step="0.05" bind:value={params.feedback.lambda_factor} />
				</label>
				<label class="field">
					Capacity check margin
					<input type="number" step="0.05" bind:value={params.feedback.check_margin} />
				</label>
				<label class="field">
					Capacity safety
					<input type="number" step="0.05" bind:value={params.feedback.safety} />
				</label>
				<label class="field">
					Capacity step cap
					<input type="number" step="0.05" bind:value={params.feedback.step_cap} />
				</label>
			</div>
			<div class="hint" style="margin-bottom: 18px">
				The acceptance thresholds are provisional and still need a modelling decision — see
				WEBTOOL.md §9.4.
			</div>
		{/if}

		<div class="row">
			<button class="primary" onclick={submit} disabled={submitting}>
				{submitting ? 'Starting…' : 'Start run'}
			</button>
			<a class="btn" href="/">Cancel</a>
		</div>
	{/if}
</div>

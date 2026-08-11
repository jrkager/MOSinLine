<script lang="ts">
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import { api } from '$lib/api';
	import { num } from '$lib/format';
	import InstanceCanvas from '$lib/components/InstanceCanvas.svelte';
	import MetricTile from '$lib/components/MetricTile.svelte';

	const DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
	const SEGMENTS = ['dry', 'fresh', 'frozen'];

	let doc = $state<any>(null);
	let name = $state('');
	let mode = $state<'select' | 'store' | 'depot'>('select');
	let selected = $state<{ kind: 'store' | 'depot'; id: number } | null>(null);
	let summary = $state<any>(null);
	let problems = $state<string[]>([]);
	let error = $state<string | null>(null);
	let saving = $state(false);
	let presets = $state<any[]>([]);
	let presetKind = $state('r101');
	let presetStores = $state(10);
	let presetIndex = $state(1);
	let loadingPreset = $state(false);
	let tab = $state<'nodes' | 'demand' | 'scenarios'>('nodes');

	/** fixed editing window; Solomon coordinates live inside 0..100 */
	const world = { minX: 0, minY: 0, maxX: 100, maxY: 100 };

	async function loadBlank() {
		const data = await api.newInstance();
		doc = data.builder;
		name = data.suggested_name;
	}

	onMount(async () => {
		try {
			presets = (await api.listInstances()).presets ?? [];
			const editName = page.url.searchParams.get('edit');
			if (editName) {
				const data = await api.getInstance(editName);
				doc = data.builder;
				name = `${editName} (copy)`;
			} else {
				await loadBlank();
			}
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		}
	});

	/** recompute totals and validation on every edit */
	let summaryTimer: ReturnType<typeof setTimeout> | null = null;
	$effect(() => {
		if (!doc) return;
		const snapshot = $state.snapshot(doc);
		if (summaryTimer) clearTimeout(summaryTimer);
		summaryTimer = setTimeout(async () => {
			try {
				const data = await api.instanceSummary(snapshot);
				summary = data.summary;
				problems = data.problems ?? [];
			} catch {
				/* leave the previous summary in place */
			}
		}, 180);
	});

	// ------------------------------------------------------------------ edit --
	function nextStoreId(): number {
		return Math.max(0, ...doc.stores.map((s: any) => s.store_id)) + 1;
	}
	function nextDepotId(): number {
		return Math.min(0, ...doc.depots.map((d: any) => d.depot_id)) - 1;
	}

	function addNode(kind: 'store' | 'depot', x: number, y: number) {
		if (kind === 'store') {
			const id = nextStoreId();
			doc.stores = [...doc.stores, { store_id: id, x, y, demand_t_per_day: 2.0 }];
			selected = { kind: 'store', id };
		} else {
			const id = nextDepotId();
			doc.depots = [
				...doc.depots,
				{ depot_id: id, x, y, fixed_cost: 5600, marginal_cost: 35, max_size: 30 }
			];
			selected = { kind: 'depot', id };
		}
	}

	function moveNode(kind: 'store' | 'depot', id: number, x: number, y: number) {
		if (kind === 'store') {
			const s = doc.stores.find((n: any) => n.store_id === id);
			if (s) { s.x = x; s.y = y; }
		} else {
			const d = doc.depots.find((n: any) => n.depot_id === id);
			if (d) { d.x = x; d.y = y; }
		}
	}

	function removeSelected() {
		if (!selected) return;
		if (selected.kind === 'store') {
			doc.stores = doc.stores.filter((s: any) => s.store_id !== selected!.id);
		} else {
			doc.depots = doc.depots.filter((d: any) => d.depot_id !== selected!.id);
		}
		selected = null;
	}

	function clearAll() {
		if (!confirm('Remove all stores and depot candidates?')) return;
		doc.stores = [];
		doc.depots = [];
		selected = null;
	}

	async function loadPreset() {
		loadingPreset = true;
		error = null;
		try {
			const data = await api.instancePreset(presetKind, presetStores, presetIndex);
			doc = data.builder;
			if (!name.trim() || name.includes('-')) name = data.suggested_name;
			selected = null;
			mode = 'select';
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			loadingPreset = false;
		}
	}

	function addScenario() {
		const id = Math.max(0, ...doc.scenarios.map((s: any) => s.scenario_id)) + 1;
		doc.scenarios = [
			...doc.scenarios,
			{ scenario_id: id, name: `scenario ${id}`, factor: 1.0 }
		];
	}

	function removeScenario(id: number) {
		doc.scenarios = doc.scenarios.filter((s: any) => s.scenario_id !== id);
	}

	/** keep the shares summing to 1 by absorbing the change into the others */
	function setShare(seg: string, value: number) {
		const v = Math.max(0, Math.min(1, value));
		const others = SEGMENTS.filter((s) => s !== seg);
		const otherTotal = others.reduce((acc, s) => acc + Number(doc.segment_shares[s] ?? 0), 0);
		const next: Record<string, number> = { [seg]: v };
		const remaining = 1 - v;
		for (const s of others) {
			next[s] =
				otherTotal > 1e-9
					? (Number(doc.segment_shares[s] ?? 0) / otherTotal) * remaining
					: remaining / others.length;
		}
		doc.segment_shares = Object.fromEntries(
			SEGMENTS.map((s) => [s, Math.round(next[s] * 1e6) / 1e6])
		);
		// put rounding drift into the largest share
		const total = SEGMENTS.reduce((acc, s) => acc + doc.segment_shares[s], 0);
		const biggest = SEGMENTS.reduce((a, b) =>
			doc.segment_shares[a] >= doc.segment_shares[b] ? a : b
		);
		doc.segment_shares[biggest] = Math.round((doc.segment_shares[biggest] + (1 - total)) * 1e6) / 1e6;
	}

	async function saveAndBack() {
		saving = true;
		error = null;
		try {
			const saved = await api.saveInstance($state.snapshot(doc), name.trim() || undefined);
			await goto(`/new?instance=${encodeURIComponent(saved.instance.name)}`);
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
			saving = false;
		}
	}

	const selectedStore = $derived(
		selected?.kind === 'store'
			? doc?.stores.find((s: any) => s.store_id === selected!.id)
			: null
	);
	const selectedDepot = $derived(
		selected?.kind === 'depot'
			? doc?.depots.find((d: any) => d.depot_id === selected!.id)
			: null
	);

	const weeklyPerStore = $derived.by(() => {
		const mult: number[] = doc?.weekday_multipliers ?? [];
		return (d: number) => mult.reduce((acc, m) => acc + d * m, 0);
	});
</script>

<div class="card">
	<div class="card-header">
		<div>
			<h1>Instance builder</h1>
			<div class="card-sub">
				Place stores and depot candidates, or load a predefined set and edit it. Saving makes the
				instance selectable on the New run page.
			</div>
		</div>
		<div class="row">
			<a class="btn" href="/new">Cancel</a>
			<button class="primary" onclick={saveAndBack} disabled={saving || !!problems.length || !doc}>
				{saving ? 'Saving…' : 'Save and back'}
			</button>
		</div>
	</div>

	{#if error}
		<div class="error-banner">{error}</div>
	{/if}

	{#if !doc}
		<div class="empty">loading…</div>
	{:else}
		<div class="form-grid">
			<label class="field" style="grid-column: span 2">
				Instance name
				<input type="text" bind:value={name} placeholder="date and time" />
			</label>
			<label class="field">
				Start from
				<select bind:value={presetKind}>
					{#each presets as p (p.id)}
						<option value={p.id}>{p.label}</option>
					{/each}
				</select>
			</label>
			{#if presetKind === 'r101'}
				<label class="field">
					Stores
					<input type="number" min="1" max="100" bind:value={presetStores} />
				</label>
				<label class="field">
					Index
					<input type="number" min="1" max="20" bind:value={presetIndex} />
				</label>
			{/if}
			<label class="field">
				&nbsp;
				<button onclick={loadPreset} disabled={loadingPreset}>
					{loadingPreset ? 'Loading…' : 'Load preset'}
				</button>
			</label>
		</div>
		<div class="hint">
			Loading a preset replaces the current canvas. Its per-store demand noise is averaged into one
			mean daily value per store, which is what this editor works with.
		</div>
	{/if}
</div>

{#if doc}
	{#if problems.length}
		<div class="error-banner">
			<strong>Cannot save yet:</strong>
			<ul style="margin:4px 0 0 16px">
				{#each problems as p}<li>{p}</li>{/each}
			</ul>
		</div>
	{/if}
	{#if summary?.warnings?.length}
		<div class="warn-banner">
			<strong>Feasibility warnings:</strong>
			<ul style="margin:4px 0 0 16px">
				{#each summary.warnings as w}<li>{w}</li>{/each}
			</ul>
		</div>
	{/if}

	<div class="editor">
		<div class="card">
			<div class="card-header">
				<div class="row">
					<button class:primary={mode === 'select'} onclick={() => (mode = 'select')}>
						Select / move
					</button>
					<button class:primary={mode === 'store'} onclick={() => (mode = 'store')}>
						Add store
					</button>
					<button class:primary={mode === 'depot'} onclick={() => (mode = 'depot')}>
						Add depot
					</button>
				</div>
				<div class="row">
					<button onclick={removeSelected} disabled={!selected} class="danger">
						Delete selected
					</button>
					<button onclick={clearAll} class="danger">Clear</button>
				</div>
			</div>
			<InstanceCanvas
				stores={doc.stores}
				depots={doc.depots}
				bind:selected
				{mode}
				{world}
				onadd={addNode}
				onmove={moveNode}
			/>
			<div class="legend-row">
				<span class="li"><span class="sw dot"></span>store (size = demand)</span>
				<span class="li"><span class="sw sq"></span>depot candidate (size = max capacity)</span>
				<span class="faint">
					{mode === 'select' ? 'Drag a node to move it.' : 'Click the canvas to place.'}
				</span>
			</div>
		</div>

		<div class="side">
			<div class="card">
				<div class="tabs">
					<button class="tab" class:active={tab === 'nodes'} onclick={() => (tab = 'nodes')}>
						Selection
					</button>
					<button class="tab" class:active={tab === 'demand'} onclick={() => (tab = 'demand')}>
						Demand shape
					</button>
					<button class="tab" class:active={tab === 'scenarios'} onclick={() => (tab = 'scenarios')}>
						Scenarios
					</button>
				</div>

				{#if tab === 'nodes'}
					{#if selectedStore}
						<h3 style="margin-top:12px">Store {selectedStore.store_id}</h3>
						<div class="form-grid" style="margin-top:8px">
							<label class="field">
								x
								<input type="number" step="0.1" bind:value={selectedStore.x} />
							</label>
							<label class="field">
								y
								<input type="number" step="0.1" bind:value={selectedStore.y} />
							</label>
							<label class="field" style="grid-column: span 2">
								Mean daily demand (t)
								<input
									type="number"
									step="0.1"
									min="0"
									bind:value={selectedStore.demand_t_per_day}
								/>
							</label>
						</div>
						<div class="hint">
							≈ {num(weeklyPerStore(selectedStore.demand_t_per_day))} t/week after the weekday
							shape. Split across segments by the shares on the next tab.
						</div>
					{:else if selectedDepot}
						<h3 style="margin-top:12px">Depot candidate D{Math.abs(selectedDepot.depot_id)}</h3>
						<div class="form-grid" style="margin-top:8px">
							<label class="field">
								x
								<input type="number" step="0.1" bind:value={selectedDepot.x} />
							</label>
							<label class="field">
								y
								<input type="number" step="0.1" bind:value={selectedDepot.y} />
							</label>
							<label class="field">
								Max capacity (t/day)
								<input type="number" step="1" min="0.1" bind:value={selectedDepot.max_size} />
							</label>
							<label class="field">
								Fixed cost
								<input type="number" step="100" min="0" bind:value={selectedDepot.fixed_cost} />
							</label>
							<label class="field">
								Marginal cost (per t/day)
								<input type="number" step="1" min="0" bind:value={selectedDepot.marginal_cost} />
							</label>
						</div>
						<div class="hint">
							Max capacity is the cap the RLRP may build up to; it decides the actual size.
						</div>
					{:else}
						<div class="empty" style="margin-top:12px">
							Select a node on the canvas to edit it.
						</div>
					{/if}
				{:else if tab === 'demand'}
					<h3 style="margin-top:12px">Segment shares</h3>
					<div class="hint">Applied to every store; always kept summing to 1.</div>
					<div class="form-grid" style="margin-top:8px">
						{#each SEGMENTS as seg}
							<label class="field">
								{seg}
								<input
									type="number"
									step="0.01"
									min="0"
									max="1"
									value={doc.segment_shares[seg]}
									oninput={(e) => setShare(seg, Number((e.target as HTMLInputElement).value))}
								/>
							</label>
						{/each}
					</div>

					<h3 style="margin-top:16px">Weekday shape</h3>
					<div class="hint">
						Multipliers on the mean daily demand, Mon–Sat. A six-day week; there is no Sunday.
					</div>
					<div class="form-grid" style="margin-top:8px; grid-template-columns: repeat(3, 1fr)">
						{#each DAYS as d, i}
							<label class="field">
								{d}
								<input
									type="number"
									step="0.05"
									min="0"
									bind:value={doc.weekday_multipliers[i]}
								/>
							</label>
						{/each}
					</div>

					<h3 style="margin-top:16px">Other</h3>
					<div class="form-grid" style="margin-top:8px">
						<label class="field">
							2nd-stage penalty factor
							<input
								type="number"
								step="0.1"
								min="1"
								bind:value={doc.second_stage_penalty_factor}
							/>
						</label>
					</div>
				{:else}
					<h3 style="margin-top:12px">Demand scenarios</h3>
					<div class="hint">
						Each scenario scales all stores by its factor. The RLRP opens depots that hold up
						across all of them.
					</div>
					<div class="table-wrap" style="margin-top:8px">
						<table>
							<thead>
								<tr><th>Id</th><th>Name</th><th>Factor</th><th></th></tr>
							</thead>
							<tbody>
								{#each doc.scenarios as sc (sc.scenario_id)}
									<tr>
										<td class="mono">{sc.scenario_id}</td>
										<td><input type="text" bind:value={sc.name} style="width:100%" /></td>
										<td>
											<input
												type="number"
												step="0.05"
												min="0.05"
												bind:value={sc.factor}
												style="width:78px"
											/>
										</td>
										<td>
											<button
												class="danger"
												onclick={() => removeScenario(sc.scenario_id)}
												disabled={doc.scenarios.length <= 1}
											>
												×
											</button>
										</td>
									</tr>
								{/each}
							</tbody>
						</table>
					</div>
					<button onclick={addScenario} style="margin-top:8px">Add scenario</button>
				{/if}
			</div>

			<div class="card">
				<div class="card-header"><h2>Instance summary</h2></div>
				{#if summary}
					<div class="tiles" style="grid-template-columns: 1fr 1fr">
						<MetricTile k="Stores" v={summary.n_stores} />
						<MetricTile k="Depot sites" v={summary.n_depots} />
						<MetricTile k="Scenarios" v={summary.n_scenarios} />
						<MetricTile
							k="Total site cap"
							v={`${num(summary.total_max_capacity_t_per_day, 0)} t/day`}
						/>
					</div>
					<div class="table-wrap" style="margin-top:10px">
						<table>
							<thead>
								<tr><th>Scenario</th><th>t/week</th><th>t/day mean</th></tr>
							</thead>
							<tbody>
								{#each summary.per_scenario as sc (sc.scenario_id)}
									<tr>
										<td>{sc.name ?? sc.scenario_id}</td>
										<td>{num(sc.weekly_demand_t)}</td>
										<td>{num(sc.daily_mean_t)}</td>
									</tr>
								{/each}
							</tbody>
						</table>
					</div>
				{:else}
					<div class="empty">—</div>
				{/if}
			</div>
		</div>
	</div>
{/if}

<style>
	.editor {
		display: grid;
		grid-template-columns: minmax(340px, 1.35fr) minmax(300px, 1fr);
		gap: 16px;
		align-items: start;
	}

	@media (max-width: 1000px) {
		.editor {
			grid-template-columns: 1fr;
		}
	}

	.side {
		display: flex;
		flex-direction: column;
		gap: 16px;
	}

	.legend-row {
		display: flex;
		gap: 14px;
		align-items: center;
		flex-wrap: wrap;
		margin-top: 8px;
		font-size: 0.76rem;
		color: var(--text-muted);
	}

	.li {
		display: inline-flex;
		align-items: center;
		gap: 5px;
	}

	.sw {
		display: inline-block;
	}

	.sw.dot {
		width: 9px;
		height: 9px;
		border-radius: 50%;
		background: var(--sim);
	}

	.sw.sq {
		width: 10px;
		height: 10px;
		border-radius: 2px;
		background: var(--rlrp);
	}

	.warn-banner {
		border: 1px solid color-mix(in srgb, var(--feedback) 55%, transparent);
		background: color-mix(in srgb, var(--feedback) 10%, transparent);
		color: var(--text);
		padding: 9px 13px;
		border-radius: 8px;
		font-size: 0.84rem;
	}

	td input {
		font-size: 0.8rem;
		padding: 3px 6px;
	}
</style>

<script lang="ts">
	import { onMount } from 'svelte';
	import { api } from '$lib/api';
	import { duration, shortTime } from '$lib/format';
	import StatusPill from '$lib/components/StatusPill.svelte';
	import type { RunManifest } from '$lib/types';

	let runs = $state<RunManifest[]>([]);
	let loading = $state(true);
	let error = $state<string | null>(null);

	async function refresh() {
		try {
			runs = (await api.listRuns()).runs ?? [];
			error = null;
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			loading = false;
		}
	}

	async function remove(runId: string) {
		if (!confirm(`Delete run ${runId}? This removes its artifacts from disk.`)) return;
		try {
			await api.deleteRun(runId);
			await refresh();
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		}
	}

	onMount(() => {
		refresh();
		const timer = setInterval(refresh, 3000);
		return () => clearInterval(timer);
	});
</script>

<div class="card">
	<div class="card-header">
		<div>
			<h1>Integrated pipeline runs</h1>
			<div class="card-sub">
				Each run traverses RLRP → PATT → SIM one or more times, feeding results back until the
				plan holds up in simulation.
			</div>
		</div>
		<a class="btn" href="/new">New run</a>
	</div>

	{#if error}
		<div class="error-banner">{error}</div>
	{:else if loading}
		<div class="empty">loading…</div>
	{:else if runs.length === 0}
		<div class="empty">
			No runs yet. <a href="/new" style="text-decoration:underline">Start one</a> — a 5-store smoke
			run finishes in about 30 seconds.
		</div>
	{:else}
		<div class="table-wrap">
			<table>
				<thead>
					<tr>
						<th>Run</th>
						<th>Status</th>
						<th>Rounds</th>
						<th>Stage</th>
						<th>Elapsed</th>
						<th>Mode</th>
						<th>Created</th>
						<th></th>
					</tr>
				</thead>
				<tbody>
					{#each runs as run (run.run_id)}
						<tr>
							<td><a href="/runs/{run.run_id}" class="mono link">{run.run_id}</a></td>
							<td><StatusPill status={run.live_status ?? run.status} /></td>
							<td>{run.current_round ?? run.rounds ?? '—'}</td>
							<td class="muted">{run.current_stage ?? '—'}</td>
							<td>{duration(run.elapsed_sec)}</td>
							<td class="muted">{run.mode ?? '—'}</td>
							<td class="muted">{shortTime(run.created_at)}</td>
							<td>
								<button class="danger" onclick={() => remove(run.run_id)}>Delete</button>
							</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	{/if}
</div>

<style>
	.link {
		text-decoration: underline;
		text-underline-offset: 2px;
	}
</style>

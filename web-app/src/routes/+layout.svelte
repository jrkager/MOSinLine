<script lang="ts">
	import './layout.css';
	import { onMount } from 'svelte';
	import { page } from '$app/state';
	import { goto } from '$app/navigation';
	import { api } from '$lib/api';
	import StatusPill from '$lib/components/StatusPill.svelte';
	import type { RunManifest } from '$lib/types';

	let { children } = $props();

	let runs = $state<RunManifest[]>([]);
	let backendError = $state<string | null>(null);

	const currentRunId = $derived(page.params.runId ?? null);

	async function refresh() {
		try {
			const data = await api.listRuns();
			runs = data.runs ?? [];
			backendError = null;
		} catch (error) {
			backendError = error instanceof Error ? error.message : String(error);
		}
	}

	onMount(() => {
		refresh();
		const timer = setInterval(refresh, 3000);
		return () => clearInterval(timer);
	});
</script>

<div class="shell">
	<aside class="sidebar">
		<div class="brand">
			MOSinLine
			<small>RLRP · PATT · SIM integration demo</small>
		</div>

		<div class="nav-group">
			<div class="nav-label">Pipeline</div>
			<a class="nav-item" class:active={page.url.pathname === '/'} href="/">Overview</a>
			<a class="nav-item" class:active={page.url.pathname === '/new'} href="/new">New run</a>
			<a class="nav-item" class:active={page.url.pathname === '/builder'} href="/builder">
				Instance builder
			</a>
			<a class="nav-item" class:active={page.url.pathname === '/docs'} href="/docs">How it works</a>
		</div>

		<div class="nav-group">
			<div class="nav-label">Runs ({runs.length})</div>
			{#if backendError}
				<div class="hint">backend unreachable</div>
			{:else if runs.length === 0}
				<div class="hint">no runs yet</div>
			{:else}
				{#each runs as run (run.run_id)}
					<a
						class="nav-item"
						class:active={currentRunId === run.run_id}
						href="/runs/{run.run_id}"
					>
						<span class="truncate">{run.run_id}</span>
						<StatusPill status={run.live_status ?? run.status} />
					</a>
				{/each}
			{/if}
		</div>

		<div class="spacer"></div>
		<div class="hint">
			Demo tool for the project report. The loop view shows which stage the algorithm is in and
			which feedback edge it took.
		</div>
	</aside>

	<main class="main">
		{#if backendError}
			<div class="error-banner">
				Backend not reachable: {backendError}. Start it with
				<code class="mono">uvicorn webtool.server.app:app --reload</code>.
			</div>
		{/if}
		{@render children()}
	</main>
</div>

<script lang="ts">
	import { page } from '$app/state';

	let { runId, round }: { runId: string; round: number | null } = $props();

	const suffix = $derived(round ? `?round=${round}` : '');
	const here = $derived(page.url.pathname);

	const links = $derived([
		{ href: `/runs/${runId}`, label: 'Loop' },
		{ href: `/runs/${runId}/rlrp`, label: 'RLRP' },
		{ href: `/runs/${runId}/patt`, label: 'PATT' },
		{ href: `/runs/${runId}/sim`, label: 'SIM' }
	]);
</script>

<div class="tabs">
	{#each links as link (link.href)}
		<a class="tab" class:active={here === link.href} href="{link.href}{suffix}">{link.label}</a>
	{/each}
</div>

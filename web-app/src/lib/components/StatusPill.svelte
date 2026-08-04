<script lang="ts">
	let { status, label }: { status?: string | null; label?: string } = $props();
	const cls = $derived(String(status ?? 'pending').toLowerCase());
	const live = $derived(cls === 'running' || cls === 'queued');
</script>

<span class="pill {cls}">
	{#if live}<span class="dot"></span>{/if}
	{label ?? status ?? 'pending'}
</span>

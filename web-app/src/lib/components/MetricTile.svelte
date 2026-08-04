<script lang="ts">
	let { k, v, hint }: { k: string; v: string | number; hint?: string } = $props();
</script>

<div class="tile">
	<div class="k">{k}</div>
	<div class="v">{v}</div>
	{#if hint}<div class="hint">{hint}</div>{/if}
</div>

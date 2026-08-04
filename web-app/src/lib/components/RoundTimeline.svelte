<script lang="ts">
	import type { RoundState } from '$lib/types';

	let {
		rounds,
		selected = $bindable(null)
	}: { rounds: RoundState[]; selected?: number | null } = $props();

	const OUTCOME_LABEL: Record<string, string> = {
		accepted: 'accepted',
		feedback_capacity: 'capacity → RLRP',
		feedback_lambda: 'λ ↓ → PATT',
		infeasible: 'infeasible',
		aborted: 'aborted'
	};

	function tone(kind?: string): string {
		if (!kind) return '';
		if (kind === 'accepted') return 'accepted';
		if (kind.startsWith('feedback')) return 'feedback';
		return 'infeasible';
	}
</script>

<div class="timeline">
	{#each rounds as r (r.round)}
		<button
			class="round"
			class:active={selected === r.round}
			onclick={() => (selected = r.round)}
			title={r.outcome?.reason ?? 'in progress'}
		>
			<span class="n">Round {r.round}</span>
			{#if r.outcome}
				<span class="pill {tone(r.outcome.kind)}">{OUTCOME_LABEL[r.outcome.kind] ?? r.outcome.kind}</span>
			{:else}
				<span class="pill running"><span class="dot"></span>running</span>
			{/if}
		</button>
		{#if r !== rounds[rounds.length - 1]}
			<span class="connector" aria-hidden="true">→</span>
		{/if}
	{/each}
</div>

<style>
	.timeline {
		display: flex;
		align-items: center;
		gap: 6px;
		flex-wrap: wrap;
	}

	.round {
		display: flex;
		flex-direction: column;
		align-items: flex-start;
		gap: 4px;
		padding: 7px 12px;
		border-radius: 9px;
		border: 1px solid var(--border);
		background: var(--surface);
		cursor: pointer;
		text-align: left;
	}

	.round.active {
		border-color: var(--rlrp);
		box-shadow: 0 0 0 2px color-mix(in srgb, var(--rlrp) 22%, transparent);
	}

	.n {
		font-weight: 650;
		font-size: 0.82rem;
	}

	.connector {
		color: var(--text-faint);
	}
</style>

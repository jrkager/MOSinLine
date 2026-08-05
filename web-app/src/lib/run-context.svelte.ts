import { api, runArtifact } from '$lib/api';
import type { Progress } from '$lib/types';

/**
 * Shared loader for the stage detail pages: resolves the run, the selected
 * round (from ?round=, defaulting to the last one that has results) and the
 * instance artifact.
 */
export function createRunContext(getRunId: () => string, getRoundParam: () => string | null) {
	let progress = $state<Progress | null>(null);
	let manifest = $state<any>(null);
	let instance = $state<any>(null);
	let error = $state<string | null>(null);
	let loading = $state(true);

	const runId = $derived(getRunId());

	const round = $derived.by(() => {
		const raw = getRoundParam();
		if (raw && Number.isFinite(Number(raw))) return Number(raw);
		const rounds = progress?.rounds ?? [];
		// prefer the last round that actually produced results
		for (let i = rounds.length - 1; i >= 0; i--) {
			if (rounds[i].stages.patt.status === 'completed') return rounds[i].round;
		}
		return rounds.at(-1)?.round ?? null;
	});

	async function load() {
		try {
			const data = await api.getRun(getRunId());
			manifest = data.manifest;
			progress = data.progress;
			instance = data.instance;
			error = null;
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			loading = false;
		}
	}

	return {
		get progress() {
			return progress;
		},
		get manifest() {
			return manifest;
		},
		get instance() {
			return instance;
		},
		get error() {
			return error;
		},
		get loading() {
			return loading;
		},
		get runId() {
			return runId;
		},
		get round() {
			return round;
		},
		load,
		artifact: <T = any>(relative: string) => runArtifact<T>(getRunId(), relative)
	};
}

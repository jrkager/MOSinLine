const base = (import.meta.env.PUBLIC_API_BASE_URL ?? '').replace(/\/$/, '');

export function apiUrl(path: string): string {
	return `${base}${path}`;
}

async function json<T>(path: string, init?: RequestInit): Promise<T> {
	const response = await fetch(apiUrl(path), {
		...init,
		headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) }
	});
	if (!response.ok) {
		let detail = `${response.status} ${response.statusText}`;
		try {
			const body = await response.json();
			if (body?.detail) detail = String(body.detail);
		} catch {
			/* keep the status line */
		}
		throw new Error(detail);
	}
	return (await response.json()) as T;
}

export const api = {
	health: () => json<any>('/api/health'),
	defaults: () => json<any>('/api/defaults'),
	listRuns: () => json<any>('/api/runs'),
	getRun: (id: string) => json<any>(`/api/runs/${encodeURIComponent(id)}`),
	createRun: (body: unknown) =>
		json<any>('/api/runs', { method: 'POST', body: JSON.stringify(body) }),
	stopRun: (id: string) =>
		json<any>(`/api/runs/${encodeURIComponent(id)}/stop`, { method: 'POST' }),
	deleteRun: (id: string) =>
		json<any>(`/api/runs/${encodeURIComponent(id)}`, { method: 'DELETE' }),
	log: (id: string, name = 'pipeline.log', tail = 300) =>
		json<any>(`/api/runs/${encodeURIComponent(id)}/log?name=${encodeURIComponent(name)}&tail=${tail}`),
	/** Any file from the artifact tree, addressed by its contract path. */
	artifact: <T = any>(path: string) =>
		json<T>(`/api/artifact?path=${encodeURIComponent(path)}`)
};

export function runArtifact<T = any>(runId: string, relative: string) {
	return api.artifact<T>(`runs/${runId}/frontend/${relative}`);
}

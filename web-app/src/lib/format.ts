export function num(value: unknown, digits = 2): string {
	if (value === null || value === undefined || value === '') return '—';
	const n = Number(value);
	if (!Number.isFinite(n)) return '—';
	return n.toLocaleString(undefined, {
		minimumFractionDigits: digits,
		maximumFractionDigits: digits
	});
}

export function pct(value: unknown, digits = 2): string {
	if (value === null || value === undefined) return '—';
	const n = Number(value);
	if (!Number.isFinite(n)) return '—';
	return `${n.toFixed(digits)}%`;
}

export function duration(seconds: unknown): string {
	const s = Number(seconds);
	if (!Number.isFinite(s) || s < 0) return '—';
	const h = Math.floor(s / 3600);
	const m = Math.floor((s % 3600) / 60);
	const sec = Math.floor(s % 60);
	if (h) return `${h}h ${String(m).padStart(2, '0')}m`;
	if (m) return `${m}m ${String(sec).padStart(2, '0')}s`;
	return `${sec}s`;
}

export function shortTime(iso: unknown): string {
	if (!iso) return '—';
	const d = new Date(String(iso));
	if (Number.isNaN(d.getTime())) return '—';
	return d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

export function depotLabel(id: unknown): string {
	return `D${Math.abs(Number(id))}`;
}

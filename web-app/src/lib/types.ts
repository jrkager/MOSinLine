export type StageKey = 'rlrp' | 'patt' | 'sim';

export type StageDef = {
	key: StageKey;
	title: string;
	subtitle: string;
	decides: string;
	engine: string;
};

export type EdgeDef = {
	id: string;
	source: StageKey;
	target: StageKey;
	kind: 'handoff' | 'feedback';
	label: string;
	detail: string;
};

export type StageState = {
	status: 'pending' | 'running' | 'completed' | 'reused' | 'blocked' | 'failed';
	started_at: string | null;
	finished_at: string | null;
	headline: string | null;
	reused: boolean;
	units: UnitState[];
};

export type UnitState = {
	id: string;
	scenario_id: number;
	depot_id: number;
	n_stores: number;
	status: string;
	iterations?: number;
	best_cost?: number;
	feasible?: boolean;
	trajectory?: { iteration: number; cost: number; elapsed_sec: number }[];
};

export type RoundOutcome = {
	kind: 'accepted' | 'feedback_capacity' | 'feedback_lambda' | 'infeasible' | 'aborted';
	reason: string;
	detail: Record<string, any>;
	edge_id: string | null;
};

export type RoundState = {
	round: number;
	status: string;
	started_at: string | null;
	finished_at: string | null;
	stages: Record<StageKey, StageState>;
	outcome: RoundOutcome | null;
};

export type Progress = {
	schema_version: number;
	run_id: string;
	instance_name: string;
	status: 'pending' | 'running' | 'completed' | 'failed' | 'stopped' | 'infeasible';
	mode: string;
	started_at: string;
	updated_at: string;
	finished_at: string | null;
	elapsed_sec: number;
	current_round: number | null;
	current_stage: StageKey | null;
	current_detail: string | null;
	current_edge: string | null;
	stages: StageDef[];
	edges: EdgeDef[];
	rounds: RoundState[];
	result: { status: string; reason?: string } | null;
	log: { t: string; elapsed_sec: number; message: string }[];
};

export type RunManifest = {
	run_id: string;
	status: string;
	created_at?: string;
	finished_at?: string;
	reason?: string;
	rounds?: number;
	instance_kind?: string;
	mode?: string;
	params?: any;
	live_status?: string;
	current_round?: number | null;
	current_stage?: string | null;
	elapsed_sec?: number;
};

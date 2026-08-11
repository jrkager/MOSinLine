<script lang="ts">
	/**
	 * Annotated protocol diagram for the documentation page: the three modules
	 * as columns, with what each one consumes and produces, and the named
	 * transform that sits between them.
	 *
	 * HTML/CSS rather than SVG on purpose - the boxes are mostly wrapped text,
	 * which SVG makes needlessly painful.
	 */
	let { selected = $bindable<string | null>(null) }: { selected?: string | null } = $props();

	const STAGES = [
		{
			key: 'rlrp',
			title: 'RLRP',
			sub: 'Robust Location–Routing',
			who: 'Thielen (ODM) · Gurobi MIP + scenario decomposition',
			input: {
				title: 'consumes',
				lines: [
					'one <b>average daily demand</b> per store, per scenario',
					'distance matrix over all nodes',
					'depot fixed + marginal cost, max size',
					'vehicle capacity Q'
				],
				note: 'No weekdays. No product segments. One number per store.'
			},
			output: {
				title: 'produces',
				lines: [
					'<code>depot_sizes[scenario][depot]</code> — t/day',
					'<code>customer_depot_assignment[scenario][depot]</code> — store ids',
					'<code>arcs</code> / <code>arc_loads</code> — the second stage\'s own tours'
				],
				note: 'Sized just big enough for the averages: capacity costs money.'
			}
		},
		{
			key: 'patt',
			title: 'PATT / DPPP',
			sub: 'Delivery Pattern Planning',
			who: 'Hübner (SCM) · ALNS (patterns) + LNS (routing)',
			input: {
				title: 'consumes',
				lines: [
					'demand per store × <b>segment</b> × <b>weekday</b>',
					'renumbered distances, Q, W₀, c_km, fuel, η',
					'<code>marginal_co2_emissions</code>, λ',
					'<code>Q_day_max</code> — the depot size RLRP chose'
				],
				note: 'One instance per (open depot, scenario).'
			},
			output: {
				title: 'produces',
				lines: [
					'one weekly <b>pattern</b> per store — 6 bits, Mon–Sat',
					'<code>p_frt</code> — delivered tonnes per (store, pattern, day)',
					'<code>S_fsr</code> — order-up-to level per (store, segment)',
					'<code>routes_by_day</code> — vehicle tours per weekday'
				],
				note: 'Plus its own predicted waste %, stockout % and emissions.'
			}
		},
		{
			key: 'sim',
			title: 'SIM',
			sub: 'Discrete-Event Simulation',
			who: 'Python DES port — runs in-process, no AnyLogic needed',
			input: {
				title: 'consumes',
				lines: [
					'<code>StoreCfg</code> per store: μ per segment/day, S, plan flags',
					'<code>routes_by_day</code> as store-index lists',
					'depot coordinates, execution variant 1–8'
				],
				note: 'Store ids shift once more: des_id = internal_id − 1. Quantities stay continuous.'
			},
			output: {
				title: 'produces',
				lines: [
					'realised waste %, stockout %, CO₂, cost, km',
					'cancelled routes, dropped stores, piggyback units'
				],
				note: 'Variant 2 executes the plan as written — it is the one compared to PATT.'
			}
		}
	];

	const TRANSFORMS = [
		{
			key: 't1',
			fn: 'main.create_patt_instance_data()',
			lines: [
				'writes one temp JSON per (depot, scenario)',
				'renumbers: depot → 0, stores → 1..n, <code>id_map</code> keeps the originals',
				'expands the single RLRP number back into segment × weekday demand',
				'passes the depot size through as <code>Q_day_max</code>'
			]
		},
		{
			key: 't2',
			fn: 'build_sim_inputs()  /  export_anylogic_csv()',
			lines: [
				'pattern + <code>p_frt</code> → per-day delivery flags',
				'<code>S_fsr</code> → order-up-to level per product',
				'segments renamed: fresh→A, dry→B, frozen→C',
				'routes → lists of DES store indices',
				'no rounding: quantities stay continuous tonnes'
			]
		}
	];
</script>

<div class="pipeline">
	{#each STAGES as stage, i (stage.key)}
		<div class="column">
			<div class="io in" class:sel={selected === stage.key}>
				<div class="io-title">{stage.input.title}</div>
				<ul>
					{#each stage.input.lines as line}<li>{@html line}</li>{/each}
				</ul>
				<div class="io-note">{stage.input.note}</div>
			</div>

			<div class="arrow-down" aria-hidden="true"></div>

			<button
				class="stage {stage.key}"
				class:sel={selected === stage.key}
				onclick={() => (selected = selected === stage.key ? null : stage.key)}
			>
				<div class="stage-title">{stage.title}</div>
				<div class="stage-sub">{stage.sub}</div>
				<div class="stage-who">{stage.who}</div>
			</button>

			<div class="arrow-down" aria-hidden="true"></div>

			<div class="io out">
				<div class="io-title">{stage.output.title}</div>
				<ul>
					{#each stage.output.lines as line}<li>{@html line}</li>{/each}
				</ul>
				<div class="io-note">{stage.output.note}</div>
			</div>
		</div>

		{#if i < TRANSFORMS.length}
			<div class="transform">
				<div class="chev" aria-hidden="true">→</div>
				<div class="transform-box">
					<div class="transform-fn mono">{TRANSFORMS[i].fn}</div>
					<ul>
						{#each TRANSFORMS[i].lines as line}<li>{@html line}</li>{/each}
					</ul>
				</div>
			</div>
		{/if}
	{/each}
</div>

<div class="feedback-bar">
	<div class="fb">
		<span class="fb-tag">feedback · PATT → RLRP</span>
		<strong>capacity shortfall</strong> — the minimum weekly delivery, spread over six days, does
		not fit the depot RLRP sized. Scale the affected stores' RLRP demand up and re-solve.
		<span class="fb-status ok">implemented &amp; validated</span>
	</div>
	<div class="fb">
		<span class="fb-tag">feedback · SIM → PATT</span>
		<strong>KPI miss</strong> — the simulated waste/stockout drifts from PATT's own prediction.
		Lower λ and re-solve PATT.
		<span class="fb-status provisional">provisional — criterion still being defined</span>
	</div>
</div>

<style>
	/* Explicit rows so the three stage boxes line up across columns even though
	   the consumes/produces boxes have different content heights. */
	.pipeline {
		display: grid;
		grid-template-columns: 1fr auto 1fr auto 1fr;
		grid-template-rows: 1fr auto auto auto 1fr;
		column-gap: 8px;
		align-items: stretch;
	}

	.column {
		display: contents;
	}

	.column > .in {
		grid-row: 1;
	}
	.column > .arrow-down:first-of-type {
		grid-row: 2;
	}
	.column > .stage {
		grid-row: 3;
	}
	.column > .arrow-down:last-of-type {
		grid-row: 4;
	}
	.column > .out {
		grid-row: 5;
	}

	.transform {
		grid-row: 1 / -1;
	}

	@media (max-width: 1000px) {
		.pipeline {
			grid-template-columns: 1fr;
			grid-template-rows: none;
			row-gap: 8px;
		}
		.column {
			display: flex;
			flex-direction: column;
			min-width: 0;
		}
		.column > * {
			grid-row: auto !important;
		}
		.transform {
			grid-row: auto;
			flex-direction: row !important;
			max-width: none;
		}
		.chev {
			transform: rotate(90deg);
		}
	}

	.io {
		border: 1px solid var(--border);
		border-radius: 9px;
		padding: 9px 12px;
		background: var(--surface-2);
		font-size: 0.78rem;
	}

	.io-title {
		font-size: 0.64rem;
		text-transform: uppercase;
		letter-spacing: 0.08em;
		color: var(--text-faint);
		margin-bottom: 5px;
	}

	.io ul {
		margin: 0;
		padding-left: 15px;
	}

	.io li {
		margin-bottom: 2px;
	}

	.io-note {
		margin-top: 6px;
		padding-top: 5px;
		border-top: 1px dashed var(--border);
		color: var(--text-faint);
		font-size: 0.73rem;
	}

	.arrow-down {
		width: 0;
		height: 14px;
		margin: 0 auto;
		border-left: 2px solid var(--border-strong);
	}

	.stage {
		border: 2px solid var(--border-strong);
		border-radius: 10px;
		padding: 11px 13px;
		background: var(--surface);
		text-align: left;
		cursor: pointer;
		font: inherit;
		color: inherit;
	}

	.stage.rlrp {
		border-color: var(--rlrp);
	}
	.stage.patt {
		border-color: var(--patt);
	}
	.stage.sim {
		border-color: var(--sim);
	}

	.stage-title {
		font-weight: 700;
		font-size: 1.05rem;
	}

	.stage-sub {
		font-size: 0.74rem;
		text-transform: uppercase;
		letter-spacing: 0.06em;
		color: var(--text-faint);
	}

	.stage-who {
		font-size: 0.75rem;
		color: var(--text-muted);
		margin-top: 4px;
	}

	.transform {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		gap: 6px;
		max-width: 230px;
		align-self: center;
	}

	.chev {
		color: var(--border-strong);
		font-size: 1.3rem;
		line-height: 1;
	}

	.transform-box {
		border: 1px dashed var(--border-strong);
		border-radius: 9px;
		padding: 8px 11px;
		background: var(--surface);
		font-size: 0.74rem;
	}

	.transform-fn {
		font-weight: 650;
		font-size: 0.72rem;
		margin-bottom: 5px;
		word-break: break-word;
	}

	.transform-box ul {
		margin: 0;
		padding-left: 14px;
		color: var(--text-muted);
	}

	.transform-box li {
		margin-bottom: 2px;
	}

	.feedback-bar {
		margin-top: 16px;
		display: grid;
		gap: 8px;
	}

	.fb {
		border: 1px dashed var(--feedback);
		border-radius: 9px;
		padding: 9px 13px;
		background: color-mix(in srgb, var(--feedback) 7%, transparent);
		font-size: 0.8rem;
	}

	.fb-tag {
		display: inline-block;
		font-size: 0.66rem;
		text-transform: uppercase;
		letter-spacing: 0.07em;
		color: var(--feedback);
		font-weight: 700;
		margin-right: 7px;
	}

	.fb-status {
		display: inline-block;
		margin-left: 7px;
		padding: 1px 8px;
		border-radius: 999px;
		font-size: 0.68rem;
		font-weight: 600;
		border: 1px solid currentColor;
	}

	.fb-status.ok {
		color: var(--ok);
	}

	.fb-status.provisional {
		color: var(--bad);
	}

	:global(.io code, .transform-box code) {
		font-family: var(--mono);
		font-size: 0.92em;
		background: color-mix(in srgb, var(--text) 7%, transparent);
		padding: 0 3px;
		border-radius: 3px;
	}
</style>

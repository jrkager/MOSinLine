<script lang="ts">
	import PipelineDiagram from '$lib/components/PipelineDiagram.svelte';

	let selected = $state<string | null>(null);
</script>

<div class="card">
	<h1>How this works</h1>
	<div class="card-sub">
		What the three modules are, what each one hands to the next, and how the data is transformed
		in between.
	</div>
</div>

<div class="card prose">
	<h2>The short version</h2>
	<p>
		MOSinLine chains together three pieces of research code that were written independently, by
		different people, for different questions. Making them agree on a single grocery-retail
		supply chain — and feed results back into each other — is the whole point of the project.
	</p>
	<ul>
		<li>
			<strong>RLRP</strong> answers the <em>strategic</em> question: which distribution centres do
			we open, how big, and which stores does each one serve? It is a Gurobi mixed-integer model
			solved robustly across several demand scenarios.
		</li>
		<li>
			<strong>PATT</strong> (also called DPPP) answers the <em>tactical</em> question: given those
			depots, on which weekdays does each store get a delivery, how much arrives, and which truck
			tours serve them? It is a metaheuristic — ALNS for the patterns, LNS for the routing.
		</li>
		<li>
			<strong>SIM</strong> answers the <em>operational</em> question: if we actually execute that
			plan for a year, with random demand and real-world execution rules, what happens? It is a
			discrete-event simulation.
		</li>
	</ul>
	<p>
		They do not just run in sequence. Each stage can discover that the previous stage's decision
		does not survive contact with its own constraints, and hand the problem back. That closed loop
		is what the <a href="/">loop view</a> visualises.
	</p>
</div>

<div class="card">
	<div class="card-header">
		<div>
			<h2>The protocol</h2>
			<div class="card-sub">
				Each module's inputs and outputs, and the named function that translates between them.
			</div>
		</div>
	</div>
	<PipelineDiagram bind:selected />
</div>

<div class="card prose">
	<h2>Why the two models disagree about demand</h2>
	<p>
		This is the single most important thing to understand about the pipeline, and it is the reason
		the first feedback edge exists at all.
	</p>
	<p>
		<strong>The RLRP works with aggregate demand.</strong> Each store enters the model as one number:
		its average daily demand, with no weekday differentiation and no product segments. The model then
		sizes each warehouse just big enough to cover those averages, because extra capacity costs money
		and nothing in the model rewards it.
	</p>
	<p>
		<strong>PATT works quite differently.</strong> Demand varies by weekday, and a store is only
		visited on a few days per week. So on a delivery day the truck brings several days of demand at
		once — and a bit extra on top.
	</p>
	<p>
		That extra comes from perishability. Under the (R,S) inventory policy, every delivery refills
		the store to its order-up-to level <span class="mono">S</span>, so any units that expired on the
		shelf since the last visit are automatically re-ordered. Over time the delivered quantity
		therefore equals <em>demand + expired units</em>. For a store visited five or six times a week
		this surplus is negligible; as the delivery frequency drops it grows.
	</p>
	<div class="callout">
		The consequence: the tonnage that must actually leave the depot on a delivery day can exceed the
		warehouse capacity the RLRP chose. Treating that capacity as a hard daily limit can make PATT
		infeasible — not because the plan is bad, but because the two models were never talking about
		the same quantity.
	</div>
</div>

<div class="card prose">
	<h2>The capacity pre-check</h2>
	<p>
		Rather than discovering this after an expensive PATT solve, the pipeline runs a cheap arithmetic
		check first, for every (scenario, depot) pair:
	</p>
	<ol>
		<li>For each store, find the pattern with the <strong>smallest total weekly delivery</strong>.</li>
		<li>Sum those minima across all stores assigned to the depot.</li>
		<li>Divide by six delivery days to get the required daily throughput.</li>
		<li>Compare against the depot capacity, with a small margin.</li>
	</ol>
	<p>
		If even this lower bound does not fit, then <em>no</em> combination of patterns can fit, and
		there is no point running PATT at all. The requirement is fed back to the RLRP by scaling up the
		affected stores' demand in the RLRP input, with a safety margin, and the RLRP is re-solved. It
		then either builds a larger warehouse or reassigns some stores to another depot. This repeats
		until every (scenario, depot) combination passes.
	</p>
	<div class="callout">
		The scaling is deliberately capped per round. A single large jump was found to crash the RLRP's
		second-stage model, so the loop walks up gently instead.
	</div>
	<p class="faint">
		In the code: <span class="mono">capacity_check()</span> and
		<span class="mono">_apply_capacity_feedback()</span> in
		<span class="mono">webtool/pipeline.py</span>, following
		<span class="mono">run_pipeline_B.py</span>.
	</p>
</div>

<div class="card prose">
	<h2>The second loop: simulation feedback</h2>
	<p>
		PATT predicts what its own plan will achieve — waste, stockouts, emissions. The simulation then
		executes that plan under realistic rules and measures what really happens. Variant 2 executes
		the plan exactly as written, so the Variant 2 row and the PATT row should agree closely; the
		other variants apply operational rules (dropping small deliveries, cancelling under-full routes,
		piggybacking) and show how much the outcome depends on execution policy rather than on the
		optimization.
	</p>
	<p>
		If they disagree beyond a tolerance, λ is lowered and PATT re-solves. Empirically, a
		stockout-driven miss calls for lowering λ, shifting weight from emissions toward cost and buying
		more delivery frequency.
	</p>
	<div class="callout warn">
		<strong>This edge is provisional.</strong> The acceptance criterion currently implemented —
		accept when the reference variant's waste % and stockout % stay within a fixed tolerance of the
		PATT prediction — is a placeholder chosen so the loop is demonstrable. The real criterion is
		still being defined, and the thresholds need a modelling decision before any number from this
		edge is quoted.
	</div>
</div>

<div class="card prose">
	<h2>Conventions that silently break things</h2>
	<p>
		Most of the translation work between the three modules is bookkeeping, and most of the bugs
		found during integration were bookkeeping errors that produced plausible-looking wrong numbers
		rather than crashes.
	</p>
	<div class="table-wrap">
		<table>
			<thead>
				<tr><th>Thing</th><th>Convention</th><th>Where it changes</th></tr>
			</thead>
			<tbody>
				<tr>
					<td>Node ids</td>
					<td class="mono">depots negative, stores positive</td>
					<td>PATT renumbers depot→0, stores→1..n; the DES shifts again to id−1</td>
				</tr>
				<tr>
					<td>Weekdays</td>
					<td class="mono">0..5 = Mon..Sat</td>
					<td>six-day week everywhere; there is no Sunday</td>
				</tr>
				<tr>
					<td>Segments</td>
					<td class="mono">dry / fresh / frozen</td>
					<td>become B / A / C in the DES and the AnyLogic CSVs</td>
				</tr>
				<tr>
					<td>Units</td>
					<td class="mono">tonnes</td>
					<td>Q = 25.6 t, W₀ = 14.4 t; θ_FW is kg CO₂e per tonne wasted</td>
				</tr>
				<tr>
					<td>Delivery quantities</td>
					<td class="mono">integer units</td>
					<td>rounded inside PATT's own shelf simulation so it matches the DES</td>
				</tr>
			</tbody>
		</table>
	</div>
	<div class="callout">
		The lookup tables <span class="mono">p_frt</span>, <span class="mono">S_fsr</span> and
		<span class="mono">pattern_assignments</span> are keyed by PATT's <em>internal</em> ids. Reading
		them with the original store ids returns zeros rather than an error — this was a real bug, and
		it made every delivery quantity silently vanish.
	</div>
</div>

<div class="card prose">
	<h2>One objective, two models</h2>
	<p>
		Both models optimise the same weighted objective,
		<span class="mono">(1−λ)·economic + λ·emissions</span>, with the same λ. They just express it
		differently: PATT computes the arc cost directly, while the RLRP sums raw coefficients.
	</p>
	<p>
		To make them agree exactly, the λ weighting is folded into the RLRP's arc coefficients
		<span class="mono">c_ij</span>, <span class="mono">alpha_ij</span> and
		<span class="mono">gamma_ij</span> when the instance is built. Warehouse costs are economic, so
		they are pre-scaled by <span class="mono">(1−λ)</span>. The result is that the RLRP minimises a
		positive multiple of exactly the same scalarization PATT uses.
	</p>
	<div class="callout warn">
		Changing λ on one side only silently de-aligns the two objectives — the models keep running and
		keep producing numbers, but they are no longer optimising the same thing. The run form always
		sets both together.
	</div>
</div>

<div class="card prose">
	<h2>What happens after the loop</h2>
	<p>
		Once every (scenario, depot) combination passes the capacity check and the plan is accepted, the
		pipeline can export the patterns and routes as CSVs for the AnyLogic simulation model, which is
		the higher-fidelity counterpart of the Python DES used inside the loop.
	</p>
	<p class="faint">
		That export exists in the codebase (<span class="mono">export_anylogic_csv.py</span>,
		<span class="mono">export_for_anylogic.py</span>) but is not yet wired into this web tool — see
		the notes in <span class="mono">WEBTOOL.md</span>.
	</p>
</div>

<style>
	.prose p {
		max-width: 78ch;
		color: var(--text-muted);
	}

	.prose li {
		max-width: 78ch;
		color: var(--text-muted);
		margin-bottom: 4px;
	}

	.prose h2 {
		margin-bottom: 8px;
	}

	.prose strong,
	.prose em {
		color: var(--text);
	}

	.callout {
		border-left: 3px solid var(--rlrp);
		background: var(--surface-2);
		border-radius: 0 8px 8px 0;
		padding: 10px 14px;
		margin: 12px 0;
		max-width: 82ch;
		font-size: 0.87rem;
	}

	.callout.warn {
		border-left-color: var(--feedback);
		background: color-mix(in srgb, var(--feedback) 8%, transparent);
	}

	.prose a {
		text-decoration: underline;
		text-underline-offset: 2px;
	}
</style>

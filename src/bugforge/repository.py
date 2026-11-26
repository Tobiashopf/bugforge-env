"""Opt-in synthetic repository families with curated, cross-module defects."""
from __future__ import annotations

import random

from .base import load, subset_tests
from .generator import (
    API_DIFFICULTY,
    LEDGER_DIFFICULTY,
    NUMERICAL_DIFFICULTY,
    RECONCILIATION_DIFFICULTY,
    REPOSITORY_DIFFICULTY,
    SCHEDULER_DIFFICULTY,
    Bug,
    Task,
    _visible_fails,
)
from .verifier import verify

# Each defect has a direct witness in the public API integration suite.
DEFECTS = (
    ('repo_pricing.py', 'unit_cents * quantity', 'unit_cents + quantity', 'test_price_quantity'),
    ('repo_pricing.py', 'subtotal >= 1000', 'subtotal > 1000', 'test_member_threshold'),
    ('repo_stock.py', 'remaining.get(sku, 0) < quantity',
     'remaining.get(sku, 0) <= quantity', 'test_exact_stock'),
    ('repository_checkout.py', 'demand.get(sku, 0) + quantity', 'quantity', 'test_duplicate_sku'),
)


CHECKOUT_INSTRUCTIONS = '''# Checkout service repair
Repair the three Python modules behind `checkout(lines, stock, member=False)`.
Each line is (SKU, unit price in integer cents, positive quantity). Aggregate
repeat SKUs, reject negative prices and insufficient inventory. Treat the input stock as
a snapshot, returning the remaining stock. Members receive a 10% discount (rounded down to whole cents)
on subtotals of at least 1000 cents. Return subtotal, discount, total and
remaining inventory. Only the three original Python modules are graded.
Visible tests are a subset of the trusted evaluation suite.
'''

LEDGER_DEFECTS = (
    ('ledger_rules.py', 'amount <= 0', 'amount < 0', 'test_positive_amount'),
    ('ledger_balances.py', 'balances[source] < amount', 'balances[source] <= amount', 'test_exact_funds'),
    ('ledger_balances.py', 'updated[target] += amount', 'updated[target] -= amount', 'test_credit_destination'),
    ('repository_ledger.py', '        continue', '        pass', 'test_duplicate_id'),
    ('repository_ledger.py', 'seen[tx_id] != payload', 'False', 'test_conflicting_id'),
)

LEDGER_INSTRUCTIONS = """# Transaction ledger repair
Repair `replay(opening, transactions)` across the three original modules.
Opening is a mapping of nonempty string account IDs to nonnegative integer cents.
Each transaction is (nonempty ID, source account, target account, positive integer cents).
Both accounts must exist and differ. For well-formed transaction tuples, reject invalid IDs, accounts, amounts or insufficient funds with ValueError.
Execute in input order, debit source and credit destination exactly once per ID.
An identical repeated ID is a no-op; a reused ID with a different payload is invalid.
Return {"balances": final balances, "applied": IDs in first-application order}.
The reference uses a private snapshot; return values and exceptions are graded.
Only the original three modules are graded.
Visible tests are a subset of the trusted evaluation suite.
"""


def _generate(seed: int, difficulty, base_name: str, defects, instructions: str,
              *, check: bool) -> Task:
    base = load(base_name)
    solution = base.sources
    rng = random.Random(f'bugforge:{difficulty.name}:{seed}')
    pairs = [(a, b) for i, a in enumerate(defects) for b in defects[i + 1:] if a[0] != b[0]]
    chosen = rng.choice(pairs)
    workspace = dict(solution)
    bugs = []
    witnesses = set()
    for filename, old, new, witness in chosen:
        source = workspace[filename]
        if source.count(old) != 1:
            raise RuntimeError(f'stale curated mutation: {filename}: {old}')
        line = source[:source.index(old)].count('\n') + 1
        workspace[filename] = source.replace(old, new, 1)
        bugs.append(Bug(filename, witness, 'curated', line, f'{old} -> {new}'))
        witnesses.add(witness)
    # A fixed visible fraction, including a witness for both injected defects.
    remaining = sorted(set(base.test_names()) - witnesses)
    count = max(0, round(difficulty.visible_frac * len(base.test_names())) - len(witnesses))
    visible = sorted(witnesses | set(rng.sample(remaining, count)))
    workspace[base.test_file] = subset_tests(base.tests, set(visible))
    workspace['README.md'] = instructions
    task = Task(seed, difficulty.name, [base.name], workspace, solution,
                bugs, {base.name: visible})
    if check:
        result = verify({f: workspace[f] for f in task.module_files}, [base], fuzz_seed=seed)
        if not result.valid or result.solved or not _visible_fails(task):
            raise RuntimeError('invalid repository task: no valid failing baseline')
        if not verify(solution, [base], fuzz_seed=seed).solved:
            raise RuntimeError('repository reference failed')
    return task


def generate_checkout(seed: int, *, check: bool = True) -> Task:
    return _generate(seed, REPOSITORY_DIFFICULTY, 'repository_checkout', DEFECTS,
                     CHECKOUT_INSTRUCTIONS, check=check)


def generate_ledger(seed: int, *, check: bool = True) -> Task:
    return _generate(seed, LEDGER_DIFFICULTY, 'repository_ledger', LEDGER_DEFECTS,
                     LEDGER_INSTRUCTIONS, check=check)


NUMERICAL_DEFECTS = (
    ('numerical_stats.py', 'math.fsum(values) / len(values)',
     'math.fsum(values) / (len(values) + 1)', 'test_mean'),
    ('numerical_stats.py', 'math.fsum(squares) / len(squares)',
     'math.fsum(squares) / max(1, len(squares) - 1)', 'test_population_scale'),
    ('numerical_transform.py', 'x = mean', 'x = 0.0', 'test_missing_imputation'),
    ('numerical_transform.py', '(x - mean) / scale', '(x - mean) * scale',
     'test_standardization'),
    ('repository_numerical.py', 'transform(inference, state)',
     'transform(inference, fit(inference))', 'test_no_inference_leakage'),
)

NUMERICAL_INSTRUCTIONS = """# Numerical pipeline repair
Repair the three modules behind fit(training), transform(values, state), and run(training, inference).
Inputs are lists of finite int or float values with magnitude at most 1e12, or None for missing.
Reject booleans and other invalid observations with ValueError. Fit the mean and population
standard deviation on observed training values only; reject training with no observations.
Use scale 1.0 for variance below 1e-200 (including constant training data).
State is a dictionary of finite mean and scale, with |mean| <= 1e12 and scale in [1e-100, 1e12].
Transform using (x - mean) / scale, imputing missing observations with the fitted mean.
Preserve input order and use the same training state for both training and inference.
Empty inference returns an empty list. Return a dict with state, training, and inference keys.
Only the three original Python modules are graded. Visible tests are a subset of the trusted suite.
"""


def generate_numerical(seed: int, *, check: bool = True) -> Task:
    return _generate(seed, NUMERICAL_DIFFICULTY, 'repository_numerical', NUMERICAL_DEFECTS,
                     NUMERICAL_INSTRUCTIONS, check=check)


API_DEFECTS = (
    ('api_schema.py', " or set(record) - {'id', 'name', 'email', 'active'}",
     '', 'test_v1_unknown_fields'),
    ('api_schema.py', "type(record['enabled']) is not bool",
     "type(record['enabled']) not in (bool, int)", 'test_v2_boolean'),
    ('api_migration.py', "record.get('active', True)",
     "record.get('active', False)", 'test_upgrade_default'),
    ('api_migration.py', "'email': record['profile']['email']",
     "'email': None", 'test_downgrade_email'),
    ('repository_api.py', 'if target == 1:', 'if target == 1 or source == 2:',
     'test_same_version_v2'),
)

API_INSTRUCTIONS = """# Versioned profile API repair
Repair the three modules behind validate_v1(record), validate_v2(record), upgrade(record),
downgrade(record) and convert(record, source, target).
V1 is a dict with required id/name (nonblank strings), optional email (None or nonblank string,
default None), optional active (strict bool, default True). Reject unknown fields.
V2 is a dict with exactly id, profile and enabled. Profile is a dict with exactly name and email;
enabled is a strict bool. Apply the same name, id and email rules. No email syntax checking.
Validators return True for valid data and raise ValueError otherwise. Upgrade and downgrade
return the canonical target representation with all fields, preserving values and defaults.
Convert accepts only integer versions 1 and 2 (not booleans), validates input, and returns the
canonical target even when source equals target. Reject invalid inputs with ValueError.
Only returned values and exceptions from the public API are graded; only the original three
modules are graded. Visible tests are a subset of the trusted suite.
"""


def generate_api(seed: int, *, check: bool = True) -> Task:
    return _generate(seed, API_DIFFICULTY, 'repository_api', API_DEFECTS,
                     API_INSTRUCTIONS, check=check)


SCHEDULER_DEFECTS = (
    ('scheduler_rules.py', "type(job['priority']) is not int",
     "type(job['priority']) not in (int, bool)", 'test_priority_type'),
    ('scheduler_rules.py', 'unique[key] != job', 'False', 'test_conflict'),
    ('scheduler_state.py', "(-jobs[key]['priority'], key)",
     "-jobs[key]['priority']", 'test_tie_break'),
    ('scheduler_state.py', 'count >= max_attempts', 'count > max_attempts', 'test_retry_limit'),
    ('repository_scheduler.py', "all((states[dep] == 'succeeded' for dep in specs[key]['deps']))",
     'True', 'test_dependency_gate'),
)

SCHEDULER_INSTRUCTIONS = """# Deterministic job scheduler repair
Repair schedule(jobs, max_attempts) across the original three Python modules.
Jobs is a list of dictionaries with exactly id (nonblank string), priority (integer),
deps (list of distinct nonblank job IDs), failures (nonnegative integer, number of
simulated failed attempts before success). max_attempts is a positive integer.
Reject booleans for integer fields, unknown dependencies, conflicting duplicate IDs
and malformed specifications with ValueError. Identical duplicate definitions run once.
At each step, select a pending job whose dependencies have all succeeded; higher priority
wins, with lexicographically smaller ID breaking ties. Attempt it once. It succeeds if
its attempt count exceeds failures; otherwise mark failed if the attempt budget is
exhausted, or pending if another attempt is allowed. Reevaluate priority after each attempt.
When no ready job remains, mark all remaining pending jobs blocked (including cycles).
Return dict with states (ID to final status), attempts (ID to count including zeros),
and events (ordered lists [ID, attempt count, status after attempt]). Empty jobs are valid.
Only returned values and exceptions are graded. Only the three original modules are graded.
Visible tests are a subset of the trusted evaluation suite.
"""


def generate_scheduler(seed: int, *, check: bool = True) -> Task:
    return _generate(seed, SCHEDULER_DIFFICULTY, 'repository_scheduler', SCHEDULER_DEFECTS,
                     SCHEDULER_INSTRUCTIONS, check=check)


RECONCILIATION_DEFECTS = (
    ('recon_rules.py', 'cents <= 0', 'cents < 0', 'test_positive_payment'),
    ('recon_rules.py', 'unique[key] != payload', 'False', 'test_conflict'),
    ('recon_amounts.py', 'min(amount, remaining)', 'amount', 'test_overpayment'),
    ('recon_amounts.py', 'amount - applied', '0', 'test_unapplied'),
    ('repository_reconciliation.py', 'outstanding[invoice] -= applied',
     'outstanding[invoice] += applied', 'test_outstanding'),
)

RECONCILIATION_INSTRUCTIONS = """# Invoice and payment reconciliation repair
Repair reconcile(invoices, payments) across the original three modules.
Invoices is a list of pairs (nonblank string invoice ID, positive integer cents).
Reject duplicate invoice IDs, even identical ones. Payments is a list of triples
(nonblank string payment ID, nonblank string invoice ID, positive integer cents).
Rows may be lists or tuples. Reject malformed records and booleans with ValueError.
Process unique payment IDs in first-seen order. Identical repeated payments are a no-op;
conflicting reuse of a payment ID raises ValueError. Allocate min(payment, outstanding)
for known invoices; unknown invoices receive zero allocation. Excess is unapplied.
Return a dictionary with outstanding (invoice ID to remaining cents) and payments
(payment ID to {invoice, applied, unapplied}, all monetary values integer cents).
Keep fully paid invoices with zero outstanding. Empty lists are valid.
Only returned values and exceptions are graded. Only the three original modules are graded.
Visible tests are a subset of the trusted evaluation suite.
"""


def generate_reconciliation(seed: int, *, check: bool = True) -> Task:
    return _generate(seed, RECONCILIATION_DIFFICULTY, 'repository_reconciliation',
                     RECONCILIATION_DEFECTS, RECONCILIATION_INSTRUCTIONS, check=check)

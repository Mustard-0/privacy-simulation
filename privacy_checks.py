# Mechanically extracted unchanged from the frozen v76 privacy controller.
import csv, hashlib, json, math, os, statistics, sys
from pathlib import Path
METHODS = ('TMAPO', 'TMAPO-NoAmp', 'MARATD3-Adapted')
EPSILONS = (0.5, 1.0, 2.0, 4.0)
RESERVED_TEST_SEEDS = [91005, 91003, 91008, 91020, 91001]

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)

def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        yield from csv.DictReader(handle)

def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else _sha_stream(handle)

def _sha_stream(handle):
    digest = hashlib.sha256()
    for block in iter(lambda : handle.read(1048576), b''):
        digest.update(block)
    return digest.hexdigest()

def verify_source(source, hashes):
    actual = {p.name: sha(p) for p in Path(source).glob('*.py')}
    if actual != hashes:
        raise ValueError('Frozen model source changed; do not resume or overwrite')

def job(method, epsilon, lr, seed):
    return dict(method=method, epsilon_req=epsilon, users_per_region=100, lr=lr, seed=seed, kind='privacy', id=f'{method}_M100_eps{epsilon:g}_LR{lr + 1}_s{seed}')

def screen_jobs(config, lr):
    seed = config['training']['seed']
    jobs = [job(method, epsilon, 0 if method == METHODS[2] else lr, seed) for epsilon in EPSILONS for method in METHODS]
    jobs.append(job(METHODS[2], 1.0, 1, seed))
    return jobs

def validation_choice(stage, candidates):
    choices = []
    for candidate in candidates:
        grouped = {}
        for row in rows(stage / candidate['id'] / 'validation_raw.csv'):
            grouped.setdefault(int(row['update']), []).append(float(row['mean_task_cost']))
        (update, cost) = min(((update, statistics.mean(values)) for (update, values) in grouped.items()), key=lambda x: x[1])
        choices.append(dict(job=candidate['id'], lr=candidate['lr'], update=update, validation_cost=cost))
    return min(choices, key=lambda x: x['validation_cost'])

def screen_gate(stage, probe):
    jobs = read(stage / 'plan.json')['jobs']
    result = {'passed': False, 'reasons': [], 'cells': [], 'not_a_superiority_test': True}
    if not read(probe / 'check.json').get('passed') or not read(stage / 'audit.json').get('passed'):
        result['reasons'].append('QL/accountant check or raw-data audit failed')
    observed = False
    for candidate in jobs:
        data = list(rows(stage / candidate['id'] / 'test_raw.csv'))
        cell = {key: candidate[key] for key in ('id', 'method', 'epsilon_req', 'lr')}
        for key in ('completion_rate', 'qos_rate', 'ql_m', 'ql_theory_m', 'amplified_publication_ratio', 'active_wait_s', 'mean_task_cost'):
            values = [float(r[key]) for r in data if r.get(key) not in (None, '')]
            cell[key] = statistics.mean(values) if values else None
        if cell['completion_rate'] < 0.9 or cell['qos_rate'] > 0.1:
            result['reasons'].append(candidate['id'] + ': completion<90% or QoS>10%')
        if cell['ql_m'] is None:
            result['reasons'].append(candidate['id'] + ': no published reports')
        diagnostics = list(rows(stage / candidate['id'] / 'train_raw.csv')) if (stage / candidate['id'] / 'train_raw.csv').exists() else []
        if diagnostics:
            last = diagnostics[-10:]
            cell['recent_kl'] = statistics.mean((float(r['kl']) for r in last))
            cell['recent_clip_fraction'] = statistics.mean((float(r['clip_fraction']) for r in last))
            if cell['recent_kl'] > 0.1 or cell['recent_clip_fraction'] > 0.5:
                result['reasons'].append(candidate['id'] + ': unstable PPO KL/clip diagnostics')
        if candidate['method'] == 'TMAPO' and candidate['epsilon_req'] in (0.5, 1.0) and (cell['amplified_publication_ratio'] > 0.01):
            pure_local_ql = 4 * read(stage / 'config.json')['environment']['region_radius_m'] / candidate['epsilon_req']
            observed |= cell['ql_theory_m'] < pure_local_ql * (1 - 1e-06)
        result['cells'].append(cell)
    if not observed:
        result['reasons'].append('No observable amplification (>1% publications) at either screened privacy budget')
    result['baseline_choice'] = validation_choice(stage, [j for j in jobs if j['method'] == METHODS[2] and j['epsilon_req'] == 1.0])
    result['passed'] = not result['reasons']
    result['scope'] = '30-update one-seed mechanism screen; diagnostic test bank only, never final paper evidence'
    return result

def model_path(out):
    meta = read(out / 'pipeline.json')
    verify_source(out / 'model_source', meta['model_hashes'])
    sys.path.insert(0, str(out / 'model_source'))

def audit_batches(out, stage):
    model_path(out)
    from privacy import delta_bound
    checks = []
    for candidate in read(stage / 'plan.json')['jobs']:
        config = read(stage / 'config.json')['environment']
        count = 0
        for batch in rows(stage / candidate['id'] / 'test_batches.csv'):
            n = int(batch['n_distinct'])
            identifiers = batch['task_ids'].split(';')
            epsilon0 = float(batch['epsilon0'])
            if n != len(identifiers) or len(set(identifiers)) != n or (not 1 <= n <= candidate['users_per_region']):
                raise ValueError('Batch membership count inconsistent with raw task IDs')
            if not math.isclose(float(batch['epsilon_loc_per_m']) * 2 * config['region_radius_m'], epsilon0):
                raise ValueError('Spatial local-budget unit mismatch')
            if candidate['method'] == 'TMAPO-NoAmp' and (not math.isclose(epsilon0, candidate['epsilon_req'])):
                raise ValueError('NoAmp changed the pure local privacy parameter')
            if delta_bound(epsilon0, n, candidate['epsilon_req']) > config['delta_req'] + 1e-12:
                raise ValueError('Published batch violates the saved conditional accountant')
            count += 1
        checks.append(dict(job=candidate['id'], audited_batches=count))
    save(stage / 'batch_audit.json', dict(passed=True, checks=checks, scope='raw task-ID counts plus frozen-source distinct-user assertion; conditional batch bound, not trajectory DP'))

def audit_full(out, stage):
    """Stream raw tasks to avoid loading millions of CSV rows at once."""
    reference = {}
    summary = []
    for candidate in read(stage / 'plan.json')['jobs']:
        folder = stage / candidate['id']
        complete = read(folder / 'complete.json')
        config = complete['config']
        e = config['environment']
        if complete['model_hashes'] != read(out / 'pipeline.json')['model_hashes'] or complete['job'] != candidate:
            raise ValueError('Evaluation source mismatch')
        if complete['checkpoint_hash'] != sha(complete['checkpoint']):
            raise ValueError('Evaluation checkpoint changed')
        aggregates = {}
        seen = set()
        for r in rows(folder / 'test_tasks.csv'):
            seed = int(r['trace_seed'])
            key = (seed, int(r['task_id']))
            if key in seen:
                raise ValueError('Duplicated task in full evaluation')
            seen.add(key)
            a = aggregates.setdefault(seed, dict(n=0, completed=0, qos=0, energy=0.0, delay=0.0, cost=0.0, signature=0, published=0, ql=0.0))
            (done, qos) = (int(r['completed']), int(r['qos']))
            duration = float(r['end_s']) - float(r['arrival_s'])
            energy = float(r['energy_j'])
            if not math.isfinite(duration + energy) or duration < 0 or energy < 0:
                raise ValueError('Invalid raw task timing/energy')
            if done not in (0, 1) or qos not in (0, 1) or qos != int(not done or duration > float(r['deadline_s']) + 1e-09):
                raise ValueError('Raw QoS does not match completion/deadline')
            components = sum((float(r[key]) for key in ('transmit_energy_j', 'compute_energy_j', 'report_energy_j')))
            if not math.isclose(energy, components, rel_tol=1e-06, abs_tol=1e-08):
                raise ValueError('Raw energy components do not sum to task energy')
            a['n'] += 1
            a['completed'] += done
            a['qos'] += qos
            a['energy'] += energy
            a['delay'] += duration * done
            a['cost'] += e['weight_time'] * duration / e['time_ref_s'] + e['weight_energy'] * energy / e['energy_ref_j'] + e['weight_violation'] * qos
            signature = [r[k] for k in ('task_id', 'user', 'region', 'arrival_s', 'task_profile', 'task_bits', 'task_cycles', 'deadline_s')]
            a['signature'] ^= int(hashlib.sha256(json.dumps(signature).encode()).hexdigest(), 16)
            if int(r['published']):
                a['published'] += 1
                a['ql'] += float(r['ql_m'])
        tests = list(rows(folder / 'test_raw.csv'))
        if {int(r['trace_seed']) for r in tests} != set(RESERVED_TEST_SEEDS) or len(tests) != 5:
            raise ValueError('Missing reserved evaluation trace')
        for r in tests:
            seed = int(r['trace_seed'])
            a = aggregates[seed]
            if float(r['window_s']) != 300 or a['n'] != int(r['arrivals']):
                raise ValueError('Incomplete full-cycle cohort')
            paired = (a['signature'], a['n'], r['trace_hash'], r['phase_s'], r['window_s'])
            if seed in reference and reference[seed] != paired:
                raise ValueError('Unpaired external full-cycle workloads')
            reference[seed] = paired
            measured = dict(completion_rate=a['completed'] / a['n'], qos_rate=a['qos'] / a['n'], energy_j_per_arrival=a['energy'] / a['n'], mean_task_cost=a['cost'] / a['n'], delay_s=a['delay'] / a['completed'] if a['completed'] else None, ql_m=a['ql'] / a['published'] if a['published'] else None)
            if any((v is None for v in measured.values())):
                raise ValueError('No completed tasks or published reports in a full evaluation trace')
            if any((not math.isclose(v, float(r[k]), rel_tol=1e-06, abs_tol=1e-08) for (k, v) in measured.items())):
                raise ValueError('Raw task/aggregate accounting mismatch')
        summary.append(dict(job=candidate['id'], traces=len(tests), paired=True, raw_task_accounting=True))
    save(stage / 'audit.json', dict(passed=True, jobs=summary, training_seeds=1, evaluation_seconds=300, scope='conditional fixed-membership batch privacy; not trajectory DP or multiseed superiority'))

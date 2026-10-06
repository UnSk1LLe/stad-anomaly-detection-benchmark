export const meta = {
  name: 'implement-review-fix',
  description: 'Class-A tasks of the experiment: implement each in its own worktree, two adversarial reviews, verified fixes',
  whenToUse: 'Several independent code tasks with written specs (e.g. a stage of docs/TZ_PROTOCOL_V2.md); args = {repo, scratch, base, attribution, tasks:[{key, branch, spec}]}',
  phases: [
    { title: 'Implement', detail: 'one agent per task, own git worktree and branch from args.base' },
    { title: 'Review', detail: 'two independent lenses per branch: spec/protocol and bugs/tests' },
    { title: 'Fix', detail: 'verify findings, apply real ones, commit <branch>-final' },
  ],
}

// Сценарий, которым выполнялся этап 1 ТЗ v2 (см. docs/RUNBOOK_TZ_V2.md §4).
// Аргументы (пример — docs/workflows/tz_v2_stage1.args.json):
//   repo        — абсолютный путь к основной копии репозитория (там data/ и reports/)
//   scratch     — каталог для временных файлов агентов (вне репозитория)
//   base        — коммит, от которого ветвятся все задачи (общий контракт уже в нём)
//   attribution — строки, которыми заканчиваются сообщения коммитов (можно пусто)
//   tasks       — [{key, branch, spec}]; в spec можно писать {repo} и {scratch}
// Результат: для каждой задачи ветка <branch>-final; интеграция (squash по одному
// коммиту на задачу, полный набор тестов перед каждым коммитом) — вручную.

const REPO = args.repo
const SCRATCH = args.scratch
const BASE = args.base
const ATTR = args.attribution || ''
const fill = s => s.replace(/\{repo\}/g, REPO).replace(/\{scratch\}/g, SCRATCH)

const common = branch => `
You are working in a git worktree of the PhD repository "stad-anomaly-detection-benchmark" (traffic anomaly detection benchmark). The main checkout is at ${REPO}; your worktree is your current directory. Base commit: ${BASE}.

Read first: docs/TZ_PROTOCOL_V2.md, docs/PROTOCOL.md, docs/DECISION_RULES.md, and the code you touch. CLAUDE.md rules are already in your context; they are hard constraints.

HARD CONSTRAINTS (violating any invalidates the work):
- Class A only: implementation changes. Do NOT edit these gated paths: src/stad/metrics/, src/stad/registry.py, src/stad/data/, src/stad/encoders/, src/stad/heads/, docs/PROTOCOL.md, docs/DECISION_RULES.md, CLAUDE.md, .claude/, configs/ — unless the task spec below says the user approved that exact class-B change. Do NOT hand-edit anything under reports/ or data/. If you believe a gated file must change, STOP and report it in "blocked".
- Never pick thresholds from test labels in anything that feeds ranking; oracle quantities must be labelled ORACLE/diagnostic. Never select best seed. Never rank by pa_f1_INVALID_for_ranking. Never remove ctrl_random / ctrl_untrained / base_pca.
- Do not change the values of alarm_budget_per_hour, persistence, half_life_min, lead_min, trail_min, node_reduce, param_budget.
- Match surrounding code: Russian docstrings/comments in the same tone and density, same idioms, line length <= 110. Keep changes minimal and focused on your task.
- Run tests with: python -m pytest tests -q -p no:cacheprovider   (run your new test file first, then the full suite once at the end). The full suite MUST be green before you commit.
- No "rm -rf", no "git reset", no "git checkout -- <file>" (blocked by the protocol hook). No git push. Do not touch other branches.

GIT: your very first command must be: git checkout -b ${branch} ${BASE}
Then check with "git log --oneline -1" that HEAD is ${BASE}. At the end: git add only your files, verify with "git status --short" and "git diff --cached --stat", then commit with a Russian message in the repository style (short summary line, blank line, body explaining why)${ATTR ? ', ending with these lines exactly:\n' + ATTR : '.'}
Return the branch name and the full commit sha.
`

const IMPL_SCHEMA = {
  type: 'object',
  properties: {
    branch: { type: 'string' },
    commit: { type: 'string' },
    summary: { type: 'string' },
    files_changed: { type: 'array', items: { type: 'string' } },
    tests_passed: { type: 'boolean' },
    test_summary: { type: 'string' },
    key_numbers: { type: 'string' },
    deviations: { type: 'array', items: { type: 'string' } },
    blocked: { type: 'array', items: { type: 'string' } },
  },
  required: ['branch', 'commit', 'summary', 'files_changed', 'tests_passed', 'test_summary', 'deviations', 'blocked'],
}

const REVIEW_SCHEMA = {
  type: 'object',
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          severity: { type: 'string', enum: ['blocker', 'major', 'minor', 'nit'] },
          file: { type: 'string' },
          line: { type: 'integer' },
          issue: { type: 'string' },
          evidence: { type: 'string' },
          suggested_fix: { type: 'string' },
        },
        required: ['severity', 'file', 'issue', 'evidence', 'suggested_fix'],
      },
    },
    tests_ran: { type: 'string' },
  },
  required: ['findings', 'tests_ran'],
}

const FIX_SCHEMA = {
  type: 'object',
  properties: {
    branch: { type: 'string' },
    commit: { type: 'string' },
    applied: { type: 'array', items: { type: 'string' } },
    rejected: {
      type: 'array',
      items: { type: 'object', properties: { issue: { type: 'string' }, reason: { type: 'string' } }, required: ['issue', 'reason'] },
    },
    tests_passed: { type: 'boolean' },
    test_summary: { type: 'string' },
  },
  required: ['branch', 'commit', 'applied', 'rejected', 'tests_passed', 'test_summary'],
}

const LENSES = [
  {
    key: 'spec',
    text: 'LENS: specification and protocol compliance. Check line by line that the diff implements EVERY requirement of the task spec below and of the corresponding item in docs/TZ_PROTOCOL_V2.md, and that it violates none of the CLAUDE.md rules (threshold never from test labels for anything reported as a result, oracle values clearly labelled, mean over seeds, controls intact, no gated files touched, no change to protocol parameters). Missing requirements are findings. Also check that numbers printed/saved are computed by the same functions as runs.csv (no silent divergence).',
  },
  {
    key: 'bugs',
    text: 'LENS: correctness bugs and test adequacy. Hunt for real bugs: off-by-one, segment boundaries, NaN/inf handling, empty inputs, dtype issues, pandas alignment/ordering bugs, wrong grouping, misuse of the resume path, merge-conflict risks with the parallel tasks. Create a detached worktree to run things: git worktree add --detach WT BRANCH (remove it at the end with git worktree remove --force WT). Run the new tests and the full suite there. Check that tests would FAIL if the implementation were wrong (temporarily break a line and re-run the relevant test, then restore). Report only real, evidenced problems.',
  },
]

const results = await pipeline(
  args.tasks,
  t => agent(
    common(t.branch) + '\n' + fill(t.spec),
    { label: 'impl:' + t.key, phase: 'Implement', schema: IMPL_SCHEMA, isolation: 'worktree', effort: 'high' }
  ),
  (impl, t) => {
    if (!impl) return null
    log(t.key + ': ' + impl.branch + ' @ ' + impl.commit + ' (tests ' + (impl.tests_passed ? 'green' : 'RED') + ')')
    return parallel(LENSES.map(l => () => agent(
      'You are an adversarial reviewer. Repository main checkout: ' + REPO + ' (run git commands there; branches are shared). ' +
      'Review the change on branch ' + impl.branch + ' relative to base ' + BASE + ': git -C ' + REPO + ' diff ' + BASE + '..' + impl.branch + '\n' +
      l.text.replace(/WT/g, SCRATCH + '/rev_' + t.key + '_' + l.key).replace('BRANCH', impl.branch) + '\n\n' +
      'Do not modify the branch. Do not touch reports/ or data/. Implementer summary: ' + impl.summary +
      '\nDeviations claimed: ' + JSON.stringify(impl.deviations) +
      '\n\nTASK SPEC GIVEN TO THE IMPLEMENTER:\n' + fill(t.spec) +
      '\nReturn findings with concrete evidence (file, line, what input breaks it). Severity: blocker = wrong numbers or rule violation; major = missing requirement or real bug; minor = robustness/clarity; nit = style.',
      { label: 'review:' + t.key + ':' + l.key, phase: 'Review', schema: REVIEW_SCHEMA, effort: 'high' }
    ))).then(reviews => ({ impl, reviews: reviews.filter(Boolean) }))
  },
  (rv, t) => {
    if (!rv) return null
    const findings = rv.reviews.flatMap(r => r.findings)
    log(t.key + ': ' + findings.length + ' review findings')
    const finalBranch = t.branch + '-final'
    return agent(
      common(finalBranch).replace('git checkout -b ' + finalBranch + ' ' + BASE, 'git checkout -b ' + finalBranch + ' ' + rv.impl.branch)
        .replace('HEAD is ' + BASE, 'HEAD is ' + rv.impl.commit) +
      '\nYou are the FIXER for task "' + t.key + '". ' +
      'Two independent reviewers produced the findings below. Verify EACH finding yourself against the code (reviewers can be wrong). ' +
      'Fix every real blocker/major/minor finding and cheap correct nits; reject false ones with a concrete reason. ' +
      'Keep fixes minimal and within the task scope and constraints. Commit early (a WIP commit is fine) so a crash does not lose the work. ' +
      'Re-run the new tests and then the full suite (must be green), then commit on ' + finalBranch + '. ' +
      'If there is nothing to fix, still create the branch and return its sha.\n\n' +
      'FINDINGS:\n' + JSON.stringify(findings, null, 1) + '\n\nTASK SPEC:\n' + fill(t.spec),
      { label: 'fix:' + t.key, phase: 'Fix', schema: FIX_SCHEMA, isolation: 'worktree', effort: 'high' }
    ).then(fix => ({ key: t.key, impl: rv.impl, findings, fix }))
  }
)

return results

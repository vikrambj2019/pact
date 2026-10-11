"""Live reviewer acceptance and rejection probe using the supplied portal source.

This reviews controlled examples; it is not a fresh full-pipeline benchmark.
"""
import argparse
import asyncio
import json
from pathlib import Path
from load_pipeline import parser, reviewer
from run_portal import LiveLLM


async def run(args):
    values = {}
    for line in Path(args.credentials).read_text().splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            k, v = line.split('=', 1)
            values[k.strip()] = v.strip().strip('\"\'')
    llm = LiveLLM(values.get('ANTHROPIC_API_KEY') or values.get('LLM_API_KEY'), args.model)
    units = parser._parse_teams(Path(__file__).with_name('portal.txt').read_text())
    for i, u in enumerate(units):
        u['id'] = f'fixture-{i:03d}'
    def source(fragment):
        return next(u for u in units if fragment in u['normalized_text'])
    fix = source('token fix started Wednesday')
    funding = source('send me the ask by Oct 30')
    grounded = {'actions': [{'text': 'Work on token fix for finding #3', 'owner_label': 'Dev Patel',
                            'owner_kind': 'named', 'status': 'committed',
                            'quote': fix['normalized_text'], 'source_unit_ids': [fix['id']]}]}
    cases = {
        'grounded_action': ('json', grounded),
        'invented_completion_deadline': ('json', {'actions': [{**grounded['actions'][0], 'due': 'Nov 6'}]}),
        'invented_funding_approval': ('json', {'claims': [{'text': 'Hal approved the $215k fallback budget',
                                    'quote': funding['normalized_text'], 'source_unit_ids': [funding['id']],
                                    'evidence_status': 'supported_within_scope'}]}),
        'invented_summary_deadline': ('summary', 'COMMITTED\nDev: finding #3 closed by Nov 6.'),
    }
    outcomes = {}
    for name, (kind, draft) in cases.items():
        try:
            if kind == 'json':
                report = await reviewer.review_extraction(draft, units, llm)
            else:
                report = await reviewer.review_summary(draft, grounded, units, llm)
            outcomes[name] = {'accepted': True, 'report': report}
        except reviewer.ReviewError as exc:
            outcomes[name] = {'accepted': False, 'reason': str(exc)}
        print(json.dumps({'case': name, 'accepted': outcomes[name]['accepted']}), flush=True)
    expected = {name: name == 'grounded_action' for name in cases}
    passed = all(outcomes[name]['accepted'] == expected[name] for name in cases)
    result = {'mode': 'fresh live semantic reviewer probes, not full extraction',
              'model': args.model, 'passed': passed, 'cases': outcomes}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    if not passed:
        raise RuntimeError('Reviewer acceptance/rejection probe failed; see saved report')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--credentials', required=True)
    p.add_argument('--model', default='claude-sonnet-5')
    p.add_argument('--output', default='validation/reviewer/probes.json')
    asyncio.run(run(p.parse_args()))

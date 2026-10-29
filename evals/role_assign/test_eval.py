"""评测框架自检：假模型驱动整条评测图，不访问网络、不花钱。"""

import json
import unittest

from langchain_core.messages import AIMessage

from core.role_agent.llm import LLMSettings
from core.role_agent.schemas import CastPlan, Character, ChunkLabels, SegmentLabel, VoiceCasting, VoiceMapping
from core.role_agent.skill import load_skill, parse_markdown_sections
from core.role_agent.segmenter import split_segments

from . import metrics
from .pipeline import Analysis, CostLedger, EvalPipeline, JudgeVerdict, SkillEdit
from .run import render_markdown

SETTINGS = LLMSettings(base_url='http://fake/v1', api_key='', model='fake', timeout=10, source='settings')


class FakeStructured:
    def __init__(self, fn):
        self.fn = fn

    def invoke(self, messages):
        message = AIMessage(content='', usage_metadata={'input_tokens': 100, 'output_tokens': 50, 'total_tokens': 150},
                            response_metadata={'token_usage': {'cost': 0.001}})
        return {'raw': message, 'parsed': self.fn(messages), 'parsing_error': None}


class FakeLLM:
    def __init__(self, responders):
        self.responders = responders

    def with_structured_output(self, schema, method='function_calling', include_raw=False):
        return FakeStructured(self.responders[schema.__name__])

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages):
        return AIMessage(content='', tool_calls=[{'name': 'Finish', 'args': {'summary': 'ok'}, 'id': 'f', 'type': 'tool_call'}])


def perfect_agent(case):
    """按金标准作答的假分角模型。"""
    roster = [Character(name=c['name'], aliases=c.get('aliases', [])) for c in case['roster']]
    segments = split_segments(case['text'])
    gold = metrics.match_gold(case, segments)['by_segment']

    def labels(messages):
        payload = json.loads(messages[-1].content)
        out = []
        for item in payload['segments']:
            g = gold.get(item['id'])
            if g is None:
                out.append(SegmentLabel(id=item['id'], category='旁白', speaker='旁白', confidence=1))
            else:
                out.append(SegmentLabel(id=item['id'], category=g['category'][0],
                                        speaker=(g['speaker'] or ['系统'])[0], confidence=1))
        return ChunkLabels(labels=out)

    return FakeLLM({
        'CastPlan': lambda m: CastPlan(characters=roster),
        'ChunkLabels': labels,
        'VoiceCasting': lambda m: VoiceCasting(mappings=[
            VoiceMapping(character=name, voice_config=voices[0]) for name, voices in case['casting'].items() if voices
        ]),
    })


def fake_judge():
    return FakeLLM({
        'JudgeVerdict': lambda m: JudgeVerdict(speaker_attribution=9, category=9, segmentation=8, roster=9,
                                               casting=8, overall=9, errors=[], comment='ok'),
        'Analysis': lambda m: Analysis(verdict='ok', orchestration='ok', token_efficiency='ok', weaknesses=['x'],
                                       skill_edits=[SkillEdit(section='executor', change='c', reason='r')]),
    })


class DatasetTest(unittest.TestCase):
    def test_gold_matches_each_segment_once(self):
        for case in metrics.load_cases():
            matched = metrics.match_gold(case, split_segments(case['text']))
            self.assertEqual(matched['unmatched'], [], case['id'])
            self.assertEqual(matched['ambiguous'], [], case['id'])

    def test_prompt_files_parse(self):
        skill = load_skill()
        self.assertTrue(all(skill.prompts[k] for k in ('planner', 'executor', 'reviewer', 'caster')))
        self.assertEqual(skill.params['chunk_max_items'], 40)
        self.assertEqual(load_skill(overrides={'low_confidence': 0.99}).params['low_confidence'], 0.99)
        meta, sections = parse_markdown_sections('---\nname: x\n---\n说明\n## a\nA\n## b\nB')
        self.assertEqual((meta['name'], sections), ('x', {'a': 'A', 'b': 'B'}))


class MetricsTest(unittest.TestCase):
    def setUp(self):
        self.case = next(c for c in metrics.load_cases() if c['id'] == 'bookstore')
        self.segments = split_segments(self.case['text'])
        gold = metrics.match_gold(self.case, self.segments)['by_segment']
        self.labels = {
            s['id']: ({'category': gold[s['id']]['category'][0], 'speaker': gold[s['id']]['speaker'][0]}
                      if s['id'] in gold else {'category': '旁白', 'speaker': '旁白'})
            for s in self.segments
        }

    def state(self, labels, **extra):
        return {'segments': self.segments, 'labels': labels, 'roster': [], 'usage': {}, **extra}

    def test_alias_counts_as_correct(self):
        labels = dict(self.labels)
        target = next(i for i, l in labels.items() if l['speaker'] == '陈老板')
        labels[target] = {'category': '对话', 'speaker': '老陈'}
        self.assertEqual(metrics.score_trial(self.case, self.state(labels))['label_acc'], 1.0)

    def test_review_fix_and_break_counted(self):
        good = dict(self.labels)
        ids = [i for i, l in good.items() if l['category'] == '对话'][:2]
        pre = dict(good)
        pre[ids[0]] = {'category': '旁白', 'speaker': '旁白'}          # 校对前错，校对后对
        post = dict(good)
        post[ids[1]] = {'category': '对话', 'speaker': '林晚' if good[ids[1]]['speaker'] != '林晚' else '我'}  # 改坏
        score = metrics.score_trial(self.case, self.state(post, pre_review_labels=pre, first_issues=[{'id': ids[0]}]))
        self.assertEqual((score['review']['fixed'], score['review']['broken']), (1, 1))
        self.assertEqual(score['validator']['precision'], 1.0)

    def test_stability(self):
        other = dict(self.labels)
        first = next(iter(other))
        other[first] = {'category': '对话', 'speaker': '林晚'}
        result = metrics.stability(self.case, [{'labels': self.labels}, {'labels': self.labels}, {'labels': other}])
        self.assertAlmostEqual(result['all_agree_rate'], (len(self.segments) - 1) / len(self.segments))
        self.assertEqual(len(result['unstable_segments']), 1)


class PipelineTest(unittest.TestCase):
    def run_pipeline(self, max_cost):
        cases = metrics.load_cases(only=['xiyou', 'teahouse'])
        voices = metrics.load_voices()
        ledger = CostLedger(max_cost)
        pipeline = EvalPipeline(
            SETTINGS, load_skill(), voices, 'fake', 'fake-judge', 2, ledger, {}, {c['id']: 0.002 for c in cases},
            log=lambda *_: None, agent_llm=None, judge_llm=fake_judge(),
        )
        # 每个用例用自己的“完美”假模型
        llms = {c['id']: perfect_agent(c) for c in cases}
        original = pipeline.run_trial

        def run_trial(payload):
            pipeline._agent_llm = llms[payload['case']['id']]
            return original(payload)

        pipeline.run_trial = run_trial
        return pipeline.run(cases, workers=1), ledger

    def test_end_to_end_with_fake_models(self):
        result, ledger = self.run_pipeline(1.0)
        overall = result['summary']['overall']
        self.assertEqual(overall['runs'], 4)
        self.assertEqual(overall['label_acc'], 1.0)
        self.assertEqual(overall['all_agree_rate'], 1.0)
        self.assertEqual(result['summary']['cases']['teahouse']['chunks'], 2)
        self.assertGreater(ledger.spent['agent'], 0)
        self.assertIn('judge', ledger.spent)
        self.assertEqual(result['analysis']['verdict'], 'ok')
        config = {'started': '-', 'elapsed_s': 1, 'skill': load_skill().info(), 'overrides': {}, 'model': 'fake',
                  'judge_model': 'fake', 'trials': 2}
        self.assertIn('指标看板', render_markdown(config, result['summary'], result['analysis']))

    def test_budget_guard_skips(self):
        result, ledger = self.run_pipeline(0.003)
        self.assertGreater(ledger.skipped, 0)
        self.assertLess(result['summary']['overall']['runs'], 4)


if __name__ == '__main__':
    unittest.main()

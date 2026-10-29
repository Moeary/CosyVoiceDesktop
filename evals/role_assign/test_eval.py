"""评测框架自检：假模型驱动整条评测图，不访问网络、不花钱。"""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage

from core.role_agent import RoleAgent, checks
from core.role_agent.llm import LLMSettings
from core.role_agent.schemas import CastPlan, Character, ChunkLabels, SegmentLabel, VoiceCasting, VoiceMapping
from core.role_agent.skill import load_skill, parse_markdown_sections
from core.role_agent.segmenter import split_segments

from . import compare, metrics
from .pipeline import Analysis, CostLedger, EvalPipeline, SkillEdit
from .run import render_markdown

SETTINGS = LLMSettings(base_url='http://fake/v1', api_key='', model='fake', timeout=10, source='settings')


class FakeStructured:
    def __init__(self, fn):
        self.fn = fn

    def invoke(self, messages):
        message = AIMessage(content='', usage_metadata={'input_tokens': 100, 'output_tokens': 50, 'total_tokens': 150},
                            response_metadata={'token_usage': {'cost': 0.001}})
        return {'raw': message, 'parsed': self.fn(messages), 'parsing_error': None}


def tool_call(name, **args):
    return {'name': name, 'args': args, 'id': f"{name}-{len(args)}", 'type': 'tool_call'}


class FakeLLM:
    """review_script：校对员每一步要发出的工具调用，用完后调用 Finish。"""

    def __init__(self, responders, review_script=None):
        self.responders = responders
        self.review_script = list(review_script or [])
        self.review_prompts = []

    def with_structured_output(self, schema, method='function_calling', include_raw=False):
        return FakeStructured(self.responders[schema.__name__])

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages):
        self.review_prompts.append(messages[1].content)
        calls = self.review_script.pop(0) if self.review_script else [tool_call('Finish', summary='ok')]
        return AIMessage(content='', tool_calls=calls)


def perfect_agent(case, wrong=None, review_script=None):
    """按金标准作答的假分角模型；wrong={片段 id: (类别, 说话人)} 可故意标错。"""
    roster = [Character(name=c['name'], aliases=c.get('aliases', [])) for c in case['roster']]
    segments = split_segments(case['text'], load_skill().params['speech_lead_verbs'])
    gold = metrics.match_gold(case, segments)['by_segment']

    def labels(messages):
        payload = json.loads(messages[-1].content)
        out = []
        for item in payload['segments']:
            g = gold.get(item['id'])
            if item['id'] in (wrong or {}):
                category, speaker = wrong[item['id']]
                out.append(SegmentLabel(id=item['id'], category=category, speaker=speaker, confidence=0.8))
            elif g is None:
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
    }, review_script)


def fake_analyst():
    return FakeLLM({
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
        self.assertIsInstance(skill.params['cast_system_speaker'], bool)
        v1 = load_skill(Path(skill.path).parent / 'history' / 'SKILL.v1.md')
        self.assertEqual((v1.version, v1.params['cast_system_speaker']), ('1', False))
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


class SelfRepairTest(unittest.TestCase):
    """确定性校验 + ReAct 校对员的自我修复闭环。"""

    def run_agent(self, case, overrides=None, **kwargs):
        voices = metrics.load_voices()
        agent = RoleAgent(SETTINGS, llm=perfect_agent(case, **kwargs), skill=load_skill(overrides=overrides))
        state = agent.run(case['text'], voices['voices'], voices['fallback'])
        return agent, state

    def test_cues_never_contradict_gold(self):
        verbs = load_skill().params['speech_lead_verbs']
        for case in metrics.load_cases():
            roster = [{'name': c['name'], 'aliases': c.get('aliases', [])} for c in case['roster']]
            segments = split_segments(case['text'], verbs)
            gold = metrics.match_gold(case, segments)['by_segment']
            for seg_id, (speaker, _) in checks.attribution_cues(segments, roster).items():
                g = gold.get(seg_id)
                if g and set(g['category']) & {'对话', '心理'}:
                    self.assertIn(speaker, g['speaker'], (case['id'], seg_id))

    def test_cue_mismatch_repaired_by_reviewer(self):
        case = next(c for c in metrics.load_cases() if c['id'] == 'village')
        segments = split_segments(case['text'], load_skill().params['speech_lead_verbs'])
        target = next(s['id'] for s in segments if '招去干啥' in s['text'])
        agent, state = self.run_agent(case, {'auto_apply_suggestions': False}, wrong={target: ('对话', '村民们')},
                                      review_script=[[tool_call('ApplySuggestions', ids=[target])]])

        issue = next(i for i in state['first_issues'] if i['id'] == target)
        self.assertEqual((issue['kind'], issue['suggest']['speaker']), ('cue_mismatch', '王二婶'))
        self.assertIn('建议=对话/王二婶', agent.llm.review_prompts[0])
        self.assertEqual(state['labels'][target]['speaker'], '王二婶')
        score = metrics.score_trial(case, {**state, 'usage': agent.usage})
        self.assertEqual((score['review']['fixed'], score['review']['broken']), (1, 0))
        self.assertEqual(score['validator']['by_kind']['cue_mismatch']['suggest_ok'], 1)
        self.assertEqual(score['label_acc'], 1.0)

    def test_suggestion_auto_applied_without_reviewer(self):
        case = next(c for c in metrics.load_cases() if c['id'] == 'village')
        segments = split_segments(case['text'], load_skill().params['speech_lead_verbs'])
        target = next(s['id'] for s in segments if '招去干啥' in s['text'])
        agent, state = self.run_agent(case, {'auto_apply_suggestions': True}, wrong={target: ('对话', '未识别角色')})

        self.assertEqual(agent.llm.review_prompts, [])  # 没有剩余疑点，不调用校对员
        self.assertEqual(state['labels'][target]['speaker'], '王二婶')
        self.assertTrue(any('自动修复' in t['message'] for t in state['trace']))
        score = metrics.score_trial(case, {**state, 'usage': agent.usage})
        self.assertEqual((score['review']['fixed'], score['review']['broken'], score['label_acc']), (1, 0, 1.0))

    def test_unused_characters_pruned(self):
        case = next(c for c in metrics.load_cases() if c['id'] == 'grandpa')
        _, state = self.run_agent(case)
        names = {c['name'] for c in state['roster']}
        self.assertNotIn('狐狸', names)  # 只在爷爷的故事里出现，没有自己的台词
        self.assertTrue({'爷爷', '小胖', '妹妹', '我'} <= names)

    def test_group_singular_flagged(self):
        roster = [{'name': '众宾客', 'aliases': [], 'gender': '群体'}]
        self.assertTrue(checks.is_group(roster[0]))
        self.assertTrue(checks.singular_line('“也不是我，我今天一天都在图书馆。”'))
        self.assertFalse(checks.singular_line('“我等即拜他为王！”'))


class PipelineTest(unittest.TestCase):
    def run_pipeline(self, max_cost):
        cases = metrics.load_cases(only=['xiyou', 'teahouse'])
        voices = metrics.load_voices()
        ledger = CostLedger(max_cost)
        pipeline = EvalPipeline(
            SETTINGS, load_skill(), voices, 'fake', 'fake-analyst', 2, ledger, {}, {c['id']: 0.002 for c in cases},
            log=lambda *_: None, agent_llm=None, analyst_llm=fake_analyst(),
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
        self.assertIn('analyst', ledger.spent)
        self.assertNotIn('judge', ledger.spent)
        self.assertEqual(result['analysis']['verdict'], 'ok')
        config = {'started': '-', 'elapsed_s': 1, 'skill': load_skill().info(), 'overrides': {}, 'model': 'fake',
                  'analyst_model': 'fake', 'trials': 2}
        markdown = render_markdown(config, result['summary'], result['analysis'])
        self.assertIn('指标看板', markdown)
        self.assertNotIn('裁判总分', markdown)
        self.assertIn('ReAct 自我修复', markdown)

        # 报告落盘后可被 compare 重新汇总（JSON 往返后片段 id 变成字符串）
        report = {'config': config, 'trials': result['trials']}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'report.json'
            path.write_text(json.dumps(report, ensure_ascii=False, default=str), encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(compare.main(['--base', str(path), '--candidate', str(path),
                                               '--group', '茶馆=teahouse', '--out', str(Path(tmp) / 'c.md')]), 0)
            text = (Path(tmp) / 'c.md').read_text(encoding='utf-8')
        self.assertIn('| 片段准确率 | 100.0% | 100.0% | = |', text)
        self.assertIn('## 总体 · 茶馆', text)

    def test_budget_guard_skips(self):
        result, ledger = self.run_pipeline(0.003)
        self.assertGreater(ledger.skipped, 0)
        self.assertLess(result['summary']['overall']['runs'], 4)


if __name__ == '__main__':
    unittest.main()

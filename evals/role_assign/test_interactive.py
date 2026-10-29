"""交互式分角：checkpoint 续跑、名单确认、按意见重规划、角色表。假模型驱动，不访问网络。"""

import json
import tempfile
import unittest
from pathlib import Path

import httpx
import numpy as np
import openai
import soundfile

from core.role_agent import RoleAgent, checks
from core.role_agent.graph import apply_roster_edits, mentioned_segments
from core.role_agent.role_table import (
    best_clip_per_speaker, build_role_table, match_characters, scan_voice_library, speaker_to_config,
)
from core.role_agent.schemas import Character, LabelUpdate, LibraryCasting, LibraryPick, RosterRevision
from core.role_agent.segmenter import split_segments
from core.role_agent.session import RoleAgentSession, open_checkpointer
from core.role_agent.skill import load_skill

from . import metrics
from .test_eval import SETTINGS, FakeLLM, perfect_agent


def village():
    case = next(c for c in metrics.load_cases() if c['id'] == 'village')
    segments = split_segments(case['text'], load_skill().params['speech_lead_verbs'])
    return case, segments


def seg_id(segments, needle):
    return next(s['id'] for s in segments if needle in s['text'])


def roster_from(messages, **changes):
    """按 replan 输入里的角色表原样返回，changes={角色名: {字段: 新值}}，值为 None 表示删除该角色。"""
    payload = json.loads(messages[-1].content)
    characters = []
    for item in payload['roster']:
        change = changes.get(item['name'], {})
        if change is None:
            continue
        fields = {k: item[k] for k in ('name', 'aliases', 'gender', 'age', 'traits', 'voice_style')}
        characters.append(Character(**{**fields, **change}))
    return characters


class CountingAgent:
    """在 perfect_agent 外面数 ChunkLabels 调用，并可让某一批第一次调用时断网。"""

    def __init__(self, case, fail_on_chunk_with=None, **kwargs):
        self.llm = perfect_agent(case, **kwargs)
        self.calls = []
        labels = self.llm.responders['ChunkLabels']
        self.fail_once = fail_on_chunk_with

        def counting(messages):
            ids = [item['id'] for item in json.loads(messages[-1].content)['segments']]
            if self.fail_once is not None and self.fail_once in ids:
                self.fail_once = None
                raise openai.APIConnectionError(request=httpx.Request('POST', 'http://fake/v1'))
            self.calls.append(ids)
            return labels(messages)

        self.llm.responders['ChunkLabels'] = counting


class SessionTest(unittest.TestCase):
    def session(self, counting, checkpointer=None, thread_id=None):
        agent = RoleAgent(SETTINGS, llm=counting.llm, skill=load_skill())
        return RoleAgentSession(agent, checkpointer, thread_id)

    def test_pause_for_roster_then_revise_and_approve(self):
        case, segments = village()
        target = seg_id(segments, '签……签了')
        counting = CountingAgent(case, wrong={target: ('对话', '李大爷')})
        session = self.session(counting)

        pending = session.start(case['text'])
        self.assertEqual(session.status(), 'review')
        lee = next(c for c in pending['roster'] if c['name'] == '李大爷')
        self.assertEqual(lee['lines'], 5)  # 4 句本人台词 + 被标错的 #28
        self.assertEqual(len(lee['samples']), 3)
        self.assertEqual(len(counting.calls), 2)

        counting.llm.responders['RosterRevision'] = lambda m: RosterRevision(
            characters=roster_from(m, 王二婶={'voice_style': '泼辣大嗓门'}),
            relabels=[LabelUpdate(id=target, category='对话', speaker='老赵')],
            reply='#28 改为赵德厚；王二婶语气改为泼辣大嗓门',
        )
        pending = session.resume({'action': 'revise', 'feedback': f'#{target} 是赵德厚说的；王二婶说话泼辣、嗓门大'})
        state = session.state
        self.assertEqual(state['labels'][target]['speaker'], '赵德厚')  # 别名 老赵 归一到规范名
        self.assertTrue(state['labels'][target]['pinned'])
        self.assertEqual(next(c for c in pending['roster'] if c['name'] == '王二婶')['voice_style'], '泼辣大嗓门')
        self.assertIn('#28', pending['reply'])
        self.assertEqual(len(counting.calls), 2)  # 只改名单和点名片段，不重标
        self.assertEqual(state['user_feedback'], [f'#{target} 是赵德厚说的；王二婶说话泼辣、嗓门大'])

        # 第二条意见：新增角色并复查王小军的台词 → 只重标受影响的批次，已钉住的标签不被覆盖
        counting.llm.responders['RosterRevision'] = lambda m: RosterRevision(
            characters=roster_from(m) + [Character(name='老张', gender='男')],
            recheck=['王小军'],
        )
        pending = session.resume({'action': 'revise', 'feedback': '漏了老张；王小军的台词再核对一下'})
        chunks = session.state['chunks']
        affected = {i for i, chunk in enumerate(chunks)
                    for s in chunk if session.state['labels'][s]['speaker'] == '王小军' or s == seg_id(segments, '老张')}
        self.assertEqual(len(counting.calls), 2 + len(affected))
        self.assertEqual(session.state['labels'][target]['speaker'], '赵德厚')
        self.assertTrue(any('重标' in t['message'] for t in session.state['trace']))
        self.assertEqual(session.state['user_feedback'][-1], '漏了老张；王小军的台词再核对一下')

        self.assertIsNone(session.resume({'action': 'approve', 'edits': [{'name': '李大爷', 'age': '老年'}]}))
        self.assertEqual(session.status(), 'done')
        self.assertEqual(next(c for c in session.state['roster'] if c['name'] == '李大爷')['age'], '老年')
        self.assertNotIn('老张', [c['name'] for c in session.state['roster']])  # 没有台词，收尾时移除

    def test_retry_resumes_from_checkpoint(self):
        case, segments = village()
        counting = CountingAgent(case, fail_on_chunk_with=segments[-1]['id'])
        session = self.session(counting)
        with self.assertRaises(openai.APIConnectionError):
            session.start(case['text'])
        self.assertEqual(session.status(), 'running')
        self.assertEqual(session.snapshot().next, ('label_chunk',))

        self.assertIsNotNone(session.retry())
        self.assertEqual(session.status(), 'review')
        self.assertEqual(len(counting.calls), 2)  # 第一批没有重做

    def test_sqlite_checkpoint_survives_restart(self):
        case, _ = village()
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            path = Path(tmp) / 'cp.sqlite'
            first = self.session(CountingAgent(case), open_checkpointer(path), 'doc-1')
            pending = first.start(case['text'])

            counting = CountingAgent(case)
            second = self.session(counting, open_checkpointer(path), 'doc-1')
            self.assertEqual(second.status(), 'review')
            self.assertEqual(second.pending(), pending)
            second.resume({'action': 'approve'})
            self.assertEqual(second.status(), 'done')
            self.assertEqual(counting.calls, [])
            first.checkpointer.conn.close()
            second.checkpointer.conn.close()

    def test_auto_replan_merges_discovered_character(self):
        case, segments = village()
        target = seg_id(segments, '我凭啥不说')
        llm = perfect_agent(case, wrong={target: ('对话', '王家媳妇')})
        llm.responders['RosterRevision'] = lambda m: RosterRevision(characters=roster_from(
            m, 王家媳妇=None, 王二婶={'aliases': ['王家媳妇'], 'voice_style': '泼辣'}))
        agent = RoleAgent(SETTINGS, llm=llm, skill=load_skill())
        state = agent.run(case['text'], [], '')

        self.assertEqual(state['labels'][target]['speaker'], '王二婶')
        self.assertEqual(state['auto_replans'], 1)
        self.assertNotIn('王家媳妇', [c['name'] for c in state['roster']])
        self.assertNotIn('voice_mapping', state)  # 没有音色配置时不选角
        self.assertTrue(any('合并 王家媳妇→王二婶' in t['message'] for t in state['trace']))

    def test_auto_replan_off_for_old_skill(self):
        case, segments = village()
        target = seg_id(segments, '我凭啥不说')
        llm = perfect_agent(case, wrong={target: ('对话', '王家媳妇')})
        skill = load_skill(Path(load_skill().path).parent / 'history' / 'SKILL.v2.2.md')
        state = RoleAgent(SETTINGS, llm=llm, skill=skill).run(case['text'], [], '')
        self.assertEqual(state['labels'][target]['speaker'], '王家媳妇')
        self.assertNotIn('auto_replans', state)

    def test_mentioned_segments(self):
        _, segments = village()
        feedback = '「签签了」是赵德厚说的；#3 和 “安静！安静！” 也不对；「的」太短；#999 不存在'
        self.assertEqual(mentioned_segments(feedback, segments),
                         sorted({3, seg_id(segments, '签……签了'), seg_id(segments, '安静！安静！')}))

    def test_lead_subject_uses_last_clause(self):
        # 实测发现：取句首主语会把外婆的话判给“我”、把孩子们齐声喊的话判给王婶，且会被自动采纳
        roster = [{'name': n, 'aliases': a} for n, a in [('我', ['小满']), ('外婆', ['奶奶']), ('王婶', []), ('两个孩子', [])]]
        self.assertEqual(checks.lead_subject('我把最后一摞碗放进橱柜，外婆在堂屋里喊：', roster), '外婆')
        self.assertEqual(checks.lead_subject('王婶身后还跟着她家的两个孩子，挤在门口直嚷嚷：', roster), '')
        self.assertEqual(checks.lead_subject('外婆拄着拐杖站起来，看了我一眼：', roster), '外婆')

    def test_roster_edits(self):
        roster = [{'name': '蒙面人', 'aliases': ['黑衣人']}, {'name': '柳随风', 'aliases': []}, {'name': '甲', 'aliases': []}]
        labels = {1: {'category': '对话', 'speaker': '蒙面人'}, 2: {'category': '对话', 'speaker': '甲'}}
        roster, labels, changes = apply_roster_edits(roster, labels, [
            {'name': '蒙面人', 'new_name': '柳随风'}, {'name': '甲', 'new_name': '老甲', 'gender': '男'},
        ])
        self.assertEqual([c['name'] for c in roster], ['柳随风', '老甲'])
        self.assertEqual(roster[0]['aliases'], ['蒙面人', '黑衣人'])
        self.assertEqual((labels[1]['speaker'], labels[2]['speaker'], roster[1]['gender']), ('柳随风', '老甲', '男'))
        self.assertEqual(len(changes), 3)


class RoleTableTest(unittest.TestCase):
    def write_wav(self, path: Path, seconds: float, text: str = ''):
        path.parent.mkdir(parents=True, exist_ok=True)
        soundfile.write(str(path), np.zeros(int(16000 * seconds), dtype='float32'), 16000)
        if text:
            path.with_suffix('.txt').write_text(text, encoding='utf-8')

    def test_scan_match_and_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_wav(root / '王二婶' / 'long.wav', 30, '太长的一段')
            self.write_wav(root / '王二婶' / 'good.wav', 5, '我家小军是念过大专的')
            self.write_wav(root / '女' / '老年' / '张婶' / 'a.wav', 6)
            self.write_wav(root / '李大爷_你自己家那三亩签了没有.wav', 4)
            self.write_wav(root / 'c.wav', 5)
            (root / 'voices.list').write_text('D:/elsewhere/c.wav|赵书记|ZH|人不齐也开等不起了\n', encoding='utf-8')

            voices = best_clip_per_speaker(scan_voice_library(str(root)))
            self.assertEqual(set(voices), {'王二婶', '张婶', '李大爷', '赵书记'})
            self.assertTrue(voices['王二婶'].path.endswith('good.wav'))
            self.assertEqual(voices['李大爷'].text, '你自己家那三亩签了没有')
            self.assertEqual(voices['赵书记'].text, '人不齐也开等不起了')
            self.assertEqual(voices['张婶'].hints, ['女', '老年', '张婶'])

            characters = [
                {'name': '王二婶', 'gender': '女', 'voice_style': '泼辣大嗓门', 'lines': 6},
                {'name': '李大爷', 'gender': '男', 'age': '老年', 'lines': 3},
                {'name': '赵德厚', 'aliases': ['赵书记'], 'gender': '男', 'lines': 5},
                {'name': '王老太', 'gender': '女', 'age': '老年', 'lines': 2},
                {'name': '林经理', 'gender': '男', 'lines': 4},
            ]
            llm = FakeLLM({'LibraryCasting': lambda m: LibraryCasting(picks=[
                LibraryPick(character='王老太', voice_id='张婶', reason='老年女声'),
                LibraryPick(character='林经理', voice_id='不存在', reason='x'),
            ])})
            matches = match_characters(characters, voices, llm=llm, usage={})
            self.assertEqual({k: v['speaker'] for k, v in matches.items()},
                             {'王二婶': '王二婶', '李大爷': '李大爷', '赵德厚': '赵书记', '王老太': '张婶'})

            rows = build_role_table(characters, matches, voices, narrator_config='旁白音色')
            by_name = {row['name']: row for row in rows}
            self.assertNotIn('旁白', by_name)
            self.assertEqual((by_name['王二婶']['mode'], by_name['王二婶']['instruct_text']),
                             ('指令控制', '请用泼辣大嗓门的语气说这句话。'))
            self.assertEqual(by_name['李大爷']['mode'], '零样本复制')
            self.assertEqual(by_name['王老太']['mode'], '精细控制')
            self.assertEqual((by_name['林经理']['prompt_audio'], by_name['林经理']['source']), ('', 'manual'))
            mapping = speaker_to_config(rows, '旁白音色')
            self.assertEqual((mapping['旁白'], mapping['系统'], mapping['赵德厚']), ('旁白音色', '旁白音色', '赵德厚'))

            rows = build_role_table(characters[:1], narrator_config='', include_system=True)
            self.assertEqual([row['name'] for row in rows], ['旁白', '系统', '王二婶'])


if __name__ == '__main__':
    unittest.main()

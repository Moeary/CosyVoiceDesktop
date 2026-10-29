"""评测编排（LangGraph）：

    prepare ──Send×(用例×次数)──▶ run_trial ──▶ collect ──Send×成功次数──▶ judge_trial ──▶ aggregate ──▶ analyst

run_trial 运行分角智能体并做确定性评分；judge_trial 由裁判模型独立打分；aggregate 汇总准确率、稳定性、
ReAct 收益、token 效率；analyst 是分析智能体，读汇总数据给出结论和 Skill 修改建议。
全程由 CostLedger 记账，超出预算的任务直接跳过。
"""

import json
import operator
import statistics
import threading
import time
import traceback
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from core.role_agent import RoleAgent
from core.role_agent.llm import LLMSettings, build_chat_model, invoke_structured
from core.role_agent.skill import Skill, parse_markdown_sections

from . import metrics

PROMPTS = parse_markdown_sections((Path(__file__).resolve().parent / 'prompts.md').read_text(encoding='utf-8'))[1]


# ---------- 结构化输出 ----------

class JudgeError(BaseModel):
    segment_id: int
    kind: Literal['speaker', 'category', 'segmentation', 'roster', 'casting'] = Field(
        description="speaker=说话人错，category=类别错，segmentation=断句不当，roster/casting=角色表或音色问题")
    problem: str
    suggestion: str = ''


class JudgeVerdict(BaseModel):
    speaker_attribution: int = Field(ge=1, le=10)
    category: int = Field(ge=1, le=10)
    segmentation: int = Field(ge=1, le=10)
    roster: int = Field(ge=1, le=10)
    casting: int = Field(ge=1, le=10)
    overall: int = Field(ge=1, le=10)
    errors: List[JudgeError] = Field(description="确定的错误，最多 10 条")
    comment: str = ''


class SkillEdit(BaseModel):
    section: Literal['planner', 'executor', 'reviewer', 'caster', 'params', 'segmenter', 'code']
    change: str
    reason: str
    expected_effect: str = ''


class Analysis(BaseModel):
    verdict: str
    orchestration: str
    token_efficiency: str
    weaknesses: List[str]
    skill_edits: List[SkillEdit]


JUDGE_DIMENSIONS = ('speaker_attribution', 'category', 'segmentation', 'roster', 'casting', 'overall')


# ---------- 预算 ----------

class CostLedger:
    """线程安全的花费账本。每个任务开始前按预估额度占位，结束后按实际花费结算。"""

    def __init__(self, max_cost: float):
        self.max_cost = max_cost
        self.spent: Dict[str, float] = {}
        self.reserved = 0.0
        self.skipped = 0
        self._lock = threading.Lock()

    @property
    def total(self) -> float:
        return sum(self.spent.values())

    def reserve(self, estimate: float) -> bool:
        with self._lock:
            if self.total + self.reserved + estimate > self.max_cost:
                self.skipped += 1
                return False
            self.reserved += estimate
            return True

    def settle(self, bucket: str, estimate: float, actual: float):
        with self._lock:
            self.reserved -= estimate
            self.spent[bucket] = self.spent.get(bucket, 0.0) + actual


def usage_cost(usage: dict, price: Optional[dict]) -> float:
    """优先用 OpenRouter 回传的实际花费；没有时按单价估算。"""
    if usage.get('cost'):
        return float(usage['cost'])
    if not price:
        return 0.0
    return usage.get('input_tokens', 0) * price['input'] + usage.get('output_tokens', 0) * price['output']


# ---------- 状态 ----------

class EvalState(TypedDict, total=False):
    cases: List[dict]
    trials: Annotated[List[dict], operator.add]
    judgments: Annotated[List[dict], operator.add]
    summary: dict
    analysis: dict


class EvalPipeline:
    def __init__(
        self,
        settings: LLMSettings,
        skill: Skill,
        voices: dict,
        model: str,
        judge_model: str,
        trials: int,
        ledger: CostLedger,
        prices: Dict[str, dict],
        estimates: Dict[str, float],
        judge: bool = True,
        analyst: bool = True,
        log=print,
        agent_llm=None,
        judge_llm=None,
    ):
        self.skill = skill
        self.voices = voices
        self.model = model
        self.judge_model = judge_model
        self.trials = trials
        self.ledger = ledger
        self.prices = prices
        self.estimates = estimates
        self.use_judge = judge
        self.use_analyst = analyst
        self.log = log
        self.agent_settings = replace(settings, model=model, timeout=max(settings.timeout, 180))
        self.judge_settings = replace(settings, model=judge_model, timeout=max(settings.timeout, 180))
        # 测试时可注入假模型
        self._agent_llm = agent_llm
        self._judge_llm = judge_llm

    # ---------- 节点 ----------

    def prepare(self, state: EvalState) -> dict:
        self.log(f"评测 {len(state['cases'])} 个用例 × {self.trials} 次，Skill {self.skill.label}，预算 ${self.ledger.max_cost:.2f}")
        return {}

    def dispatch_trials(self, state: EvalState):
        return [
            Send('run_trial', {'case': case, 'trial': index})
            for case in state['cases']
            for index in range(self.trials)
        ]

    def run_trial(self, payload: dict) -> dict:
        case, index = payload['case'], payload['trial']
        estimate = self.estimates.get(case['id'], 0.01)
        record = {'case_id': case['id'], 'trial': index}
        if not self.ledger.reserve(estimate):
            self.log(f"  [跳过] {case['id']}#{index + 1}：预算不足")
            return {'trials': [{**record, 'skipped': True}]}

        agent = None
        started = time.perf_counter()
        try:
            agent = RoleAgent(self.agent_settings, skill=self.skill, llm=self._agent_llm)
            state = agent.run(case['text'], self.voices['voices'], self.voices['fallback'])
            elapsed = time.perf_counter() - started
            score = metrics.score_trial(case, {**state, 'usage': agent.usage})
            cost = usage_cost(agent.usage, self.prices.get(self.model))
            score['usage']['cost'] = cost
            record.update({
                'elapsed': elapsed,
                'score': score,
                'efficiency': metrics.efficiency(score, len(case['text'])),
                'state': {
                    'segments': [{'id': s['id'], 'text': s['text']} for s in state['segments']],
                    'labels': state['labels'],
                    'roster': state.get('roster', []),
                    'voice_mapping': state.get('voice_mapping', {}),
                    'first_issues': state.get('first_issues', []),
                },
            })
            self.log(f"  [完成] {case['id']}#{index + 1}：准确率 {score['label_acc']:.0%}，"
                     f"{agent.usage.get('calls', 0)} 次调用，${cost:.4f}，{elapsed:.0f}s")
        except Exception as e:
            usage = getattr(agent, 'usage', {})
            cost = usage_cost(usage, self.prices.get(self.model))
            record.update({'error': f"{e}", 'traceback': traceback.format_exc(limit=3), 'usage': usage})
            self.log(f"  [失败] {case['id']}#{index + 1}：{e}")
        self.ledger.settle('agent', estimate, cost)
        return {'trials': [record]}

    def collect(self, state: EvalState) -> dict:
        done = [t for t in state['trials'] if 'score' in t]
        self.log(f"分角运行结束：成功 {len(done)}/{len(state['trials'])}，累计 ${self.ledger.total:.4f}")
        return {}

    def dispatch_judges(self, state: EvalState):
        done = [t for t in state['trials'] if 'score' in t]
        if not self.use_judge or not done:
            return 'aggregate'
        cases = {case['id']: case for case in state['cases']}
        return [Send('judge_trial', {'trial': t, 'case': cases[t['case_id']]}) for t in done]

    def judge_trial(self, payload: dict) -> dict:
        trial, case = payload['trial'], payload['case']
        estimate = self.estimates.get(f"judge:{case['id']}", 0.005)
        key = {'case_id': trial['case_id'], 'trial': trial['trial']}
        if not self.ledger.reserve(estimate):
            return {'judgments': [{**key, 'skipped': True}]}

        llm = self._judge_llm or build_chat_model(self.judge_settings, temperature=0, max_tokens=4096)
        usage: dict = {}
        cost = 0.0
        try:
            verdict = invoke_structured(llm, JudgeVerdict, [
                SystemMessage(content=PROMPTS['judge']),
                HumanMessage(content=self._judge_input(trial['state'])),
            ], usage, 'judge')
            cost = usage_cost(usage, self.prices.get(self.judge_model))
            result = {**key, 'verdict': verdict.model_dump(), 'usage': {**usage, 'cost': cost}}
            self.log(f"  [裁判] {case['id']}#{trial['trial'] + 1}：overall {verdict.overall}/10，${cost:.4f}")
        except Exception as e:
            cost = usage_cost(usage, self.prices.get(self.judge_model))
            result = {**key, 'error': str(e)}
            self.log(f"  [裁判失败] {case['id']}#{trial['trial'] + 1}：{e}")
        self.ledger.settle('judge', estimate, cost)
        return {'judgments': [result]}

    def _judge_input(self, state: dict) -> str:
        mapping = state['voice_mapping']
        lines = []
        for segment in state['segments']:
            label = state['labels'][segment['id']]
            voice = (mapping.get(label['speaker']) or {}).get('voice') or '未指定'
            lines.append(f"#{segment['id']} [{label['category']}/{label['speaker']} → {voice}] {segment['text']}")
        roster = '；'.join(
            f"{c['name']}（{c.get('gender', '未知')}/{c.get('age', '未知')}，又称 {'、'.join(c.get('aliases', [])) or '无'}）"
            for c in state['roster']
        )
        voices = '；'.join(f"{v['name']}：{v.get('instruct_text', '')}" for v in self.voices['voices'])
        return f"角色表：{roster}\n音色库：{voices}\n旁白默认音色：{self.voices['fallback']}\n\n台本：\n" + '\n'.join(lines)

    def aggregate(self, state: EvalState) -> dict:
        summary = summarize(state['cases'], state['trials'], state.get('judgments', []), self.trials)
        summary['cost'] = {k: round(v, 6) for k, v in self.ledger.spent.items()}
        summary['cost']['total'] = round(self.ledger.total, 6)
        summary['skipped_for_budget'] = self.ledger.skipped
        return {'summary': summary}

    def analyst(self, state: EvalState) -> dict:
        if not self.use_analyst:
            return {}
        estimate = 0.01
        if not self.ledger.reserve(estimate):
            return {'analysis': {'error': '预算不足，跳过分析'}}
        llm = self._judge_llm or build_chat_model(self.judge_settings, temperature=0.2, max_tokens=6000)
        usage: dict = {}
        cost = 0.0
        try:
            analysis = invoke_structured(llm, Analysis, [
                SystemMessage(content=PROMPTS['analyst']),
                HumanMessage(content=self._analyst_input(state['summary'])),
            ], usage, 'analyst')
            cost = usage_cost(usage, self.prices.get(self.judge_model))
            result = analysis.model_dump()
            self.log(f"分析智能体完成，${cost:.4f}")
        except Exception as e:
            cost = usage_cost(usage, self.prices.get(self.judge_model))
            result = {'error': str(e)}
        self.ledger.settle('analyst', estimate, cost)
        summary = dict(state['summary'])
        summary['cost'] = {**summary['cost'], 'analyst': round(cost, 6), 'total': round(self.ledger.total, 6)}
        return {'analysis': result, 'summary': summary}

    def _analyst_input(self, summary: dict) -> str:
        compact = {k: summary[k] for k in ('overall', 'judge_reliability', 'cases')}
        prompts = '\n\n'.join(f"## {name}\n{text}" for name, text in self.skill.prompts.items())
        return (
            f"Skill：{self.skill.label}\n参数：{json.dumps(self.skill.params, ensure_ascii=False)}\n\n"
            f"评测汇总：\n{json.dumps(compact, ensure_ascii=False, default=str)}\n\n当前 Skill 提示词：\n{prompts}"
        )

    def build(self):
        graph = StateGraph(EvalState)
        graph.add_node('prepare', self.prepare)
        graph.add_node('run_trial', self.run_trial)
        graph.add_node('collect', self.collect)
        graph.add_node('judge_trial', self.judge_trial)
        graph.add_node('aggregate', self.aggregate)
        graph.add_node('analyst', self.analyst)
        graph.add_edge(START, 'prepare')
        graph.add_conditional_edges('prepare', self.dispatch_trials, ['run_trial'])
        graph.add_edge('run_trial', 'collect')
        graph.add_conditional_edges('collect', self.dispatch_judges, ['judge_trial', 'aggregate'])
        graph.add_edge('judge_trial', 'aggregate')
        graph.add_edge('aggregate', 'analyst')
        graph.add_edge('analyst', END)
        return graph.compile()

    def run(self, cases: List[dict], workers: int = 3) -> EvalState:
        return self.build().invoke({'cases': cases}, config={'max_concurrency': workers, 'recursion_limit': 100})


# ---------- 汇总 ----------

def _mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def _pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    return statistics.correlation(xs, ys)


def summarize(cases: List[dict], trials: List[dict], judgments: List[dict], expected: int) -> dict:
    judged = {(j['case_id'], j['trial']): j for j in judgments if 'verdict' in j}
    per_case = {}
    all_ok = []
    for case in cases:
        runs = sorted((t for t in trials if t['case_id'] == case['id'] and 'score' in t), key=lambda t: t['trial'])
        failed = [t for t in trials if t['case_id'] == case['id'] and 'error' in t]
        all_ok.extend(runs)
        if not runs:
            per_case[case['id']] = {'title': case['title'], 'runs': 0, 'failed': len(failed),
                                    'errors': [t['error'] for t in failed]}
            continue
        scores = [t['score'] for t in runs]
        effs = [t['efficiency'] for t in runs]
        verdicts = [judged[(case['id'], t['trial'])]['verdict'] for t in runs if (case['id'], t['trial']) in judged]

        wrong_counter = Counter(f"#{w['id']} {w['text']} | 得到 {w['got']} | 应为 {w['want']}" for s in scores for w in s['wrong'])
        per_case[case['id']] = {
            'title': case['title'],
            'runs': len(runs),
            'failed': len(failed),
            'segments': scores[0]['segments'],
            'chunks': scores[0]['chunks'],
            'label_acc': metrics.mean_std([s['label_acc'] for s in scores]),
            'speaker_acc': metrics.mean_std([s['speaker_acc'] for s in scores]),
            'category_acc': metrics.mean_std([s['category_acc'] for s in scores]),
            'narration_false_positive': _mean([s['narration_false_positive'] for s in scores]),
            'pre_review_label_acc': _mean([s['pre_review_label_acc'] for s in scores]),
            'review': {
                'trigger_rate': _mean([float(s['review']['triggered']) for s in scores]),
                'fixed': sum(s['review']['fixed'] for s in scores),
                'broken': sum(s['review']['broken'] for s in scores),
                'changed': sum(s['review']['changed'] for s in scores),
                'net_gain': _mean([s['review']['net_gain'] for s in scores]),
            },
            'validator': {
                'flagged': _mean([s['validator']['flagged'] for s in scores]),
                'precision': _mean([s['validator']['precision'] for s in scores]),
                'recall': _mean([s['validator']['recall'] for s in scores]),
            },
            'roster': {
                'recall': _mean([s['roster']['recall'] for s in scores]),
                'precision': _mean([s['roster']['precision'] for s in scores]),
                'split': sorted({x for s in scores for x in s['roster']['split']}),
                'extra': sorted({x for s in scores for x in s['roster']['extra']}),
            },
            'casting_acc': _mean([s['casting_acc'] for s in scores]),
            'casting_wrong': sorted({x for s in scores for x in s['casting_wrong']}),
            'stability': metrics.stability(case, [
                {'labels': t['state']['labels'], 'roster': t['state']['roster']} for t in runs
            ]),
            'efficiency': {
                'calls': _mean([s['usage'].get('calls', 0) for s in scores]),
                'input_tokens': _mean([s['usage'].get('input_tokens', 0) for s in scores]),
                'output_tokens': _mean([s['usage'].get('output_tokens', 0) for s in scores]),
                'tokens': metrics.mean_std([e['tokens'] for e in effs]),
                'tokens_per_segment': _mean([e['tokens_per_segment'] for e in effs]),
                'cost': metrics.mean_std([s['usage'].get('cost', 0) for s in scores]),
                'cost_per_kchar': _mean([e['cost_per_kchar'] for e in effs]),
                'reasoning_share': _mean([e['reasoning_share'] for e in effs]),
                'cache_hit_share': _mean([e['cache_hit_share'] for e in effs]),
                'structured_fallbacks': sum(s['usage'].get('structured_fallbacks', 0) for s in scores),
                'latency_s': metrics.mean_std([t['elapsed'] for t in runs]),
                'node_token_share': {
                    node: _mean([e['node_token_share'].get(node, 0) for e in effs])
                    for node in sorted({n for e in effs for n in e['node_token_share']})
                },
            },
            'judge': {dim: _mean([v[dim] for v in verdicts]) for dim in JUDGE_DIMENSIONS} if verdicts else {},
            'judge_comments': [v['comment'] for v in verdicts if v.get('comment')],
            'top_wrong': [f"{text}（{n}/{len(runs)} 次）" for text, n in wrong_counter.most_common(8)],
            'gold_unmatched': scores[0]['gold_unmatched'],
        }

    # 裁判可信度：裁判总分与金标准准确率的相关性；裁判指出的错误有多少确实错了
    xs, ys, judge_hits, judge_total = [], [], 0, 0
    for t in all_ok:
        j = judged.get((t['case_id'], t['trial']))
        if not j:
            continue
        xs.append(j['verdict']['overall'])
        ys.append(t['score']['label_acc'])
        wrong_ids = {w['id'] for w in t['score']['wrong']}
        for err in j['verdict']['errors']:
            # 只有说话人/类别类错误能和金标准对照
            if err.get('kind') not in ('speaker', 'category'):
                continue
            judge_total += 1
            judge_hits += err['segment_id'] in wrong_ids

    ok_cases = [c for c in per_case.values() if c.get('runs')]
    total_cost_agent = sum(t['score']['usage'].get('cost', 0) for t in all_ok)
    total_chars = sum(len(c['text']) for c in cases for t in all_ok if t['case_id'] == c['id'])
    overall = {
        'runs': len(all_ok),
        'expected_runs': len(cases) * expected,
        'label_acc': _mean([c['label_acc']['mean'] for c in ok_cases]),
        'speaker_acc': _mean([c['speaker_acc']['mean'] for c in ok_cases]),
        'category_acc': _mean([c['category_acc']['mean'] for c in ok_cases]),
        'pre_review_label_acc': _mean([c['pre_review_label_acc'] for c in ok_cases]),
        'roster_recall': _mean([c['roster']['recall'] for c in ok_cases]),
        'roster_precision': _mean([c['roster']['precision'] for c in ok_cases]),
        'casting_acc': _mean([c['casting_acc'] for c in ok_cases]),
        'all_agree_rate': _mean([c['stability'].get('all_agree_rate') for c in ok_cases]),
        'pairwise_agreement': _mean([c['stability'].get('pairwise_agreement') for c in ok_cases]),
        'roster_jaccard': _mean([c['stability'].get('roster_jaccard') for c in ok_cases]),
        'raw_roster_jaccard': _mean([c['stability'].get('raw_roster_jaccard') for c in ok_cases]),
        'naming_consistency': _mean([c['stability'].get('naming_consistency') for c in ok_cases]),
        'label_acc_std': _mean([c['label_acc']['std'] for c in ok_cases]),
        'failed_runs': sum(c['failed'] for c in per_case.values()),
        'structured_fallbacks': sum(c['efficiency']['structured_fallbacks'] for c in ok_cases),
        'cost_cv': _mean([(c['efficiency']['cost']['std'] / c['efficiency']['cost']['mean'])
                          if c['efficiency']['cost']['mean'] else None for c in ok_cases]),
        'latency_cv': _mean([(c['efficiency']['latency_s']['std'] / c['efficiency']['latency_s']['mean'])
                             if c['efficiency']['latency_s']['mean'] else None for c in ok_cases]),
        'review_trigger_rate': _mean([c['review']['trigger_rate'] for c in ok_cases]),
        'review_fixed': sum(c['review']['fixed'] for c in ok_cases),
        'review_broken': sum(c['review']['broken'] for c in ok_cases),
        'validator_precision': _mean([c['validator']['precision'] for c in ok_cases]),
        'validator_recall': _mean([c['validator']['recall'] for c in ok_cases]),
        'cost_per_run': total_cost_agent / len(all_ok) if all_ok else None,
        'cost_per_kchar': total_cost_agent / total_chars * 1000 if total_chars else None,
        'tokens_per_segment': _mean([c['efficiency']['tokens_per_segment'] for c in ok_cases]),
        'reasoning_share': _mean([c['efficiency']['reasoning_share'] for c in ok_cases]),
        'latency_s': _mean([c['efficiency']['latency_s']['mean'] for c in ok_cases]),
        'judge_overall': _mean([c['judge'].get('overall') for c in ok_cases if c['judge']]),
    }
    return {
        'overall': overall,
        'cases': per_case,
        'judge_reliability': {
            'pearson_overall_vs_label_acc': _pearson(xs, ys),
            'error_precision': judge_hits / judge_total if judge_total else None,
            'errors_reported': judge_total,
        },
    }

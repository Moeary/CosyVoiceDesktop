"""确定性评分：对照金标准算准确率、校对净收益、校验命中率、角色表与选角质量，以及多次运行的稳定性。

全部是纯函数，不访问网络。
"""

import statistics
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

import yaml

DATASET_DIR = Path(__file__).resolve().parent / 'dataset'
NARRATION = '旁白'


# ---------- 数据集 ----------

def _as_list(value) -> List[str]:
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def load_voices(dataset_dir: Path = DATASET_DIR) -> dict:
    return yaml.safe_load((dataset_dir / 'voices.yaml').read_text(encoding='utf-8'))


def load_cases(dataset_dir: Path = DATASET_DIR, only: Optional[List[str]] = None) -> List[dict]:
    cases = []
    for path in sorted(dataset_dir.glob('[0-9]*.yaml')):
        case = yaml.safe_load(path.read_text(encoding='utf-8'))
        if only and case['id'] not in only:
            continue
        for item in case['gold']:
            item['category'] = _as_list(item['category'])
            item['speaker'] = _as_list(item.get('speaker'))
        cases.append(case)
    return cases


def gold_names(case: dict) -> Dict[str, set]:
    """金标准角色 → 该角色所有可接受的称呼。"""
    return {c['name']: {c['name'], *c.get('aliases', [])} for c in case['roster']}


def match_gold(case: dict, segments: List[dict]) -> dict:
    """把 gold 条目按子串挂到片段上。返回 {seg_id: gold_item}，以及未命中 / 多命中的条目。"""
    by_segment, unmatched, ambiguous = {}, [], []
    for item in case['gold']:
        hits = [s['id'] for s in segments if item['match'] in s['text']]
        if not hits:
            unmatched.append(item['match'])
            continue
        if len(hits) > 1:
            ambiguous.append(item['match'])
        for seg_id in hits:
            by_segment.setdefault(seg_id, item)
    return {'by_segment': by_segment, 'unmatched': unmatched, 'ambiguous': ambiguous}


# ---------- 单次运行 ----------

def canonical_speaker(speaker: str, case: dict, roster: List[dict]) -> str:
    """把模型给出的说话人映射到金标准角色名；映射不上时原样返回。"""
    names = gold_names(case)
    candidates = {speaker}
    for entry in roster or []:
        if entry.get('name') == speaker:
            candidates |= set(entry.get('aliases', []))
    for gold, accepted in names.items():
        if candidates & accepted:
            return gold
    return speaker


def label_correct(label: dict, gold: Optional[dict], case: dict, roster: List[dict]) -> bool:
    if gold is None:
        return label['category'] == NARRATION
    if label['category'] not in gold['category']:
        return False
    if label['category'] == NARRATION or not gold['speaker']:
        return True
    return canonical_speaker(label['speaker'], case, roster) in gold['speaker']


def _accuracy(labels: Dict[int, dict], segments, matched, case, roster) -> dict:
    total = correct = dlg_total = dlg_correct = narr_total = narr_fp = cat_correct = 0
    wrong = []
    for segment in segments:
        seg_id = segment['id']
        label = labels.get(seg_id) or {'category': NARRATION, 'speaker': NARRATION}
        gold = matched.get(seg_id)
        ok = label_correct(label, gold, case, roster)
        total += 1
        correct += ok
        cat_correct += label['category'] in (gold['category'] if gold else [NARRATION])
        if gold is None:
            narr_total += 1
            narr_fp += label['category'] != NARRATION
        elif set(gold['category']) & {'对话', '心理'}:
            dlg_total += 1
            dlg_correct += ok
        if not ok:
            wrong.append({
                'id': seg_id,
                'text': segment['text'][:40],
                'got': f"{label['category']}/{label['speaker']}",
                'want': (f"{'|'.join(gold['category'])}/{'|'.join(gold['speaker']) or '-'}" if gold else '旁白'),
            })
    return {
        'label_acc': correct / total if total else 0.0,
        'category_acc': cat_correct / total if total else 0.0,
        'speaker_acc': dlg_correct / dlg_total if dlg_total else 1.0,
        'narration_false_positive': narr_fp / narr_total if narr_total else 0.0,
        'wrong': wrong,
        'correct_ids': {s['id'] for s in segments if s['id'] not in {w['id'] for w in wrong}},
    }


def score_trial(case: dict, state: dict) -> dict:
    """对一次智能体运行打分。state 为 RoleAgent.run 的返回值。"""
    segments = state['segments']
    roster = state.get('roster', [])
    matched = match_gold(case, segments)
    by_segment = matched['by_segment']

    final = _accuracy(state['labels'], segments, by_segment, case, roster)
    pre_labels = state.get('pre_review_labels') or state['labels']
    pre = _accuracy(pre_labels, segments, by_segment, case, roster)

    # ReAct 校对净收益
    changed = [seg_id for seg_id in state['labels']
               if (pre_labels.get(seg_id) or {}).get('category') != state['labels'][seg_id]['category']
               or (pre_labels.get(seg_id) or {}).get('speaker') != state['labels'][seg_id]['speaker']]
    fixed = [i for i in changed if i in final['correct_ids'] and i not in pre['correct_ids']]
    broken = [i for i in changed if i not in final['correct_ids'] and i in pre['correct_ids']]

    # 校验规则的命中率：首轮疑点中有多少真的错了，错的里面有多少被抓到
    flagged = {issue['id'] for issue in state.get('first_issues', [])}
    pre_wrong = {w['id'] for w in pre['wrong']}
    validator = {
        'flagged': len(flagged),
        'precision': len(flagged & pre_wrong) / len(flagged) if flagged else None,
        'recall': len(flagged & pre_wrong) / len(pre_wrong) if pre_wrong else None,
    }

    # 角色表
    names = gold_names(case)
    required = {c['name'] for c in case['roster'] if not c.get('optional')}
    optional = {c['name'] for c in case['roster'] if c.get('optional')}
    mapped = {}
    for entry in roster:
        mapped.setdefault(canonical_speaker(entry['name'], case, roster), []).append(entry['name'])
    hits = {gold for gold in mapped if gold in names}
    extras = [names_ for gold, names_ in mapped.items() if gold not in names]
    countable = sum(len(v) for g, v in mapped.items() if g not in optional)
    roster_score = {
        'recall': len(hits & required) / len(required) if required else 1.0,
        'precision': sum(len(v) for g, v in mapped.items() if g in required) / countable if countable else 1.0,
        'split': sorted(g for g, v in mapped.items() if g in names and len(v) > 1),
        'extra': sorted(n for group in extras for n in group),
    }

    # 选角
    mapping = state.get('voice_mapping', {})
    casting_total = casting_ok = 0
    casting_wrong = []
    used = {canonical_speaker(label['speaker'], case, roster): label['speaker'] for label in state['labels'].values()}
    for character, accepted in (case.get('casting') or {}).items():
        if character not in used:
            continue
        voice = (mapping.get(used[character]) or {}).get('voice', '')
        casting_total += 1
        # 列表里的空字符串表示“留空待手动指定”也可以接受
        if voice in accepted or (not voice and ('' in accepted or not accepted)):
            casting_ok += 1
        else:
            casting_wrong.append(f"{character}→{voice or '未指定'}")

    usage = state.get('usage', {})
    return {
        'segments': len(segments),
        'chunks': len(state.get('chunks', [])),
        'gold_unmatched': matched['unmatched'],
        'gold_ambiguous': matched['ambiguous'],
        'label_acc': final['label_acc'],
        'category_acc': final['category_acc'],
        'speaker_acc': final['speaker_acc'],
        'narration_false_positive': final['narration_false_positive'],
        'pre_review_label_acc': pre['label_acc'],
        'review': {
            'triggered': bool(state.get('first_issues')),
            'changed': len(changed),
            'fixed': len(fixed),
            'broken': len(broken),
            'net_gain': final['label_acc'] - pre['label_acc'],
        },
        'validator': validator,
        'roster': roster_score,
        'casting_acc': casting_ok / casting_total if casting_total else None,
        'casting_wrong': casting_wrong,
        'wrong': final['wrong'],
        'usage': usage,
    }


def efficiency(trial: dict, text_chars: int) -> dict:
    """token 利用效率：每段 / 每千字的 token 与花费，推理 token 占比，节点占比。"""
    usage = trial['usage']
    tokens = usage.get('input_tokens', 0) + usage.get('output_tokens', 0)
    by_node = usage.get('by_node', {})
    return {
        'tokens': tokens,
        'tokens_per_segment': tokens / trial['segments'] if trial['segments'] else 0,
        'cost_per_kchar': usage.get('cost', 0) / text_chars * 1000 if text_chars else 0,
        'reasoning_share': usage.get('reasoning_tokens', 0) / usage['output_tokens'] if usage.get('output_tokens') else 0,
        'cache_hit_share': usage.get('cached_tokens', 0) / usage['input_tokens'] if usage.get('input_tokens') else 0,
        'node_token_share': {
            node: (v.get('input_tokens', 0) + v.get('output_tokens', 0)) / tokens if tokens else 0
            for node, v in by_node.items()
        },
        # 每 1% 准确率花多少美元的倒数：越高越划算
        'acc_per_cent': trial['label_acc'] / (usage.get('cost', 0) * 100) if usage.get('cost') else None,
    }


# ---------- 多次运行的稳定性 ----------

def stability(case: dict, states: List[dict]) -> dict:
    """同一文本多次运行：片段级标签一致率、角色表一致性、准确率与花费的离散程度。"""
    if len(states) < 2:
        return {}
    keyed = []
    for state in states:
        roster = state.get('roster', [])
        keyed.append({
            seg_id: (label['category'], canonical_speaker(label['speaker'], case, roster))
            for seg_id, label in state['labels'].items()
        })
    seg_ids = sorted(set.intersection(*(set(k) for k in keyed)))
    all_agree = sum(1 for i in seg_ids if len({k[i] for k in keyed}) == 1)
    pairwise = statistics.mean(
        sum(1 for i in seg_ids if a[i] == b[i]) / len(seg_ids) for a, b in combinations(keyed, 2)
    ) if seg_ids else 0.0

    rosters = [{canonical_speaker(e['name'], case, s.get('roster', [])) for e in s.get('roster', [])} for s in states]
    jaccard = statistics.mean(len(a & b) / len(a | b) if a | b else 1.0 for a, b in combinations(rosters, 2))

    unstable = [
        {'id': i, 'variants': sorted({f"{c}/{s}" for c, s in (k[i] for k in keyed)})}
        for i in seg_ids if len({k[i] for k in keyed}) > 1
    ]

    # 命名一致性：同一金标准角色在各次运行里用的原始名字（界面按原始名字分组，名字一变分组就变）
    raw_names: Dict[str, set] = {}
    for state in states:
        roster = state.get('roster', [])
        for label in state['labels'].values():
            if label['category'] in ('对话', '心理'):
                gold = canonical_speaker(label['speaker'], case, roster)
                if gold in gold_names(case):
                    raw_names.setdefault(gold, set()).add(label['speaker'])
    raw_rosters = [{e['name'] for e in s.get('roster', [])} for s in states]
    return {
        'all_agree_rate': all_agree / len(seg_ids) if seg_ids else 0.0,
        'pairwise_agreement': pairwise,
        'roster_jaccard': jaccard,
        'raw_roster_jaccard': statistics.mean(
            len(a & b) / len(a | b) if a | b else 1.0 for a, b in combinations(raw_rosters, 2)),
        'naming_consistency': (sum(1 for names in raw_names.values() if len(names) == 1) / len(raw_names)
                               if raw_names else 1.0),
        'name_variants': {gold: sorted(names) for gold, names in raw_names.items() if len(names) > 1},
        'unstable_segments': unstable,
    }


def mean_std(values: List[float]) -> dict:
    values = [v for v in values if v is not None]
    if not values:
        return {'mean': None, 'std': None}
    return {'mean': statistics.mean(values), 'std': statistics.pstdev(values) if len(values) > 1 else 0.0}

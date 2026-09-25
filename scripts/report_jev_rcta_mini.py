"""Offline replay audit and matched-baseline report for the new Mini run."""
import argparse
import collections
import csv
import json
from pathlib import Path

import evaluate as ev
import jev_rcta as method
from evaluate_jev_rcta import write_json


class Replay:
    def __init__(self, directory):
        self.directory, self.tags = directory, []

    def call(self, state, questions, tag):
        saved = json.loads((self.directory / (tag + '.json')).read_text())
        digest = ev.sha(ev.dumps({'model': ev.MODEL, 'state': state, 'questions': questions}).encode())
        if saved['request_sha256'] != digest or ev.sha(ev.dumps(saved['request']).encode()) != digest:
            raise ValueError('Replay request mismatch: ' + tag)
        if saved['endpoint'] != ev.ENDPOINT or saved['response']['model'] != ev.MODEL:
            raise ValueError('Endpoint/model mismatch')
        self.tags.append(tag)
        return saved['response']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = ev.ROOT / args.output
    rows, manifest = ev.load_data()
    config = json.loads((out/'config.json').read_text())
    if config['manifest_sha256'] != ev.sha(ev.dumps(manifest).encode()):
        raise ValueError('Data manifest changed')
    for name, digest in config['source_sha256'].items():
        if ev.sha((ev.ROOT/'scripts'/name).read_bytes()) != digest:
            raise ValueError('Frozen source changed: '+name)
    execution = json.loads((out/'execution.json').read_text())
    for field, name in [('runner_sha256', 'run_jev_rcta_mini.py'),
                        ('shared_runner_sha256', 'run_jev_rcta_hosted_full.py')]:
        if execution[field] != ev.sha((ev.ROOT/'scripts'/name).read_bytes()):
            raise ValueError('Runner changed: '+name)
    predictions, completed = {}, []
    for row in rows:
        qid = row['question_ID']
        path = out/'predictions'/(qid+'.json')
        if not path.exists():
            continue
        p = json.loads(path.read_text())
        if p['history_sha256'] != ev.sha(ev.dumps([ev.record(h) for h in row['history']]).encode()):
            raise ValueError('History mismatch: '+qid)
        replay = Replay(out/'calls'/qid)
        replayed = method.predict(row['history'], replay)
        if any(p.get(k) != v for k, v in replayed.items()) or len(replay.tags) != p['api_calls']:
            raise ValueError('Prediction replay mismatch: '+qid)
        predictions[qid] = p
        completed.append(row)
    progress = json.loads((out/'progress.json').read_text())
    if progress['status'] == 'complete' and len(completed) != 200:
        raise ValueError('False completion status')
    calls = [json.loads(p.read_text()) for p in (out/'calls').glob('*/*.json')]
    usage = {'successful_calls_including_partial_cases': len(calls),
             'http_attempts_for_successful_calls': sum(c.get('attempts',1) for c in calls),
             'input_tokens': sum(c['response'].get('usage',{}).get('input_tokens',0) for c in calls),
             'output_tokens': sum(c['response'].get('usage',{}).get('output_tokens',0) for c in calls),
             'actual_cost_usd': None,
             'note': 'API supplies no dollar charges. Compatibility probe excluded. Response-less interrupted requests cannot be accounted for.'}
    audit = {'status': progress['status'], 'completed_n': len(completed), 'expected_n': 200,
             'skipped_ids': progress.get('skipped_ids', []),
             'replay': 'pass', 'frozen_sources': 'pass', 'dataset_hashes': 'pass',
             'source_counts': dict(collections.Counter(r['source'] for r in completed)),
             'usage': usage, 'execution': execution}
    if (out/'stop_diagnosis.json').exists():
        audit['stop_diagnosis'] = json.loads((out/'stop_diagnosis.json').read_text())
        audit['status'] = audit['stop_diagnosis']['status']
    if (out/'recovery.json').exists():
        audit['recovery'] = json.loads((out/'recovery.json').read_text())
    if (out/'access_recovery_20260924.json').exists():
        audit['access_recovery'] = json.loads((out/'access_recovery_20260924.json').read_text())
    audit['previous_attempts'] = [json.loads(p.read_text()) for p in sorted((out/'attempts').glob('*/execution.json'))]
    probe_dir = out.with_name(out.name+'_probe')
    probe_calls = [json.loads(p.read_text()) for p in (probe_dir/'calls').glob('*.json')]
    extra_probes = [json.loads(p.read_text()) for pattern in ('new_key_control_*.json','connectivity_*.json')
                    for p in probe_dir.glob(pattern)]
    extra_probes = [p for p in extra_probes if p.get('passed')]
    audit['control_probe_usage'] = {'successful_calls': len(probe_calls),
        'input_tokens': sum(c['response'].get('usage',{}).get('input_tokens',0) for c in probe_calls),
        'output_tokens': sum(c['response'].get('usage',{}).get('output_tokens',0) for c in probe_calls)}
    audit['control_probe_usage']['successful_calls'] += len(extra_probes)
    for field in ('input_tokens','output_tokens'):
        audit['control_probe_usage'][field] += sum(p.get('usage',{}).get(field,0) for p in extra_probes)
    summary = ev.summarize(completed, predictions, manifest) if completed else {}
    summary.pop('estimated_input_cost_usd', None)
    summary.update(protocol=config, audit=audit, cost_note=usage['note'])
    summary['candidate_recall'] = {f: sum(r['mistake_step'] in predictions[r['question_ID']][f]
        for r in completed)/len(completed) if completed else None
        for f in ('recall_candidates','trace_seeds','expanded_candidates','final_candidates')}
    baseline = json.loads((ev.ROOT/'reports/phase1_metrics.json').read_text())
    baseline_predictions = {p.stem: json.loads(p.read_text())
                            for p in (ev.ROOT/'results/phase1/predictions').glob('*.json')}
    if baseline['dataset_revision'] != manifest['revision']:
        raise ValueError('Baseline data revision mismatch')
    matched = ev.summarize(completed, {r['question_ID']:baseline_predictions[r['question_ID']] for r in completed}, manifest) if completed else {}
    summary['matched_baseline_overall'] = matched.get('overall')
    summary['matched_baseline_usage'] = {k: matched.get(k) for k in ('successful_api_calls', 'input_tokens')}
    summary['coverage'] = len(completed)/len(rows)
    summary['missing_ids'] = [r['question_ID'] for r in rows if r['question_ID'] not in predictions]
    summary['all_200_accuracy_missing_counted_wrong'] = {
        key:sum(ev.score_row(r,predictions[r['question_ID']])[key] for r in completed)/len(rows)
        for key in ('role_correct','step_exact','step_within_5')}
    compact = {k:v for k,v in summary.items() if k!='details'}
    report = ev.ROOT/'reports'
    stem = out.name
    write_json(report/(stem+'_metrics.json'), compact)
    write_json(report/(stem+'_audit.json'), audit)
    with (report/(stem+'_predictions.csv')).open('w') as handle:
        fields = ['question_ID','source','reference_role','predicted_role','reference_step','predicted_step','role_correct','step_exact','step_within_5']
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        for d in summary.get('details',[]):
            writer.writerow({k:d['prediction'][k] if k.startswith('predicted_') else d[k] for k in fields})
    pct=lambda n: '—' if n is None else '%.1f%%'%(100*n)
    lines=['# JEV-RCTA Mini 评测（2026-09-23 启动）','',
           '状态：**%s**；完成 **%d/200** 条。'%(audit['status'],len(completed)), '',
           '方法：`jev-rcta-choice-v2`；模型：`jev-1.13.0`；端点：`%s`。'%ev.ENDPOINT,
           '预测未读取人工标签；完成后通过逐请求哈希和离线重放核对。', '']
    if completed:
        lines+=['## 与相同样本的原 JEV 基线比较','','| 指标 | 原 JEV | JEV-RCTA | 变化 |','|---|---:|---:|---:|']
        for key,label in [('role_correct','责任角色准确率'),('step_exact','根因步骤 Exact'),('step_within_5','根因步骤 ±5')]:
            old,new=matched['overall'][key],summary['overall'][key]
            lines.append('| %s | %s | %s | %+.1f 个百分点 |'%(label,pct(old),pct(new),100*(new-old)))
        lines+=['','根因步骤 MAE：%.2f → %.2f（越低越好）。'%(matched['overall']['valid_output_root_mae'],summary['overall']['valid_output_root_mae']),
                '', '## 各来源','','| 来源 | 条数 | 角色准确率 | Exact | ±5 |','|---|---:|---:|---:|---:|']
        for source,m in summary['by_source'].items():
            lines.append('| %s | %d | %s | %s | %s |'%(source,m['n'],pct(m['role_correct']),pct(m['step_exact']),pct(m['step_within_5'])))
        lines+=['','## 候选召回率','']
        for key,value in summary['candidate_recall'].items():
            lines.append('- `%s`：%s'%(key,pct(value)))
        lines+=['','## 相同已完成样本的推理开销','',
                '| 用量 | 原 JEV | JEV-RCTA |','|---|---:|---:|',
                '| 成功调用 | %s | %s |'%(matched['successful_api_calls'],summary['successful_api_calls']),
                '| 输入 tokens | %s | %s |'%(format(matched['input_tokens'],','),format(summary['input_tokens'],','))]
    lines+=['','## 用量与边界','',
            '- 成功请求：%d 次；输入 tokens：%s；输出 tokens：%s。'%(len(calls),format(usage['input_tokens'],','),format(usage['output_tokens'],',')),
            '- 接口不返回实际美元扣费，不能将估算当成账单。兼容性探针单独记录。',
            '- 从串行切换并发时已复用成功调用缓存；被中断且未收到响应的请求可能计费，但无法纳入返回用量。',
            '- 这是单次 Mini 评测，不代表 Full 分数或统计显著改进。',
            '- 若未完成 200 条，表格仅比较已完成的相同样本。','']
    if 'recovery' in audit:
        lines+=['期间出现 HTTP 520，已暂停并用小请求确认恢复，再复用缓存继续。故障记录保存在运行目录的 `attempts/` 中。','']
    if 'access_recovery' in audit:
        lines+=['2026-09-24，6 条此前被拦截的原始请求均返回 HTTP 200；请求内容与哈希不变，已保存响应并继续剩余推理。',
                '各轮使用相同模型与端点；已完成预测及精确请求缓存复用，原始响应可供审计。','']
    if 'stop_diagnosis' in audit:
        lines+=['本轮停止诊断：`%s`。'%audit['stop_diagnosis'].get('root_cause',audit['stop_diagnosis']['status']),
                '**未完成 Mini，以上指标只适用于已完成的 %d 条样本，不能当作完整 Mini 得分。**'%len(completed),'']
    if audit['skipped_ids']:
        lines+=['持续被拦截而保留为未完成的样本：'+', '.join('`'+q+'`' for q in audit['skipped_ids'])+'。',
                '模型、方法和请求内容保持不变；已完成的旧 Key 结果由精确请求缓存复用，其余使用新 Key 完成。','']
    if len(completed)<len(rows):
        lines+=['覆盖率：%d/200。缺失预测按错误计、以全部 200 条为分母的准确率（运行结果，非完整推理结果）：'%len(completed)]
        for k,v in summary['all_200_accuracy_missing_counted_wrong'].items():
            lines.append('- `%s`：%s'%(k,pct(v)))
        lines.append('')
    lines+=['兼容性/恢复检查额外成功调用 %d 次，输入 %d tokens，输出 %d tokens。'%(
        audit['control_probe_usage']['successful_calls'],audit['control_probe_usage']['input_tokens'],audit['control_probe_usage']['output_tokens']),'']
    (report/(stem+'_report.md')).write_text('\n'.join(lines))
    print(json.dumps(compact,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()

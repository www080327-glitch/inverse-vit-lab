"""Generate C09 RESULT.json and reproducible delivery zip; <=1024 tokens."""
import base64
import hashlib
import io
import json
import math
import platform
import random
import time
import unittest
import zipfile
from pathlib import Path
from reference_attention import *

ROOT=Path(__file__).resolve().parent
SEED=9027
MODES=("mean_key","max","logsumexp","mean_log_size")


def probabilities_case(name, s, regions):
    true=softmax(approximate_region_logits(s,regions,"logsumexp"))
    approximations={}
    for mode in MODES:
        scores=approximate_region_logits(s,regions,mode)
        p=softmax(scores)
        approximations[mode]={
            "region_probabilities": p,
            "top1": select_topk(scores,1)[0],
            "probability_l1_error": math.fsum(abs(a-b) for a,b in zip(p,true))}
    return {"name":name,"region_sizes":[len(r) for r in regions],
            "true_region_probabilities":true,"approximations":approximations}


def run():
    t0=time.perf_counter()
    cpu0=time.process_time()
    suite=unittest.defaultTestLoader.discover(str(ROOT),pattern="test_reference_attention.py")
    test_output=io.StringIO()
    tested=unittest.TextTestRunner(stream=test_output,verbosity=2).run(suite)
    (ROOT/"TEST_REPORT.txt").write_text(test_output.getvalue(),encoding="utf-8")
    if not tested.wasSuccessful():
        raise RuntimeError(test_output.getvalue())
    rng=random.Random(SEED)
    sizes=[1,3,12,52,188,768,0]
    regions=[]
    offset=0
    for n in sizes:
        regions.append(list(range(offset,offset+n)))
        offset+=n
    n=offset
    d=8
    dv=6
    # Exactly encode chosen logits as q dot k / sqrt(d).
    q=[math.sqrt(d)]+[0.]*(d-1)
    normal=[rng.gauss(0,1.5) for _ in range(n)]
    peaked=[-6.]*n
    peaked[0]=2.
    peaked[256]=9.  # hidden in the largest region
    scenarios={
        "uniform_unequal": [0.]*n,
        "gaussian": normal,
        "hidden_peak": peaked,
        "extreme_logits": [10000.,9999.]+[-10000.]*(n-2),
    }
    rows=[]
    max_equivalence=0.
    for case,s in scenarios.items():
        keys=[[x]+[rng.uniform(-1,1) for _ in range(d-1)] for x in s]
        values=[]
        for i in range(n):
            row=[rng.gauss(0,1) for _ in range(dv)]
            scale=max(1.,l2(row))
            values.append([x/scale for x in row])
        full_scores=attention_logits(q,keys)
        dense=dense_attention(full_scores,values)
        h=hierarchical_attention(full_scores,values,regions)
        max_equivalence=max(max_equivalence,
            max(abs(a-b) for a,b in zip(dense["output"],h["output"])))
        centroids=prepare_centroids(keys,regions)
        for mode in MODES:
            for k in (1,2,4,6):
                route=route_attention(q,keys,values,regions,k,mode,centroids)
                diag=truncation_diagnostic(full_scores,values,regions,
                                           route["selected_regions"])
                direct=selected_attention(full_scores,values,regions,
                                           route["selected_regions"])["output"]
                assert max(abs(a-b) for a,b in zip(route["output"],direct))<2e-12
                assert diag["l2_output_error"] <= diag["error_bound_2B_epsilon"]+2e-12
                rows.append({
                    "case":case,"mode":mode,"k":k,
                    "selected_regions":route["selected_regions"],**diag,
                    "online_cost":route["cost"],
                    "offline_centroid_preparation":{
                        "needed":mode in ("mean_key","mean_log_size"),
                        "fine_key_scalars_read_per_K_or_partition_change":n*d,
                        "stored_centroid_scalars":6*d,
                        "amortization_not_assumed":True},
                    "separate_full_reference_diagnostic_cost":{
                        "fine_key_rows":n,"fine_value_rows":n,
                        "qk_scalar_pairs":n*d,"av_scalar_pairs":n*dv},
                    "dense_baseline_cost":{"qk_scalar_pairs":n*d,
                                           "av_scalar_pairs":n*dv}})
    examples=[
        probabilities_case("uniform_1_vs_9",[0.]*10,[[0],list(range(1,10))]),
        probabilities_case("size_defeats_max",[3.]+[0.]*64,[[0],list(range(1,65))]),
        probabilities_case("dispersion_defeats_mean_log_size",[1.,8.]+[-8.]*63,
                           [[0],list(range(1,65))]),
        probabilities_case("single_token_regions",[3.,-2.,1.],[[0],[1],[2]]),
    ]
    sharp=truncation_diagnostic([math.log(.9),math.log(.1)],[[1.],[-1.]],
                                [[0],[1]],[0])
    unbounded=truncation_diagnostic([math.log(1-1e-6),math.log(1e-6)],
                                   [[0.],[1e6]],[[0],[1]],[0])
    # Expand residual range: s in [a,b] implies Jensen gap <= (b-a)^2/8.
    jensen_checks=[]
    for width in [0.,.1,1.,4.,16.]:
        x=[rng.uniform(-width/2,width/2) for _ in range(64)]
        mean=math.fsum(x)/len(x)
        gap=logsumexp(x)-math.log(len(x))-mean
        bound=(max(x)-min(x))**2/8
        assert gap>=-2e-12 and gap<=bound+2e-12
        jensen_checks.append({"requested_width":width,"gap":gap,
                              "hoeffding_bound":bound})
    # Hold fixed keys/values; uniform duplication vs partial duplication.
    duplicates={
        "base_output":dense_attention([0.,0.],[[1.],[-1.]])["output"],
        "one_region_duplicated":dense_attention([0.,0.,0.],[[1.],[1.],[-1.]])["output"],
        "uniform_duplication":dense_attention([0.]*4,[[1.],[-1.],[1.],[-1.]])["output"],
        "known_identical_copy_correction":dense_attention(
            [-math.log(2),-math.log(2),0.],[[1.],[1.],[-1.]])["output"],
        "correction_not_general_for_nonidentical_coarse_and_fine":True}
    result={
        "task":"C09","scope":"synthetic CPU mathematical/numerical reference only",
        "execution":{"python":platform.python_version(),"platform":platform.system(),
                     **binary64_environment(),"seed":SEED,
                     "single_query":True,"key_dimension":d,"value_dimension":dv,
                     "elapsed_seconds":time.perf_counter()-t0,
                     "process_cpu_seconds":time.process_time()-cpu0,
                     "ten_minute_budget":600,"models_loaded":False,
                     "data_downloaded":False,"gpu_used":False,
                     "dependencies":"Python standard library only"},
        "tests":{"run":tested.testsRun,"failures":len(tested.failures),
                 "errors":len(tested.errors),"passed":tested.wasSuccessful()},
        "dense_hierarchical_max_abs_error":max_equivalence,
        "topk_rows":rows,"approximation_counterexamples":examples,
        "bound_tight_example":sharp,"missing_norm_assumption_counterexample":unbounded,
        "jensen_checks":jensen_checks,"duplicate_content":duplicates,
        "claims_not_supported":["GPU speedup","end-to-end speedup",
                                "model quality gain","learned routing accuracy"],
        "cost_note":"Online routes are actually executed without dense logits for centroid routing. Full diagnostic reads are separately charged. Counts ignore Python overhead and memory layout; offline centroid cost is not free.",
        "source_sha256":{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                         for name in ("reference_attention.py","test_reference_attention.py",
                                      "run_experiment.py","REPORT.md","FAILURE_EXAMPLES.md","README.md")}
    }
    assert result["execution"]["elapsed_seconds"]<600
    (ROOT/"RESULT.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,
                                               allow_nan=False)+"\n",encoding="utf-8")
    url="https://github.com/www080327-glitch/inverse-vit-lab"
    handoff=[
        "# C09 交接：现有分层原型最容易算错的三件事",
        "",
        "1. 区域内 softmax 后平均会丢失真实区域质量；正确跨区权重来自 logsumexp。",
        "2. 精确 logsumexp / 最大分数需要读取全部细 key，不能当作免费粗路由。",
        "3. 粗细直接拼接会重复计权；替换、残差、拼接必须先定义模型语义。",
        "",
        "## 实测与入口",
        "",
        f"- 22 项测试中实际运行 {tested.testsRun} 项，失败/错误均为 0。",
        f"- 64 组固定 Top-k 对照；最多 1024 个细 Token；seed={SEED}。",
        f'- Linux Python {platform.python_version()}，binary64；实验墙钟 {result["execution"]["elapsed_seconds"]:.6f} 秒，进程 CPU {result["execution"]["process_cpu_seconds"]:.6f} 秒（不含打包和解释器启动）。',
        f"- 四个固定场景 dense/精确分层最大绝对输出差：{max_equivalence:.17g}。",
        "- 推导、条件、接口与接入建议：[REPORT.md](REPORT.md)。",
        "- 实现：[reference_attention.py](reference_attention.py)。",
        "- 失败反例：[FAILURE_EXAMPLES.md](FAILURE_EXAMPLES.md)。",
        "- 所有 64 行、质量与成本：[RESULT.json](RESULT.json)。",
        "- 实际测试输出：[TEST_REPORT.txt](TEST_REPORT.txt)。",
        "- 完整包：[C09_DELIVERY.zip](C09_DELIVERY.zip)。",
        "",
        "## Top-1 实测摘录（固定场景全部列出）",
        "",
        "| 场景 | 区域打分 | 保留真实质量 | L2 输出误差 | 选中细 Token | 在线 qk+av 乘加对 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["k"]==1:
            cost=row["online_cost"]
            pairs=cost["qk_scalar_multiply_add_pairs"]+cost["av_scalar_multiply_add_pairs"]
            handoff.append(f'| {row["case"]} | {row["mode"]} | {row["retained_mass"]:.9g} | {row["l2_output_error"]:.9g} | {row["selected_fine_tokens"]} | {pairs} |')
    handoff += [
        "",
        "计数不是时间，不能从这张表推断 GPU 加速。精确 max/LSE 的全部细 key 成本已计入在线列；粗 key 缓存更新与独立的全细 oracle 诊断另列于 JSON，表中未把它们混进在线计数。不同区域长度下，Top-1 对应的细 Token 成本差别很大。",
        "",
        "只要 ||v_i||₂≤B，且选中细 Token 沿用原 logits 精确重归一化，就有 ||y−y_S||₂≤2Bε。尖锐例达到等号；无 value 范数界、或选中后改权重的例子会使不当套用失效。详见报告。",
        "",
        "## 范围与复现",
        "",
        "本任务只验证数学与小型 CPU 数值，不含 Qwen 特征、模型训练、GPU 吞吐量或视觉质量结论。",
        "",
        "运行 python3 run_experiment.py 即可重跑测试和实验并生成此交接、结果与包。只需 Python 标准库。源文件 SHA256 在 RESULT.json；其他会话或现有原型无需修改。",
        "",
        f"代码审阅与云端 CI：[PR #12]({url}/pull/12)。",
        "",
    ]
    (ROOT/"HANDOFF.md").write_text("\n".join(handoff),encoding="utf-8")
    names=["HANDOFF.md","reference_attention.py","test_reference_attention.py","run_experiment.py",
           "REPORT.md","FAILURE_EXAMPLES.md","README.md","RESULT.json","TEST_REPORT.txt"]
    # Deterministic member timestamps. The measured elapsed time is intentionally
    # part of RESULT and will differ across independent reruns.
    with zipfile.ZipFile(ROOT/"C09_DELIVERY.zip","w",zipfile.ZIP_DEFLATED) as z:
        for name in names:
            info=zipfile.ZipInfo("C09/"+name,date_time=(2026,10,27,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,(ROOT/name).read_bytes())
    with zipfile.ZipFile(ROOT/"C09_DELIVERY.zip") as z:
        assert z.testzip() is None
    print(json.dumps({"tests":result["tests"],
                      "elapsed_seconds":result["execution"]["elapsed_seconds"],
                      "dense_hierarchical_max_abs_error":max_equivalence,
                      "topk_rows":len(rows)},allow_nan=False))


if __name__=="__main__":
    run()

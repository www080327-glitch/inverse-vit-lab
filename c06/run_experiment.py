"""Single bounded CPU run; writes measured evidence, report and self-contained zip."""
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "2"
import datetime
import hashlib
import io
import json
from pathlib import Path
import platform
import resource
import signal
import sys
import time
import unittest
import zipfile
import numpy as np
from detail_bypass import (save_bank, BankReader, save_residual_bank, ResidualReader,
    coarse_and_residual, SemanticInput, bind_source, read_details, read_coarse_only)
from scenes import generate, draw_scene, png

ROOT = Path(__file__).resolve().parent
OUT = ROOT/"outputs"
START_WALL = time.perf_counter()
START_CPU = time.process_time()
STARTED = datetime.datetime.now(datetime.timezone.utc).isoformat()
if hasattr(os,"sched_setaffinity"):
    available = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0,available[:2])
signal.alarm(1100)  # entire numerical run, including tests, <20 min
resource.setrlimit(resource.RLIMIT_CPU,(1100,1150))


def dump(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8")


def error(a,b):
    diff = a.astype(np.int64)-b.astype(np.int64)
    return dict(max_abs_error=int(np.abs(diff).max()) if diff.size else 0,
                mae=float(np.abs(diff).mean()) if diff.size else 0.0,
                mse=float((diff*diff).mean()) if diff.size else 0.0,
                differing_values=int(np.count_nonzero(diff)),
                differing_pixels=int(np.count_nonzero(np.any(diff != 0,axis=2))),
                exact=bool(np.array_equal(a,b)))


def timed_call(fn):
    start=time.perf_counter()
    value=fn()
    return value, (time.perf_counter()-start)*1000


def roi_pair(path,roi,reader_cls):
    with reader_cls(path) as reader:
        image,ms=timed_call(lambda:reader.read_roi(roi))
        stats=reader.stats()
    return image,stats,ms


def detail_view(a,coarse,recon,raw,roi,path):
    x0,y0,x1,y1=roi
    crops=[x[y0:y1,x0:x1] for x in (a,coarse,recon,raw)]
    panel=np.concatenate(crops,axis=1)
    zoom=6 if panel.shape[1]*6 <= 1600 else 3
    png(path,panel.repeat(zoom,0).repeat(zoom,1))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    import test_bypass
    stream=io.StringIO()
    runner=unittest.TextTestRunner(stream=stream,verbosity=2)
    tests=runner.run(unittest.defaultTestLoader.loadTestsFromModule(test_bypass))
    (OUT/"TEST_LOG.txt").write_text(stream.getvalue(),encoding="utf-8")
    if not tests.wasSuccessful():
        raise RuntimeError("correctness tests failed; see TEST_LOG")
    records=generate(OUT/"scenes")
    scenes=[]
    for rec in records:
        a,targets=draw_scene(rec["height"],rec["width"],rec["variant"])
        h,w,_=a.shape
        d=OUT/rec["name"]
        d.mkdir(exist_ok=True)
        raw_storage=save_bank(d/"raw.bank",a)
        residual_storage=save_residual_bank(d/"residual",a)
        (coarse_fulls,coarse_io),coarse_ms=timed_call(
            lambda:read_coarse_only(d/"residual",[(0,0,w,h)]))
        coarse_full=coarse_fulls[0]
        raw_full,raw_io,raw_ms=roi_pair(d/"raw.bank",(0,0,w,h),BankReader)
        residual_full,residual_io,residual_ms=roi_pair(d/"residual",(0,0,w,h),ResidualReader)
        np.testing.assert_array_equal(raw_full,a)
        np.testing.assert_array_equal(residual_full,a)
        sem=SemanticInput(np.random.default_rng(6006).normal(size=(4,16)).astype(np.float32),
                          "SYNTHETIC-independent-random-vectors",True,rec["rgb_sha256"])
        semantic_bytes=sem.save(d)
        binding_bytes=bind_source(a,sem,"raw.bank",d)
        handle=json.loads((d/"source_handle.json").read_text())
        integration,_=read_details(handle,d,[tuple(targets[0]["roi"])],rec["rgb_sha256"])
        x0,y0,x1,y1=targets[0]["roi"]
        np.testing.assert_array_equal(integration[0],a[y0:y1,x0:x1])
        queries=[]
        for target in targets:
            roi=tuple(target["roi"])
            x0,y0,x1,y1=roi
            truth=a[y0:y1,x0:x1]
            raw,rs,rm=roi_pair(d/"raw.bank",roi,BankReader)
            residual,bs,bm=roi_pair(d/"residual",roi,ResidualReader)
            (cs,cio),cm=timed_call(lambda:read_coarse_only(d/"residual",[roi]))
            c=cs[0]
            assert np.array_equal(raw,raw_full[y0:y1,x0:x1])
            assert np.array_equal(residual,residual_full[y0:y1,x0:x1])
            assert np.array_equal(c,coarse_full[y0:y1,x0:x1])
            queries.append(dict(**target,output_rgb_bytes=truth.nbytes,
                raw=dict(error=error(truth,raw),io=rs,elapsed_ms=rm),
                coarse_residual=dict(error=error(truth,residual),io=bs,elapsed_ms=bm),
                coarse_only=dict(error=error(truth,c),io=cio,elapsed_ms=cm)))
            detail_view(a,coarse_full,residual_full,raw_full,roi,
                        d/f'comparison_{target["name"]}.png')
        # Explicit multi-tile crossing, odd edge and overlap; query coordinates are oracle.
        boxes=[(31,31,35,35),(319,191,329,199),(320,192,333,204),
               (w-4,h-4,w,h)]
        boundaries={}
        for method,path,cls in [("raw",d/"raw.bank",BankReader),
                                ("coarse_residual",d/"residual",ResidualReader)]:
            with cls(path) as reader:
                crops=reader.read_many(boxes)
                stats=reader.stats()
            independent_bytes=sum(roi_pair(path,b,cls)[1]["bytes_read"] for b in boxes)
            for roi,crop in zip(boxes,crops):
                x0,y0,x1,y1=roi
                np.testing.assert_array_equal(crop,a[y0:y1,x0:x1])
            boundaries[method]=dict(exact=True,boxes=boxes,batch_io=stats,
                                   independent_read_bytes=independent_bytes,
                                   overlap_saving_bytes=independent_bytes-stats["bytes_read"])
        # Dense random ROI verification on this high-resolution scene.
        rng=np.random.default_rng(6006)
        random_boxes=[]
        for _ in range(128):
            x0=int(rng.integers(0,w))
            y0=int(rng.integers(0,h))
            random_boxes.append((x0,y0,min(w,x0+int(rng.integers(1,75))),
                                 min(h,y0+int(rng.integers(1,75)))))
        for path,cls in [(d/"raw.bank",BankReader),(d/"residual",ResidualReader)]:
            with cls(path) as reader:
                for roi,crop in zip(random_boxes,reader.read_many(random_boxes)):
                    x0,y0,x1,y1=roi
                    np.testing.assert_array_equal(crop,a[y0:y1,x0:x1])
        scenes.append(dict(**rec,source_rgb_bytes=a.nbytes,raw_storage=raw_storage,
            residual_storage=residual_storage,semantic_storage_bytes=semantic_bytes,
            source_handle_bytes=binding_bytes,
            semantic_values_payload_bytes=4*16*4,
            full=dict(raw=dict(error=error(a,raw_full),io=raw_io,elapsed_ms=raw_ms),
                      coarse_residual=dict(error=error(a,residual_full),io=residual_io,elapsed_ms=residual_ms),
                      coarse_only=dict(error=error(a,coarse_full),io=coarse_io,elapsed_ms=coarse_ms)),
            queries=queries,boundaries=boundaries,random_roi_cases_per_method=128))
    # Constructive counterexample: exact same coarse eye/accessory cells, different details.
    a,_=draw_scene(1025,1537,0)
    b,_=draw_scene(1025,1537,1)
    ca,_=coarse_and_residual(a)
    cb,_=coarse_and_residual(b)
    collisions=[]
    for name,roi in [("eyes",[512,288,528,296]),("accessory",[640,384,648,392])]:
        x0,y0,x1,y1=roi
        same=np.array_equal(ca[y0//8:y1//8,x0//8:x1//8],cb[y0//8:y1//8,x0//8:x1//8])
        distinct=not np.array_equal(a[y0:y1,x0:x1],b[y0:y1,x0:x1])
        assert same and distinct
        collisions.append(dict(target=name,roi=roi,coarse_equal=bool(same),source_different=bool(distinct),
                               interpretation="coarse summary alone cannot recover which source geometry occurred"))
    result=dict(task="C06",status="completed",started_utc=STARTED,
        completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        actual_run=dict(tests_run=tests.testsRun,tests_success=True,scenes=4,
            high_resolution_random_roi_checks=4*128*2,
            target_roi_checks=4*5*3,boundary_batches=8,
            numerical_wall_seconds=time.perf_counter()-START_WALL,
            numerical_process_cpu_seconds=time.process_time()-START_CPU),
        environment=dict(platform=platform.platform(),machine=platform.machine(),
            python=sys.version.split()[0],numpy=np.__version__,executor="GitHub Actions public standard ubuntu-24.04",
            affinity_cpus=sorted(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else None,
            observed_process_threads=len(os.listdir("/proc/self/task")) if Path("/proc/self/task").exists() else None,
            numeric_thread_cap=2,wall_alarm_seconds=1100,cpu_soft_limit_seconds=1100,
            workflow_run_url=os.environ.get("C06_RUN_URL"),source_commit=os.environ.get("GITHUB_SHA"),
            model_downloads=0,paid_api_calls=0,paid_rental_used=False,
            inputs="this task prompt; original deterministic procedural scenes; numpy; Python stdlib",
            local_host_probe="runtime platform check only; Mac executor rejected before file reads or computation"),
        limitations=[
            "No Qwen features, Qwen weights, neural training, learned routing or semantic QA were used.",
            "All semantic vectors are SYNTHETIC; ROI selection is oracle coordinates, not semantic localization.",
            "Source-bypass exact reconstruction is NOT evidence of semantic encoder invertibility.",
            "Text-only generation has no source RGB or residual bank; source-detail read is unavailable.",
            "No lossless size reduction is claimed: exact residual coefficients cost more than source uint8.",
            "IO bytes are observed file.read return bytes including metadata, not physical disk/page-cache traffic.",
            "Read latency is one CPU sample, not an end-to-end speed benchmark; no backbone speedup measured.",
            "Token counts are interface accounting proxies, not measured Qwen tokenization or FLOPs.",
            "No user files, Mac project, DCU, LAN, model caches or other Work sessions were read."],
        coarse_collision_counterexamples=collisions,scenes=scenes)
    result["actual_run"]["within_20min"]=result["actual_run"]["numerical_wall_seconds"] < 1200
    assert result["actual_run"]["within_20min"]
    assert result["environment"]["observed_process_threads"] <= 2
    dump(ROOT/"RESULT.json",result)
    write_report(result)
    # Bundle every deliverable under c06, excluding itself and bytecode.
    zip_path=ROOT/"C06_DELIVERY.zip"
    with zipfile.ZipFile(zip_path,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(ROOT.rglob("*")):
            if p.is_file() and p.name not in {"C06_DELIVERY.zip","PACKAGE.json"} and "__pycache__" not in p.parts:
                z.write(p,"C06_DELIVERY/"+str(p.relative_to(ROOT)))
    assert zip_path.stat().st_size < 30*1024*1024
    with zipfile.ZipFile(zip_path) as z:
        assert z.testzip() is None
        assert "C06_DELIVERY/RESULT.json" in z.namelist()
    dump(ROOT/"PACKAGE.json",dict(filename=zip_path.name,bytes=zip_path.stat().st_size,
         sha256=hashlib.sha256(zip_path.read_bytes()).hexdigest(),
         zip_integrity="all members CRC checked",contains="code, tests, scenes, banks, reports, RESULT.json",
         packaging_wall_seconds=time.perf_counter()-START_WALL,
         total_process_cpu_seconds=time.process_time()-START_CPU))
    print(json.dumps(dict(status="completed",numerical_seconds=result["actual_run"]["numerical_wall_seconds"],
                          zip_bytes=zip_path.stat().st_size),ensure_ascii=False))


def write_report(r):
    lines=["# C06：保留源图细节、按需读取的视觉旁路","",
        "本实验只验证源图旁路的存储与恢复。未使用 Qwen 特征、未训练镜像解码器或分层 Attention。所有场景均为原始程序化数据，语义向量明确标记为 SYNTHETIC。","",
        "## 三种“不压缩”分别意味着什么","",
        "|要求|本原型的实际行为|",
        "|---|---|",
        "|不丢弃原始信息|完整 patch 库和粗图+精确残差逐像素还原；粗图单独有损。|",
        "|不减少进入主干的 Token|与无损保存独立。这里保留库中全部像素，仅在接口上选 ROI；未运行真实主干。若要求全部原始 patch 每次进入主干，不能把选区读取当成同一要求。|",
        "|不降低全部位置计算成本|本原型没有实现或验证这种约束。选区可少读数据，不能同时声称已经对所有位置执行完整视觉计算。|","",
        "## 表示与存储账本","",
        "A：RGB 经 zero-padding、reshape、transpose 变成 [N,T,T,3]。逆排列后裁去已知补齐区；没有池化、插值或降维。坐标以每块4个 uint32 保存。文件为12字节前缀、完整JSON头、坐标表和未压缩像素系数。","",
        "B：每8×8有效区域取整数 floor mean，边界只计算有效像素。R=int16(X)-int16(expand(C))；恢复 X=expand(C)+R。粗图是有损摘要，残差完整保存。系数、坐标、padding、dtype、manifest 全部计费；没有熵编码或无损压缩率宣称。","",
        "|场景|源RGB字节|仅粗图+元数据字节|原始patch总字节|粗图+精确残差总字节|粗图MAE|精确路径最大误差|",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for s in r["scenes"]:
        lines.append(f'|{s["name"]}|{s["source_rgb_bytes"]}|{s["residual_storage"]["coarse_only_bytes"]}|{s["raw_storage"]["total_bytes"]}|{s["residual_storage"]["total_bytes"]}|{s["full"]["coarse_only"]["error"]["mae"]:.6f}|0|')
    lines += ["","各 bank 的 header、坐标表、有效系数、补齐字节和 dtype 均在 RESULT.json 单列。语义 .npy 的128字节格式头、向量256字节及 semantic.json、source_handle.json 也单列；上表只计算对应图像表示，不把语义向量偷偷并入/排除一种对照。",
        "","## 区域读取的真实测量","",
        "坐标为源图像素的半开区间(x0,y0,x1,y1)。Reader 根据坐标算相交块，只 seek/read 所需坐标项和块数据；不 mmap、不预先读取整个payload。跨块会多读边缘，重叠 ROI 在同一 read_many 调用内共用块；跨调用不保留缓存。只统计实际 file.read 返回字节，包含每个新 reader 的头和manifest，不等同于物理磁盘I/O。","",
        "|场景/目标|目标占比|输出RGB字节|仅粗图读取字节|原始patch读取字节|粗图+残差读取字节|粗图局部MAE|精确路径最大误差|",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in r["scenes"]:
        for q in s["queries"]:
            lines.append(f'|{s["name"]}/{q["name"]}|{100*q["image_fraction"]:.5f}%|{q["output_rgb_bytes"]}|{q["coarse_only"]["io"]["bytes_read"]}|{q["raw"]["io"]["bytes_read"]}|{q["coarse_residual"]["io"]["bytes_read"]}|{q["coarse_only"]["error"]["mae"]:.4f}|0|')
    lines += ["","所有 fixed target 均小于图像1%；小字形是硬编码3×5笔画，细线1像素，眼睛/附件有成对几何变化。比较图每行从左到右：源图、仅粗图、粗图+残差、完整patch；采用整数重复放大以便看清像素。全部固定场景与全部目标均提供，未挑选样本。","",
        "![眼睛对照](outputs/desktop_1537x1025_v0/comparison_eyes_open.png)","",
        "## 边界与重叠证据","",
        "|场景/路径|跨块/奇数/重叠一致|合并请求读取字节|逐个请求字节|合并节省字节|",
        "|---|---|---:|---:|---:|"]
    for s in r["scenes"]:
        for k,b in s["boundaries"].items():
            lines.append(f'|{s["name"]}/{k}|逐像素一致|{b["batch_io"]["bytes_read"]}|{b["independent_read_bytes"]}|{b["overlap_saving_bytes"]}|')
    lines += ["",f'除上述证据外，还执行{r["actual_run"]["high_resolution_random_roi_checks"]}次高分辨率随机ROI一致性检查、uint8/int8/uint16/int16/int32/uint32/int64/uint64的pack往返（含非连续视图、uint64极值）、单像素/小于块/奇数尺寸、空ROI、非法ROI、损坏坐标及截断文件、语义身份误配检查。单元测试记录见outputs/TEST_LOG.txt。',"",
        "眼睛开/闭以及附件方块/横条在对应8×8区域的粗表示完全相同，而源像素不同（RESULT.json给出断言）。因此粗表示单独不存在唯一恢复；精确恢复来自残差/原图，不能来自“更聪明的上采样”这一假设。","",
        "## 与真实视觉特征实验分栏","",
        "|对照|源RGB/残差可读|语义输入|本次实际结果|",
        "|---|---|---|---|",
        "|仅粗图|仅粗摘要|无|局部与全图误差见上表，有信息碰撞。|",
        "|粗图+残差|是|无|本次全部测试逐像素一致。|",
        "|完整patch库|是|可独立挂合成向量|本次全部测试逐像素一致。|",
        "|只用真实Qwen视觉特征的镜像模型|否|真实Qwen特征|未运行；不得由旁路结果推断。|",
        "|纯文字生图|不存在源图|文字条件|本旁路不可用；未运行。|","",
        "## Token、读取与计算账本","",
        "以最大场景为例，32×32存储块并不等于Qwen token；源块数量与8×8粗图像素数可以计算，但未测量Qwen tokenizer。合成语义输入固定4×16 float32，完全随机；它不是Qwen、不是语义能力证据，也未被送入主干。","",
        "源库不删像素；ROI读取仅返回被请求的源像素及相交块。未来可先把语义摘要送给主干，再根据问题产生ROI，将选区另行编码。这样能减少进入主干的数据量和部分计算，但不等于“全部位置计算量不变”，也不保证端到端更快。这里用已知目标坐标演示按需读取，未实现自然语言定位器。","",
        "## 执行和边界","",
        f'实测数值运行墙钟 {r["actual_run"]["numerical_wall_seconds"]:.3f} 秒，进程CPU {r["actual_run"]["numerical_process_cpu_seconds"]:.3f} 秒；Linux CPU亲和性限制为2个CPU，数值线程上限2，观察到{r["environment"]["observed_process_threads"]}个线程。环境与执行URL见RESULT.json。打包总耗时、zip字节和SHA256见PACKAGE.json。',"",
        "所有耗时仅单次CPU样本，不是吞吐基准；没有GPU、模型权重、付费API或训练。检测到桌面命令运行器为Mac后停止使用；未读取本机文件或模型缓存，实验在GitHub Actions隔离Linux运行。依赖只使用NumPy与Python标准库，在venv内优先尝试现有NumPy，缺失时只安装公开二进制wheel。","",
        "## 最值得接入主实验的接口","",
        "接入 read_details(handle, directory, rois, expected_rgb_sha256) -> (RGB crops, IO statistics)。先保存 source_shape/dtype、源RGB哈希、raw_bank句柄及语义摘要的独立句柄；ROI必须映射回源图像素坐标。主实验重建对照显式标注 has_source_bypass，把 feature_only 与 feature_plus_source 分栏，旁路关闭时不能静默读取源图。语义张量以后替换为真实Qwen输出，不改变原始库；生产接入需补充真实resize/crop坐标变换、视频帧ID/版本、授权及多请求缓存。"]
    (ROOT/"REPORT_ZH.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()

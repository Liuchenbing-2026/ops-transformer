# BF16 MegaMoe EP2 优化与复现（2026-09-07）

已完成 BF16 完整融合算子优化，最终保留版本为 **BF16 V3**。它覆盖路由、dispatch、两次专家矩阵计算、激活与 combine，是完整的 MoE 算子。算子已取得可重复收益；Qwen3.6 BF16 TP2/EP2 的整模型四轮短测仍比关闭慢32.05%，尚未取得整模型收益。

## 1. 已测收益

第一步批量搬运和专家并行分发，以原 BF16 算子／V1／V1／原算子顺序测试：

| 每 rank token／分布 | 原算子 ms | V1 ms | 耗时下降 |
|---|---:|---:|---:|
| uniform256 | 3.0369 | 3.0962 | -1.95% |
| uniform2048 | 12.0345 | 6.3417 | 47.30% |
| skew512 | 5.8259 | 3.5205 | 39.57% |
| concentrated1024 | 10.4498 | 5.7723 | 44.76% |

第二步省去 EP2 同机 combine 中的逐专家全核同步及不会执行的 RDMA 扫描，以 V1／V3／V3／V1 测试：

| 每 rank token／分布 | 本轮 V1 ms | V3 ms | 增量耗时下降 |
|---|---:|---:|---:|
| uniform256 | 3.1229 | 2.9784 | 4.63% |
| uniform2048 | 6.4577 | 6.2013 | 3.97% |
| skew512 | 3.5304 | 3.3265 | 5.77% |
| concentrated1024 | 5.7946 | 5.6220 | 2.98% |

每轮每场景预热10次、测量30次，使用 `perf_counter + torch.npu.synchronize`，跨rank barrier在计时区间外。每次取较慢rank，先求该轮中位数，再平均同版本两轮中位数。V3四个场景的两轮结果均优于两轮V1。不同时段的两张表不能直接叠加百分比；没有把旧原算子与V3的不同时段计时冒充同轮对照。

## 2. 整模型开启／关闭四轮短测

官方非量化 `/data2/weights/Qwen3.6-35B-A3B`，TP2/EP2、DP1/PCP1，物理卡0/5。相同输入4096 token、输出1 token、并发1；每轮预热4次、计时12次。测量是HTTP单请求延迟，包含整模型预填充及1个输出token，不是纯算子耗时，也不是AISBench吞吐。

| 轮次 | 中位数 ms |
|---|---:|
| on_F1 | 471.1843 |
| off_E1 | 357.6565 |
| off_E2 | 359.4592 |
| on_F2 | 475.7496 |

开启两轮均值 **473.4670 ms**，关闭两轮均值 **358.5578 ms**，开启耗时相对变化 **+32.05%**。本次没有新AISBench吞吐结果。原BF16开启版本此前4K短测905.5320 ms，V1为486.8552 ms，仅作历史参考。

这也解释了为什么降低通信小块和同步开销后，仍不能预设整模型必然超过关闭路径：本次DP1的MoE序列并行为False，关闭时已有完整输入，ALLGATHER准备阶段不需要再次收集输入；开启则先按TP切分，再逐专家收发。不能把旧FlashComm环境变量当作MoE序列并行已生效的证据。普通GMM的完整权重／专家列表在此前图执行隔离实验中性能接近，不足以解释大幅差距。

## 3. 修改与精度边界

最终仅改CANN `mc2/mega_moe/op_kernel/arch22/mega_moe_kernel_a2_bf16.hpp`：同rank/IPC数据改用32KiB双缓冲分块复制；EP2/40 AIV每波20专家、每专家两条peer通道；EP2同机combine省去逐专家同步，保留AIC就绪等待、MTE3排空及最终跨rank同步。不改变GMM算术、激活、舍入或输出累加过程。

150组静态地址、容量、通知与DMA边界检查覆盖44,100个地址。V1和V3均通过四种路由分布、两个rank的全部返回张量逐位比较，以及每组5次同步、20次连续异步重复调用。V3实际二进制选择已核对：两个进程各104次调用，tiling key为0。

两轮整模型开启均通过固定2K文本和logprobs、4K生成文本与旧BF16开启版本的精确对比，重复输出检查通过。独立普通BF16 GMM参考的累加／舍入路径不同，算子与它只要求容差和相对L2通过；不宣称逐位相同。上述证据不等于任意输入或全量任务的数学无损，也未新增BF16整模型GSM8K全量评测。

只对本报告EP2工作点做过真实NPU验收。其他EP规模保留串行分发调度但会使用分块复制，未在本次验证；本报告不外推到FP16或EP8 BF16。此前INT8 EP8 V4收益是独立结果。

两项实验已撤回：BF16 V2六行激活合批退化1.53%–4.65%；BF16 V4激活分界126/2→64/64退化0.24%–2.46%。两者均通过精度但无性能收益，源代码、制品和结果保留供复核，未进入最终实现。

## 4. 固定基线与制品

- CANN基线 `eaa08b425b5bc1e7af759b98599cbcec10d39323`；保留功能提交 `e204056851f94a8b60b386eaa7689a0b1f3381ba`。后续撤销实验的提交与该功能源码逐字节一致。
- vLLM-Ascend基线 `f4e1ad8a4bfff443e2072c829cb6f5867bf0e730`，BF16适配 `44abd7d1c06e1c2c642f03101db295654aceabef`，新增BF16门控及普通GMM扁平专家列表回退；已有161项CPU测试及真实NPU回退精度检查通过。本轮未改该源码。
- 分类：原创本地性能改动，没有引入未合并PR。已核对官方CANN上游 `119ea5f1c9f7fdffcc8a8559cddfbd29b9737e02`。
- 复用服务容器 `mm_bf16_serve_20260907`，镜像 `8dd949bff550c8e33211bbf6f0daa899c13eab3a1a140105bb2001509a6b568b`；vLLM0.27.1+empty、CANN9.1.0、Python3.12.13、Torch2.10.0+cpu、torch-npu2.10.0.post4。没有重建运行时。
- 算子H2048/I512/E256/top-k8，每rank128专家，BF16输入和ND权重，无量化。四个有效token规模256/2048/512/1024，每rank另加16个有效哨兵。环境HCCL_BUFFSIZE=200，MC2进程组实际206MiB，SymmBuffer总432013312字节，脚本显式检查。
- BF16对象 `MegaMoe_d95b9716a3c9814f6ca27f3b4f54fc18.o`，V3 SHA `3039de92bc2d7abe4e990903f1a600ec9e65e8e4baa0fa074096828bead65334`；原算子SHA `f0880dda0b15959ec995287b8e8e0e5676bb1b7d7b878d3f233d1ba6bf4b2ff9`。
- V3源码SHA `342358d91a779edebb5a10f83df337086cd3cb53fef725591a26eccb611eb8ee`。原包保留在 `/data1/megamoe_gain_20260905/package_v4`；该名称是旧INT8 V4包，包含未优化的BF16基线，区别于已撤回的本次BF16 V4。
- 构建依赖opbase固定 `ec8c387b251655664f4679f1bc5fd30a69efc5a7`，ops-tensor固定 `602d648ae62e0bb7b51494451e32e5c7bd50c1c3`。保留依赖独立Git元数据、已知工作树内容及哈希清单，不省略.git后触发CMake固定提交checkout失败。

## 5. 复现

[复现输入、源码差异和汇总包](bf16_ep2_reproduction_inputs.zip)，SHA256 `5b94d0295b1801b8f48217a83891f1b8945b0de079ff9fd6e7c9f71c243fa45a`。包内`manifest.json`逐文件记录SHA256；`source/`包括精确提交的git archive与差异，必须应用到对应基线。运行时、依赖二进制和模型权重另行使用已验证环境。原始完整证据包SHA256 `951ee2f127e5a20ba910d4f5c8e8b0fbdaa5bf7d21dcddb1350e3cb2694e2048`，本地归档名`bf16_final_performance_evidence.tar.gz`。

下面是已部署验收环境的**Linux Bash**命令。先确认卡0/5、端口8077/29541和worker空闲；每次使用新结果目录，不停止其他任务。完整启动环境、参数与身份清单随复现包提供。Linux源码必须用精确提交的git archive传输，校验哈希、列表和顶层布局后解压到独立目录；保留原运行时和包。

下面仅记录本次已经完成的构建命令，不要在保留的`source_v3`目录重复构建覆盖制品。重建时先从固定提交归档，在新目录恢复已校验依赖，并将下面的目录和vendor名改为新的复现专用名称；随后单独解包、核验制品SHA及动态库解析，再修改复现脚本的vendor路径。仅重跑计时可以直接使用保留的V1/V3包，无需重建。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/Ascend/cann-9.1.0/tools/bisheng_compiler/bin:$PATH
export CMAKE_BUILD_PARALLEL_LEVEL=32
cd /data1/megamoe_gain_20260905/bf16_opt_20260907/source_v3
bash build.sh --pkg --soc=ascend910b --ops=mega_moe --vendor_name=mmbf16v3 -j32
```

算子正确性及本轮V1/V3增量复现（以下在Linux宿主机运行）：

```bash
set -euo pipefail
r=/data1/megamoe_gain_20260905/bf16_opt_20260907
docker exec mm_bf16_serve_20260907 bash "$r/run_bench_v3.sh" v3 compare repro_v3
test "$(cat "$r/results/repro_v3/run.rc")" = 0
test "$(cat "$r/results/repro_v3/postflight.rc")" = 0
for spec in v1:repro_A1 v3:repro_B1 v3:repro_B2 v1:repro_A2; do
  docker exec mm_bf16_serve_20260907 bash "$r/run_bench_v3.sh"     "${spec%%:*}" perf "${spec##*:}"
  test "$(cat "$r/results/${spec##*:}/run.rc")" = 0 || exit 1
  test "$(cat "$r/results/${spec##*:}/postflight.rc")" = 0 || exit 1
done
```

`bench_bf16.py`固定seed、权重和路由输入，compare/perf核对已保留的`results/baseline_bf16_reference`输入／权重哈希及所有输出。新环境应先用原BF16包执行`baseline record baseline_bf16_reference`生成参考，并完成包选择、实际206MiB进程组和重复稳定性检查。不能把`complete_rank*.json`出现当作进程已结束，必须等待run.rc和postflight.rc均为0。

整模型原样复现需要使用新的任务目录：在随附`stage_model_v3_prefill.sh`、`start_model_v3.sh`及`summarize_model_v3_prefill.sh`中，将`model_v3`统一替换为新的目录名，例如`model_v3_repro01`，保留源码、模型、vendor与缓存路径。先运行stage，再运行start，等待controller_v2.rc和evaluation_v2.rc均为0，再运行summarize。stage会从保留基线复制完整辅助文件、cpu_tests.rc、model_ready.json及source_manifest.json，并创建源码和数据链接；不要只复制serve脚本后启动。替换后检查LF/UTF8无BOM及`bash -n`。`reproduce/model_v3/`保留本次实际执行脚本供比对。`run_pair_v2.sh`按ON/OFF/OFF/ON启动，逐轮执行精度、重复稳定性、4次预热和12次4K/1token请求，最后仅停止该轮记录的进程并检查资源释放。该脚本本次已去除AISBench段，不应宣称它复现4K/1K/C16吞吐。

静态布局检查器`verify_bf16_ep2_layout.py`接受三个位置参数：待核验BF16头文件绝对路径、该文件预期SHA256、新的输出JSON路径。它不使用固定Windows路径；实际性能仍以真实NPU比较为准。

原生profiling采用CANN msprof CLI非重放采集。向量、MTE和Cube活动会重叠，不应相加成阶段时间；Torch profiler曾在导出阶段产生不完整数据，未用于结论。方法参见[官方msprof说明](https://www.hiascend.com/document/detail/en/mindstudio/700/TITools/Profiling/atlasprofiling_16_0011.html)。

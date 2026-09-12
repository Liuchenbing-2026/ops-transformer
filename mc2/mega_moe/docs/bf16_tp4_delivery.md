# BF16 TP4 local-partial MegaMoe 交付

本分支 `codex/megamoe-bf16-e2e-20260907` 包含完整 BF16 内核、packed ND 权重扩展及后续本卡专家行范围优化。配套框架分支为 [Liuchenbing-2026/vllm-ascend 的 codex/megamoe-bf16-eval-20260907](https://github.com/Liuchenbing-2026/vllm-ascend/tree/codex/megamoe-bf16-eval-20260907)。

完整部署身份、启动和压测命令、精度要求、正反序复现包、整网与单算子结果均以框架仓库的 [BF16 交付报告](https://github.com/Liuchenbing-2026/vllm-ascend/blob/codex/megamoe-bf16-eval-20260907/docs/source/developer_guide/performance_and_debug/megamoe_bf16_delivery.md) 为统一入口。

## 精确源码和制品

- 内核验收源码：`23617531fd023f5a90612e6bddb8d21f951af234`。
- torch_extension 验收源码：`a97d4ace25f5f8a51ca4584ebab941249022d5c7`；到上述内核提交之间扩展源码无差异。
- BF16 ND 内核 SHA256：`325ec7a0b8060e8070f8156c8a685003d3be4ffd1a3b7a8f54051fcf6154a42b`。
- `npu_mega_moe.so` SHA256：`3d0fe0d4dd3f86a5917f701ed434f176e8df1be4a95031f3e90500fe55bfb705`。
- 18 机 vendor：`/data1/megamoe_gain_20260905/bf16_opt_20260907/package_v30/packages/vendors/mmbf16v30_transformer`。

当前交付在 TP4/EP4、DP1、非 SP、H2048/I512/E256/TopK8、BF16 ND 下使用完整复制输入、本卡专家部分输出。框架再将 routed 和 shared 的本卡部分结果按基线顺序相加，保留最终 BF16 TP AllReduce。输入附加 32 个哨兵行，最大真实 tokens=8192。

V30 将本卡行范围判断放入 unpermute，省去 expandedRowIdx 的一次全量读改写和同步，未修改浮点计算。旧 `replicated_input` 实验的提交历史已合并保留，当前复现使用 `local_partial_tp4`，不重新启用旧实验。

## 构建与运行边界

仅复现已测性能时，使用报告中保留且通过哈希验证的 V30 vendor 和 V26 扩展。模型、完整 CANN/驱动和容器镜像是外部依赖。

需要重建时，从本分支的固定提交使用 `git archive` 创建源码归档，核对文件列表与哈希，在新的目录构建。已验证环境为 CANN 9.1、Ascend 910B4-1；依赖 opbase 提交 `ec8c387b251655664f4679f1bc5fd30a69efc5a7`，ops-tensor 提交 `602d648ae62e0bb7b51494451e32e5c7bd50c1c3`，保留各依赖的 Git 元数据。

构建沿用仓库 `build.sh --pkg --soc=ascend910b --ops=mega_moe --vendor_name=<新的名称> -j32`，并设置 CANN 环境与 bisheng 编译器路径。扩展按 `torch_extension/README.md` 和仓库中的构建器编译。更换制品后必须重新做 loader/import、精度、重复稳定性和性能验收；不能只以源码相同推断二进制哈希或收益相同。

## 已验证结果

首次整网四轮复现的 4K/6K 吞吐增幅为 1.388% / 1.332%。2026-09-13 在 18 机使用交付输入包重新跑四轮，吞吐分别提高 0.858% / 1.475%；4K TTFT 变慢 5.251%，6K TTFT 改善 2.643%。不能承诺 TTFT 稳定提升。

最新单算子复测的 8192 tokens 四卡瓶颈设备区间约快 9.4%，但含准备和最终 AllReduce 的独立调用仍慢约 7.3%。这两个范围不同，详见统一报告，不能写成单算子全面加速。

算子精度包含此前 56 项 case/rank 边界用例，以及最新同版 48 项普通计时用例；均检查 BF16 逐位一致。框架固定请求和重复稳定性检查也通过。这些有限用例不是全模型、全输入无损的证明。

# AGENTS.md

给 Codex、Claude Code 的长期开发规则。`CLAUDE.md` 只引用本文件。服务器、启动方式、hook 和分工的规则在服务器的 `~/.codex/AGENTS.md`。

本仓库是 [Data-Juicer Sandbox](https://github.com/datajuicer/data-juicer-sandbox) 的 fork，在上游基础上做科学数据的配比实验：按候选配比采样、训练、评测，再建议下一轮配比。

## 改动原则

- 能随时合并上游：新功能放在新模块里，上游文件只做注册所需的最小改动；不格式化、不重排、不批量改名上游代码。
- 按领域分目录：hook 和 worker 放 `data_juicer_sandbox/specific_hooks/<领域>/`，实验配置放 `configs/<领域>/`，测试放 `tests/<领域>/`。
- 自定义 hook 通过 `HOOK_MAPPING` 注册，不改上游的 pipeline 和已有 hook。hook 只负责交接，训练和评测在 worker 自己的环境里运行。
- 先用上游已有的机制和成熟库，不重新实现已有的算法。
- 从其他项目复制的代码放在包内的 `data_juicer_sandbox/thirdparty/<来源>/`（随包安装，不新增顶层包名），附原许可证和注明来源仓库、路径、commit 的 NOTICE；复制的部分不改，适配写在我们自己的代码里。
- 不夹带无关改动，不复制粘贴大段代码。注释、测试名和提交说明用英文，与上游一致。

## 测试

- 改行为先写会失败的测试，确认它失败，再实现到通过。修 bug 先写复现测试。
- 测试和代码在同一次提交里，目录与源码对应。删除或改名功能时同步删改测试。
- 单元测试用小的合成数据，不依赖网络、GPU 和服务器上的大文件；训练用替身或极小模型。真实训练和配比实验写成配置，在后台运行，输出放 `runs/`，不放进单元测试。
- 合入 main 前必须通过：改动所在领域的测试，以及改动文件上的 pre-commit。

  ```bash
  python -m pytest tests/<领域> -q
  pre-commit run --files <改动的文件>
  ```

- 测试失败不靠跳过、删断言或标记 xfail 过关。上游原有的失败测试要在 REPORT 里注明，并说明与本次改动无关的依据。

## 工具

- 环境：用提示词指定的 `envs/` 下的环境；不往 `.venv` 和别人的环境装包。依赖改动写进 `pyproject.toml` 并更新 `uv.lock`。
- 格式和检查：用仓库的 pre-commit 配置，只对改动文件运行。
- git：Codex 只用只读命令（`status`、`diff`、`log`、`show`）和 `worktree add`；commit、merge、rebase、reset、push、改 remote 由 Claude 做。

## 分支

- `main` = 上游 + 我们合入的改动。开发在 worktree 的功能分支上，Claude 审查后合入 main。
- remote：`origin` 是 fork，`upstream` 是官方仓库（push 已禁用）。同步上游由 Claude 用 merge 完成。

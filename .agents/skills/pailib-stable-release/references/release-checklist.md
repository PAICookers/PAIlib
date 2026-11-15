# 稳定版发布检查清单

按本次检查面阅读，不要求每次加载全部内容。命令示例从仓库根目录运行，
以当前配置为准；本文件不固定未来版本号或工具版本。

## 版本和产物

- 项目分发名为 `paicorelib`。以 `pyproject.toml` 的 PEP 621 元数据为准，
  确认目标稳定版本、`requires-python`、Python classifiers、依赖及平台 marker、
  license/license-files、README、项目链接和 build-system 相互一致。
  不在发版准备时顺带提高最低 Python 或依赖范围。
- 当前使用 `uv_build` 和 `src/paicorelib` 布局，主锁文件为 `uv.lock`；
  以实际文件为准，不因旧任务提过 Poetry 就创建或要求 Poetry 锁文件。
  改版本/依赖后更新 uv 锁文件，再运行 `uv lock --check`；若锁文件变化超出
  目标版本和必要依赖，先解释原因，不顺手全量升级。
- 检查 `src/paicorelib/__init__.py` 的真实版本来源。目前使用分发元数据，
  不添加第二份手写版本常量；`tests/test_versions.py` 的宽松下界断言不足以
  证明目标版本一致，还应检查实际构建/安装后的元数据与 `__version__`。
- 运行 `uv build --no-sources`，同时检查 wheel 与 sdist，而非只检查命令退出码。
  输出目录使用本次专用目录或确认其中只含本次产物；不要误验旧构建文件。
- 检查 wheel 的 `METADATA` 和 sdist 的 `PKG-INFO`：Name、Version、
  Requires-Python、Requires-Dist、许可证与项目链接应符合配置。
  核对所需包模块、许可证、说明文档进入对应产物；开发配置/技能不要求进入 wheel。
  检查没有本地缓存、凭据、原始调试数据或源码检出路径混入分发包。
- 在源码目录之外的独立环境安装本次 wheel，运行导入、版本和 README 的最小示例。
  清除 `PYTHONPATH=src`/可编辑安装等干扰，确认导入来自安装位置；避免重新使用
  已安装旧版本。必要时从本次 sdist 重建 wheel 验证可复现的打包输入。
  环境/网络不具备时明确记录安装验证未执行，不能用源码测试代替。

## 自动化

以现有流程为起点逐项核对，有真实缺陷才最小修正。

| 文件 | 需要确认的发布契约 |
| --- | --- |
| `.github/workflows/codecov.yml` | PR/push 分支与 paths 过滤是否覆盖本次变化；Python/OS 矩阵与支持范围一致；安装和测试使用锁文件；区分 pytest 与覆盖上传失败 |
| `.github/workflows/release-drafter.yml` 与 `.github/release-drafter.yml` | 草稿触发分支、标签类别和 semver 推断一致；未使用废弃字段；token 权限适合实际行为；草稿不等于已发布 |
| `.github/workflows/release.yml` | `release.published` 事件、checkout 的 release tag、tag 与项目版本检查、构建和发布目标相互一致 |
| `.github/dependabot.yml` | 位于 `.github/`，不是 `.github/workflows/`；默认分支上存在配置后才生效；target-branch 决定更新 PR 目标，不能替代默认分支要求 |
| `.pre-commit-config.yaml`、`ruff.toml` | 检查器、格式化器、锁文件检查和当前代码规范相容；不重复引入 Black 或无关的大范围格式修改 |

- 当前 tag 约定为 `vX.Y.Z`，发布 workflow 会与项目版本拼接的 `v` 前缀比较。
  检查目标 tag 尚未占用，以及 GitHub Release/PyPI 是否已存在同版；无远端访问时
  标注未知，不覆盖标签、不尝试重复上传。历史撤回记录也不能当作可复用版本号。
- 当前发布使用 `uv publish`，权限包含 `id-token: write`，环境名为 `pypi`。
  核对是否有可用的 OIDC trusted publisher（仓库、workflow、environment 匹配）
  或另行明确配置的认证；认证和 environment 设置可能不在 Git 文件中。
  只能查看时报告证据，不能擅自新建 secret、修改环境保护或索要凭据正文。
- 当前 Release Drafter 的 `publish: false` 与 `github.token` 用于写草稿。
  不为历史 token 设计机械换回 PAT。只有准备改变自动发布链路时才核对
  GITHUB_TOKEN 产生的事件能否触发下游；跨 workflow 的环境变量不会自动传递。
- 解析 YAML 并检查 workflow 的事件、jobs、表达式和权限；YAML 能解析不代表
  GitHub 接受整个 workflow。现有工具可用时做 Actions 专用静态检查。
  特别注意 YAML 1.1 解析器可能把 `on` 解析为布尔值，不能据此误判触发器缺失。
- 查看候选提交实际触发的 CI，而不只看最后一次成功记录；仅改文档时 paths
  过滤可能导致测试未触发，记录此事实。发现源码/打包缺陷需修复后重验，不能
  将缺失检查当成功。`ubuntu-latest` 迁移 notice 本身不构成失败证据。

## 验证与交付

真正准备稳定版时，按当前配置完成下列本地检查。单纯编写/修订本技能不等于
执行发布准备，不需要因此运行项目全套测试或修改版本。

准备模式的最小文件动作顺序是：确认基线和候选差异；更新项目版本并锁定；新增
目标 CHANGELOG 区块并修复确定的仓库内链接；同步双语 README；修正确定错误的
Dependabot/CI paths；再运行下面的验证。每一步都先查看 diff，避免覆盖用户修改。
如果候选差异不足以支持发布说明，保留事实边界并报告缺少的证据，不编造条目。

```sh
uv lock --check
uv sync --locked --dev
uv run --locked pytest -q
ruff check src tests
ruff format --check src tests
uv build --no-sources
git diff --check
```

- Ruff 命令需要可用的项目兼容版本；优先使用项目现有环境或 pre-commit 固定版本，
  未安装时报告/按约定准备工具，不把当前未声明 Ruff 的 dev 组当作安装保证。
- 涉及公共行为时验证全套测试；失败要定位是新回归、原有故障还是环境限制，
  不能静默跳过。仅完成本机 Python/Linux 检查，不代表 Windows、所有 Python
  版本或真板均通过；补充与候选提交匹配的 CI 证据。
- 手工检查 CHANGELOG：第一版是目标稳定版，其余已发布版本语义升序；移动完整
  区块不改写历史，含撤回/预发布条目不丢失，版本号不重复；Unreleased 独立保留。
  例如原序列 `2.0.0, 1.0.0, 1.10.0`，发布 `2.1.0` 后应为
  `2.1.0, 1.0.0, 1.10.0, 2.0.0`，不是全降序或字符串升序。
- 核对 README 双语一致性、示例、相对链接和变更相关说明。共享 AGENTS 或
  开发规范若尚未跟踪，要作为可审阅的新文件提出；不能依赖它在别人机器上存在。
  不引用个人任务记录、外部未共享 PDF、用户目录或只在本机存在的技能路径。
- 按需检查/刷新索引；检查图文件更新不夹带 artifact.json。没有 MCP 或持久化图时，
  明确报告未更新；不得编造图查询成功或为此偷偷安装全局技能/工具。
- 汇报文件清单和待提交内容时使用显式路径，避免 `git add .` 将无关文件、
  生成的元数据或个人资料带入仓库。提交/打标签/推送/发布均按用户另行授权执行。
- 给出结论：本地准备就绪 / 有阻塞 / 证据不足，并分别列出检查命令及结果、
  远端提交/运行证据、未验证平台与剩余发布步骤。任何“就绪”结论都不代表已发布。

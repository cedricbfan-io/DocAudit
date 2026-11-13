# DocAudit 快速上手（网页版 demo）

## 1. 启动（先看这里）

在解压后的 `app/` 文件夹里打开终端：

```bash
# ① 第一次：建环境（必须 Python 3.12）
# Windows:  py -3.12 -m venv .venv
python3.12 -m venv .venv
# Windows:  .venv\Scripts\python -m pip install -r requirements.txt
.venv/bin/python -m pip install -r requirements.txt

# ② 每次：打开网页界面
# Windows:  .venv\Scripts\python -m src.ui.app     （或双击 start_ui.bat）
.venv/bin/python -m src.ui.app
```

- 网址：**http://127.0.0.1:7860** （浏览器会自动打开；没打开就手动复制这个地址）
- **怎么知道已经好了**：终端出现 `Running on local URL:  http://127.0.0.1:7860` 这一行。之前只看到终端输出，是因为跑的是 `python -m src.extract.extract`（命令行抽取，只打印 JSON），网页界面是 `src.ui.app`。
- 停止：在终端按 `Ctrl+C`。端口被占用：`python -m src.ui.app --port 7861`。
- 不需要 `.env` / API key：图片流程只用本地 PaddleOCR + 规则。第一次点 Analyse 会下载 PaddleOCR 模型（需要联网，约 1 分钟）。
- 依赖坑：`numpy` 必须 < 2，`opencv-python` 必须 < 4.12（requirements 已经固定）。numpy 2 下 paddle 会在 Analyse 时报错。

## 2. 用法（图片 → 审核 → 导出）

1. **Image (full pipeline)** 标签页：上传小票照片 → **Analyse**（真实 OCR，demo 图在 Mac CPU 上约 5–20 秒）。
2. 看表格：每一行有状态和 “Why flagged” 原因；右边证据图上红框 = 有告警的行，蓝框 = 当前选中行。
3. 下拉框选 **任意一行或 TOTAL** → **Accept / Reject**（可以改主意）；**Correct** = 填新数值并视为接受。
4. 规则：
   | 行 | 没操作 | 点了 Accept | 点了 Reject |
   |---|---|---|---|
   | 没告警 | 自动接受 | 接受 | 拒绝 |
   | 有告警 | **自动拒绝** | 接受 | 拒绝 |
5. **只有所有行和 TOTAL 都是接受状态才能导出**。页面会写出 “⛔ Export blocked … 哪一行 / TOTAL”。TOTAL 被 reject 也会挡住整张小票（即使 ledger.csv 没有 total 列）。
6. **Export receipt** → 下面出现 `ledger.csv` 和 `audit_export.json` 的下载链接，同时存到 `exports/<小票>_<时间>/`。之后只要改任何决定，下载链接就清空，避免下载旧版本。
7. 换一张小票重新 Analyse，之前的决定全部清空。

**JSON fixture (no OCR)** 标签页：上传 `demo/*.json`，**不经过 OCR**，用来演示检测规则。

## 3. Demo 文件

| 文件 | 走 OCR? | 预期 |
|---|---|---|
| `demo/test_image_clean.png` | 是 | 无告警，不点任何按钮直接可导出 |
| `demo/test_image_duplicate_item.jpg` | 是 | 两行 Plastic 重复 + 商品合计 62.00 ≠ TOTAL 31.00 → 两行和 TOTAL 都要审核 |
| `demo/test_image_incorrect_total.jpg` | 是 | `Total Qty` 377.00 ≠ 商品合计 327.00；第 0、1 行 OCR 置信度低 ⚠️ |
| `demo/demo_clean.json` 等 6 个 + `demo_duplicate_62_31.json`、`demo_missing_rows.json` | 否 | 见 CHANGES_2026-11-11.md |

## 4. 命令行（同一套规则）

```bash
python -m src.ui.cli demo/test_image_clean.png                 # 无告警 → 导出
python -m src.ui.cli demo/demo_multi_fault.json                # 有未审核告警 → BLOCKED，退出码 2
python -m src.ui.cli demo/demo_multi_fault.json --accept 0,2,3 --accept-total   # → 导出
python -m pytest -q tests/                                     # 测试
```

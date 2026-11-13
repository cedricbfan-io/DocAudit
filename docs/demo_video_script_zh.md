# Demo 录屏脚本（60–90 秒，简体中文）

> 本机环境里无法直接录视频。下面每一步都已在真实界面上跑过（2026-11-13，Mac，真实 PaddleOCR），
> 对应截图在 `docs/demo_screenshots/`。录屏时照着做即可。

| 时间 | 画面 / 操作 | 旁白 |
|---|---|---|
| 0–8 秒 | 终端运行 `python -m src.ui.app`，出现 `Running on local URL: http://127.0.0.1:7860`，浏览器自动打开 | “这是 DocAudit 的网页 demo，本地运行，不需要 API key。” |
| 8–20 秒 | Image 标签页上传 `demo/test_image_duplicate_item.jpg` → 点 **Analyse** | “上传一张小票，PaddleOCR 读出文字和坐标，规则把它们整理成商品行和总额。” |
| 20–35 秒 | 表格：两行 Plastic 都是 *Auto-rejected*，TOTAL 也是；右侧 “Why flagged” 写着商品合计 62.00、小票总额 31.00，差额正好等于重复的那一行。下方 **⛔ Export blocked** 列出三项。（截图 1） | “系统发现同一商品出现两次，而且商品加起来 62 块，小票只印 31 块。有告警的行默认是拒绝，所以现在不能导出。” |
| 35–45 秒 | 下拉框选 Row 0 → **Accept**；仍然 blocked（截图 2）。选 Row 1 → **Reject**（截图 3） | “我们可以一行一行决定。接受一行之后，其他没处理的行还挡着；拒绝任何一行，整张小票都不能导出。” |
| 45–60 秒 | Row 1 改成 **Accept** → 点 **Accept all remaining flagged entries**（TOTAL）→ **✅ Ready to export**（截图 4） | “改主意也可以，状态马上重新计算。所有行和 TOTAL 都接受后才可以导出。” |
| 60–75 秒 | 点 **Export receipt** → 出现 ledger.csv 和 audit_export.json 下载（截图 5）；打开 CSV：2 行，状态 accepted，原来的告警原因还保留在 warnings 列 | “导出的文件保留了原始告警和人工决定，方便追溯。” |
| 75–90 秒 | 选 TOTAL → **Reject**：下载链接立刻消失，导出按钮变灰（截图 6）。再上传 `test_image_clean.png` → Analyse → 不点任何按钮直接 Export | “改了决定，旧文件就不能再下载。换一张干净的小票，之前的决定不会带过来，没有告警可以直接导出。” |

只在代码层面验证、录屏里没有出现的：Correct 按钮、命令行 `python -m src.ui.cli`、JSON 标签页的 `demo_missing_rows.json`（“总额无法核对”）——这些由 `tests/test_ui_review.py`、`tests/test_cli_export.py` 覆盖。

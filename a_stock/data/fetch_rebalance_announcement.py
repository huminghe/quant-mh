"""
中证指数公司样本股定期调整公告抓取（指数调整效应实盘接入用）

背景：第九轮已验证指数样本股调整效应（中证500，T+1建仓-生效日前1个
交易日退出，净超额+1.11%，p=0.003，详见research.md）通过完整组合回测，
但回测用的调入调出名单来自tushare的index_weight月末快照——这个数据只有
"事后"才能拿到（调整生效之后的月末快照才会体现新名单），无法满足实盘
"生效日前提前建仓"的时间要求。中证指数公司会在生效日前约2周发布调整
公告，本脚本抓取该公告，尽早拿到官方名单。

接口来源：中证指数官网公告页(https://www.csindex.com.cn/#/announcement)
是前端SPA，公告数据由异步JSON接口提供，已实测逆向确认：
- 列表接口 queryAnnouncementByType：只返回每个分类最新5条，不支持真分页
  （pageNum/pageSize参数不生效，已实测确认，不要浪费时间调试分页）。
  分类里 indexrebalancingAnnouncements（theme=index_rebalance）是指数
  样本调整类公告，但混杂全市场所有中证系列指数（中证500/沪深300/北证50/
  科创50等），需要用标题关键词自己筛选目标指数。
- 详情接口 queryAnnouncementById：返回HTML正文 + enclosureList附件数组，
  样本名单以xlsx附件形式提供（不是正文表格，也不是纯文字列举）。

已知限制（不要试图用代码修复，只能靠人工兜底，已在下方WARNING里注明）：
中证500每年只调整2次（6月/12月），若近期公告较多（其他指数扎堆发布），
目标公告可能在这5条免登录窗口内被挤出而抓不到。IMPORTANT：6月/12月
生效日前2周左右，除了跑本脚本，仍需人工登录官网核实一次，不能完全依赖
自动化（YAGNI——写复杂的登录后翻页逆向成本远超2次/年人工核对的成本）。

用法：
  cd a_stock/data
  python fetch_rebalance_announcement.py            # 检查是否有新的中证500调整公告
  python fetch_rebalance_announcement.py --dry-run   # 只打印候选公告，不下载解析
"""

import re
import argparse
import pathlib
import tempfile

import requests
import pandas as pd

DATA_DIR = pathlib.Path(__file__).parent
MEMBERS_FILE = DATA_DIR / "hs500_members.parquet"
PENDING_FILE = DATA_DIR / "rebalance_pending.parquet"  # 已抓到但尚未被信号脚本处理的调整事件

BASE_URL = "https://www.csindex.com.cn"
LIST_URL = f"{BASE_URL}/csindex-home/announcement/queryAnnouncementByType"
DETAIL_URL = f"{BASE_URL}/csindex-home/announcement/queryAnnouncementById"
HEADERS = {
    "Referer": "https://www.csindex.com.cn/",
    "User-Agent": "Mozilla/5.0 (compatible; quant-research-bot/1.0)",
}

# 标题关键词：必须同时含"中证500"和"样本"两类词才算命中，避免误抓沪深300/
# 北证50等其他指数的调整公告（各指数公告标题格式一致，只是指数名称不同）。
TARGET_INDEX_KEYWORD = "中证500"
ADJUSTMENT_KEYWORDS = ["样本", "调整"]


def fetch_announcement_list() -> list[dict]:
    """拉取公告分类列表，返回指数样本调整类(indexrebalancingAnnouncements)公告"""
    resp = requests.get(LIST_URL, params={"pageNum": 1, "pageSize": 10},
                         headers=HEADERS, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", {}).get("indexrebalancingAnnouncements", [])


def is_target_announcement(title: str) -> bool:
    return TARGET_INDEX_KEYWORD in title and any(k in title for k in ADJUSTMENT_KEYWORDS)


def fetch_announcement_detail(ann_id: str) -> dict:
    resp = requests.get(DETAIL_URL, params={"id": ann_id}, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    return resp.json().get("data", {})


def download_xlsx(url: str) -> pathlib.Path:
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    tmp = pathlib.Path(tempfile.mkstemp(suffix=".xlsx")[1])
    tmp.write_bytes(resp.content)
    return tmp


CODE_COL_PATTERN = r"证券代码|股票代码|成分券代码"


def find_header_row(raw: pd.DataFrame) -> int:
    """
    实测中证指数公司xlsx附件表头不在第一行（前面有标题行+空行，见
    北证50样本股名单.xlsx：第0行是标题，第1行空，第2行才是真实表头），
    需要扫描前几行找到含"证券代码"关键词的那一行作为表头。
    """
    for i in range(min(10, len(raw))):
        row_values = raw.iloc[i].astype(str)
        if row_values.str.contains(CODE_COL_PATTERN, regex=True, na=False).any():
            return i
    raise ValueError("在前10行内未找到含证券代码关键词的表头行")


def find_backup_list_row(raw: pd.DataFrame, start: int) -> int:
    """
    实测xlsx在正式样本名单之后常附带"备选名单"区块（候补池，非正式样本，
    只在正式样本停牌/退市时启用，不应计入调入调出diff，否则会污染信号），
    从start行开始扫描，找到含"备选"关键词的标题行，返回其行号作为正式
    样本名单的结束边界（不含该行）。找不到则说明没有备选区块，返回总行数。
    """
    for i in range(start, len(raw)):
        row_values = raw.iloc[i].astype(str)
        if row_values.str.contains("备选", na=False).any():
            return i
    return len(raw)


def parse_sample_list(xlsx_path: pathlib.Path) -> list[str]:
    """
    解析样本名单xlsx，提取正式成分股代码列（不含备选名单，见find_backup_list_row）。
    附件表头格式未知（不同批次公告可能不完全一致，且表头前常有标题/空行，
    见find_header_row），动态定位表头行后，用正则从列名里找
    "证券代码"/"股票代码"类列，兼容600000/600000.SH两种代码格式，
    统一补全交易所后缀（.SH/.SZ，按代码首位数字规则判断）。
    """
    raw = pd.read_excel(xlsx_path, header=None)
    header_row = find_header_row(raw)
    backup_row = find_backup_list_row(raw, header_row + 1)
    df = pd.read_excel(xlsx_path, header=header_row, nrows=backup_row - header_row - 1)

    code_col = None
    for col in df.columns:
        if re.search(CODE_COL_PATTERN, str(col)):
            code_col = col
            break
    if code_col is None:
        raise ValueError(f"未在xlsx表头中找到代码列，实际表头：{list(df.columns)}")

    codes = []
    for raw_val in df[code_col].dropna():
        code = str(raw_val).strip()
        if not re.fullmatch(r"\d+(\.\d+)?", code):
            continue
        code = str(int(float(code))).zfill(6)
        if code.startswith(("6",)):
            codes.append(f"{code}.SH")
        else:
            codes.append(f"{code}.SZ")
    return sorted(set(codes))


def diff_against_latest_snapshot(new_codes: list[str]) -> tuple[list[str], list[str]]:
    """与本地hs500_members.parquet最新快照对比，得出调入(added)/调出(removed)"""
    if not MEMBERS_FILE.exists():
        raise FileNotFoundError(f"缺少{MEMBERS_FILE}，无法diff，请先运行fetch_index_members.py --index hs500")
    members = pd.read_parquet(MEMBERS_FILE)
    latest_date = members["trade_date"].max()
    old_codes = set(members[members["trade_date"] == latest_date]["con_code"])
    new_set = set(new_codes)
    added = sorted(new_set - old_codes)
    removed = sorted(old_codes - new_set)
    return added, removed


def save_pending(ann: dict, added: list[str], removed: list[str]) -> None:
    row = {
        "ann_id": ann["id"], "title": ann["title"], "publish_date": ann["publishDate"],
        "added": ",".join(added), "removed": ",".join(removed),
    }
    pd.DataFrame([row]).to_parquet(PENDING_FILE, index=False)
    print(f"已保存待处理调整事件至 {PENDING_FILE}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="只打印候选公告标题，不下载解析")
    args = parser.parse_args()

    print("拉取中证指数公司公告列表...")
    announcements = fetch_announcement_list()
    print(f"最新{len(announcements)}条指数调样类公告：")
    for ann in announcements:
        print(f"  [{ann['publishDate']}] {ann['title']}")

    targets = [a for a in announcements if is_target_announcement(a["title"])]
    if not targets:
        print(f"\n未发现{TARGET_INDEX_KEYWORD}样本调整公告（可能已被挤出5条窗口，"
              f"请人工登录官网核实：{BASE_URL}/#/announcement）")
        return

    ann = targets[0]
    print(f"\n命中目标公告：[{ann['publishDate']}] {ann['title']}")
    if args.dry_run:
        return

    detail = fetch_announcement_detail(ann["id"])
    enclosures = detail.get("enclosureList", [])
    xlsx_enclosures = [e for e in enclosures if str(e.get("fileName", "")).endswith((".xlsx", ".xls"))]
    if not xlsx_enclosures:
        print(f"该公告无xlsx附件（附件列表：{[e.get('fileName') for e in enclosures]}），"
              f"可能是需要去指数详情页查询拟生效样本的类型，需人工处理")
        return

    xlsx_path = download_xlsx(xlsx_enclosures[0]["fileUrl"])
    codes = parse_sample_list(xlsx_path)
    xlsx_path.unlink()
    print(f"解析到 {len(codes)} 只样本股")

    added, removed = diff_against_latest_snapshot(codes)
    print(f"调入 {len(added)} 只：{added}")
    print(f"调出 {len(removed)} 只：{removed}")

    save_pending(ann, added, removed)


if __name__ == "__main__":
    main()

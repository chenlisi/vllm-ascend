#!/usr/bin/env python3
"""
规则检索脚本 - vllm-ascend AI Reviewer Skill

用法:
  # 按关键词搜索规则
  python search_rules.py search "环境变量"

  # 按 PR 编号查找相关规则
  python search_rules.py pr 10023

  # 按维度筛选规则
  python search_rules.py dim correctness

  # 按 report_id 读取具体报告
  python search_rules.py report PR10023_001

  # 按 PR 编号读取所有相关报告
  python search_rules.py reports 10023

  # 列出所有规则（按 pr_count 排序）
  python search_rules.py list

  # 按文件路径匹配相关规则
  python search_rules.py match "vllm_ascend/envs.py"
"""

import json
import os
import sys
import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SKILL_DIR = os.path.dirname(SCRIPT_DIR)
REFERENCES_DIR = os.path.join(SKILL_DIR, "references")
REPORTS_DIR = os.path.join(REFERENCES_DIR, "reports")
RULE_SUMMARY = os.path.join(REFERENCES_DIR, "_rule_summary.json")

# 文件路径 -> 规则关键词映射
PATH_KEYWORDS = {
    "envs.py": ["环境变量", "env", "os.environ", "envs"],
    "patch/": ["patch", "补丁", "注册", "__init__"],
    "ops/": ["算子", "kernel", "triton", "aclnn", "npu"],
    "attention/": ["attention", "注意力", "mla", "dsa", "sfa", "fa3"],
    "worker/": ["worker", "model_runner", "block_table", "scheduler"],
    "models/": ["model", "模型", "weight", "权重", "loading"],
    "distributed/": ["distributed", "分布式", "tp", "pp", "dp", "ep"],
    "tests/": ["test", "测试", "unittest", "pytest", "coverage"],
    "docs/": ["doc", "文档", "教程", "tutorial", "deployment"],
    "tools/": ["tool", "工具", "benchmark", "bisect"],
    "compilation/": ["compilation", "acl_graph", "acl graph", "图重放"],
    "quantization/": ["quantiz", "量化", "fp8", "dtype"],
    "sample/": ["sampler", "sampling", "采样", "penalty"],
    "lora/": ["lora", "punica", "adapter"],
    "spec_decode/": ["speculative", "推测解码", "eagle", "mtp"],
    "eplb/": ["eplb", "expert", "专家", "moe"],
    "kv_offload/": ["kv_offload", "offload", "卸载", "cpu_npu"],
    "CMakeLists": ["cmake", "build", "构建", "编译"],
    "Dockerfile": ["docker", "镜像", "container", "容器"],
    ".github/": ["ci", "workflow", "pipeline", "流水线"],
}


def load_rules():
    with open(RULE_SUMMARY, "r", encoding="utf-8") as f:
        return json.load(f)


def find_report_files(pr_number):
    """查找某个 PR 的所有报告文件"""
    pattern = os.path.join(REPORTS_DIR, "*", f"PR{pr_number}_*.json")
    return sorted(glob.glob(pattern))


def load_report(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cmd_search(keyword):
    """按关键词搜索规则"""
    rules = load_rules()
    results = []
    for rule in rules:
        desc = rule["rule_description"]
        if keyword.lower() in desc.lower():
            results.append(rule)

    if not results:
        print(f"未找到包含 '{keyword}' 的规则")
        return

    print(f"找到 {len(results)} 条包含 '{keyword}' 的规则:\n")
    for i, rule in enumerate(results, 1):
        prs = ", ".join([f"PR{p['pr_number']}" for p in rule["prs"][:5]])
        if len(rule["prs"]) > 5:
            prs += f" ...等{len(rule['prs'])}个"
        print(f"{i}. [{prs}]")
        print(f"   {rule['rule_description']}")
        if rule.get("checkpoints"):
            for cp in rule["checkpoints"][:3]:
                print(f"   - {cp}")
        print()


def cmd_pr(pr_number):
    """按 PR 编号查找相关规则"""
    rules = load_rules()
    results = []
    for rule in rules:
        prs = [p["pr_number"] for p in rule["prs"]]
        if pr_number in prs:
            results.append(rule)

    if not results:
        print(f"未找到 PR#{pr_number} 相关的规则")
        return

    print(f"PR#{pr_number} 涉及 {len(results)} 条规则:\n")
    for i, rule in enumerate(results, 1):
        dim = rule["prs"][0].get("dimension", []) if rule["prs"] else []
        print(f"{i}. [{','.join(dim)}] {rule['rule_description']}")
        if rule.get("checkpoints"):
            for cp in rule["checkpoints"][:3]:
                print(f"   - {cp}")
        print()


def cmd_dim(dimension):
    """按维度筛选规则"""
    rules = load_rules()
    results = []
    for rule in rules:
        dims = rule["prs"][0].get("dimension", []) if rule["prs"] else []
        if dimension in dims:
            results.append(rule)

    if not results:
        print(f"未找到维度 '{dimension}' 的规则")
        return

    print(f"维度 '{dimension}' 共 {len(results)} 条规则:\n")
    for i, rule in enumerate(results, 1):
        prs = ", ".join([f"PR{p['pr_number']}" for p in rule["prs"][:3]])
        print(f"{i}. [{prs}] {rule['rule_description']}")
    print()


def cmd_report(report_id):
    """读取具体报告"""
    # 搜索所有类别目录
    for subdir in os.listdir(REPORTS_DIR):
        subdir_path = os.path.join(REPORTS_DIR, subdir)
        if os.path.isdir(subdir_path):
            path = os.path.join(subdir_path, f"{report_id}.json")
            if os.path.exists(path):
                report = load_report(path)
                print(json.dumps(report, ensure_ascii=False, indent=2))
                return

    print(f"未找到报告 {report_id}")


def cmd_reports(pr_number):
    """读取某个 PR 的所有报告"""
    files = find_report_files(pr_number)
    if not files:
        print(f"未找到 PR#{pr_number} 的报告")
        return

    print(f"PR#{pr_number} 共 {len(files)} 份报告:\n")
    for f in files:
        report = load_report(f)
        rule = report.get("rule", {})
        print(f"--- {os.path.basename(f)} ---")
        print(f"维度: {report.get('dimension', [])}")
        print(f"严重度: {report.get('severity', '')}")
        print(f"规则: {rule.get('description', '')}")
        if rule.get("checkpoints"):
            print(f"检查点:")
            for cp in rule["checkpoints"]:
                print(f"  - {cp}")
        if rule.get("antipattern", {}).get("code"):
            code = rule["antipattern"]["code"][:200]
            print(f"反模式: {code}...")
        if rule.get("pattern", {}).get("code"):
            code = rule["pattern"]["code"][:200]
            print(f"正模式: {code}...")
        print()


def cmd_list():
    """列出所有规则"""
    rules = load_rules()
    print(f"共 {len(rules)} 条规则:\n")
    for i, rule in enumerate(rules, 1):
        prs = ", ".join([f"PR{p['pr_number']}" for p in rule["prs"][:3]])
        dim = rule["prs"][0].get("dimension", []) if rule["prs"] else []
        print(f"{i:3d}. [{','.join(dim):15s}] [{prs}] {rule['rule_description'][:80]}")


def cmd_match(file_path):
    """按文件路径匹配相关规则"""
    rules = load_rules()
    keywords = []
    for pattern, kws in PATH_KEYWORDS.items():
        if pattern in file_path:
            keywords.extend(kws)

    if not keywords:
        print(f"未找到 '{file_path}' 的匹配关键词")
        return

    results = []
    for rule in rules:
        desc = rule["rule_description"].lower()
        score = sum(1 for kw in keywords if kw.lower() in desc)
        if score > 0:
            results.append((score, rule))

    results.sort(key=lambda x: -x[0])

    print(f"文件 '{file_path}' 匹配到 {len(results)} 条规则 (关键词: {keywords}):\n")
    for score, rule in results[:20]:
        prs = ", ".join([f"PR{p['pr_number']}" for p in rule["prs"][:3]])
        print(f"  [匹配度:{score}] [{prs}] {rule['rule_description'][:80]}")
    if len(results) > 20:
        print(f"\n  ... 还有 {len(results) - 20} 条")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]

    if cmd == "search" and len(sys.argv) >= 3:
        cmd_search(" ".join(sys.argv[2:]))
    elif cmd == "pr" and len(sys.argv) >= 3:
        cmd_pr(int(sys.argv[2]))
    elif cmd == "dim" and len(sys.argv) >= 3:
        cmd_dim(sys.argv[2])
    elif cmd == "report" and len(sys.argv) >= 3:
        cmd_report(sys.argv[2])
    elif cmd == "reports" and len(sys.argv) >= 3:
        cmd_reports(int(sys.argv[2]))
    elif cmd == "list":
        cmd_list()
    elif cmd == "match" and len(sys.argv) >= 3:
        cmd_match(" ".join(sys.argv[2:]))
    else:
        print(__doc__)


if __name__ == "__main__":
    main()

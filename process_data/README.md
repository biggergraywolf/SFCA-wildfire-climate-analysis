# Process data

This directory contains derived analytical tables produced from the wildfire–climate literature corpus. It does not contain the original Web of Science exports, the complete record-level processed corpus, fitted model objects, final manuscript figures, or submission-ready supplementary tables.

## Directory structure

```text
process_data/
├── README.md
└── analysis/
    ├── analysis_summary.json
    ├── annual_relevance_counts.csv
    ├── discipline_concept_index.csv
    ├── discipline_summary.csv
    ├── ignition_endpoint_matrix.csv
    ├── ml_relevance_validation.json
    ├── nmf_k_selection.csv
    ├── nmf_topic_period.csv
    ├── nmf_topic_period_labeled.csv
    ├── nmf_topics_labeled.csv
    ├── nmf_topics_unlabelled.csv
    ├── relevance_summary.csv
    ├── semantic_map_sample.csv
    ├── sfca_concept_jaccard.csv
    ├── sfca_concept_summary.csv
    ├── sfca_evidence_ladder.csv
    └── topic_concept_matrix.csv
```

## Contents

- Corpus and screening summaries: `analysis_summary.json`, `annual_relevance_counts.csv`, `relevance_summary.csv`, and `ml_relevance_validation.json`.
- NMF topic-model outputs: `nmf_k_selection.csv`, `nmf_topics_unlabelled.csv`, `nmf_topics_labeled.csv`, `nmf_topic_period.csv`, and `nmf_topic_period_labeled.csv`.
- SFCA evidence tables: `sfca_concept_summary.csv`, `sfca_concept_jaccard.csv`, `sfca_evidence_ladder.csv`, `ignition_endpoint_matrix.csv`, and `topic_concept_matrix.csv`.
- Disciplinary summaries: `discipline_summary.csv` and `discipline_concept_index.csv`.
- Semantic-map sample: `semantic_map_sample.csv`.

The submission-ready Supplementary Tables S1–S6b are maintained separately under the supplementary-materials directory. Figure-specific source tables that are not direct outputs of the main analysis are maintained with their corresponding plotting scripts.

---

# 过程数据说明

本目录保存野火—气候文献分析产生的派生统计表和中间分析结果，不包括Web of Science原始导出文本、完整记录级处理数据、拟合模型、最终手稿插图及投稿版补充表格。

- 语料库与筛选结果：总体分析汇总、年度相关性分层、相关性类别汇总及机器筛选诊断。
- NMF主题结果：主题数选择、未命名和已命名主题、主题时期组成。
- SFCA分析结果：概念覆盖、概念共现、累积证据链、点火来源—端点矩阵及主题—概念矩阵。
- 学科分析结果：学科分布及学科—概念覆盖。
- 语义空间样本：`semantic_map_sample.csv`。

Supplementary Tables S1–S6b应单独存放在补充材料目录；并非主分析直接生成的图件专用数据，应与相应制图代码放在一起。

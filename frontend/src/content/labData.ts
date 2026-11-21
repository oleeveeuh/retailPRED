/** Typed access to the governed lab data bundles (frontend/public/data/lab).
 *
 * Generated from frozen artifacts and the governed store by
 * scripts/build_lab_data.py and scripts/build_shap_analysis.py; see each
 * bundle's own `provenance` block for lineage. The tests in
 * labContent.test.ts pin the shapes this module relies on.
 */

import policyTauSweepJson from '../../public/data/lab/policy_tau_sweep.json';
import shapAnalysisJson from '../../public/data/lab/shap_analysis.json';
import vintageRevisionsJson from '../../public/data/lab/vintage_revisions.json';
import windowCategoriesJson from '../../public/data/lab/window_categories.json';

export interface TauSweepRow {
  tau: number;
  policy_mean_cat_wape: number;
  delta_vs_naive_pp: number;
  theta_call_rate: number;
}

export interface TauSweepData {
  provenance: { note: string; source_artifacts: string[] };
  frozen_tau: number;
  frozen_model: string;
  guardrail_max_loss_pp: number;
  choose_oof_brier: number;
  n_choose_oof_pairs: number;
  n_report_pairs: number;
  references: { naive: number; theta: number; blend: number };
  sweep: TauSweepRow[];
  report_point: {
    tau: number;
    policy_mean_cat_wape: number;
    naive_mean_cat_wape: number;
    categories_beating_naive: number;
  };
}

export const policyTauSweep = policyTauSweepJson as unknown as TauSweepData;

export interface VintagePoint {
  value: number;
  vintage_date: string;
}

export interface VintageCategory {
  series_id: string;
  latest_revised: number;
  n_vintages_total: number;
  vintages: VintagePoint[];
}

export interface VintageMonth {
  month: string;
  label: string;
  categories: VintageCategory[];
}

export const vintageRevisions = vintageRevisionsJson as unknown as {
  provenance: { availability_rule: string; rule: string; source_artifact: string };
  months: VintageMonth[];
};

export interface WindowCategoryRow {
  window: string;
  model: string;
  series_id: string;
  wape: number;
}

export const windowCategories = windowCategoriesJson as unknown as {
  provenance: { note: string; source_artifacts: string[] };
  windows: string[];
  models: string[];
  mean_category_wape: Record<string, Record<string, number>>;
  rows: WindowCategoryRow[];
};

export interface ShapGlobalRow {
  feature: string;
  mean_abs_shap: number;
  share: number;
}

export interface ShapContribution {
  feature: string;
  shap: number;
  feature_value: number;
}

export interface ShapWaterfall {
  series_id: string;
  month: string;
  base_value: number;
  model_output: number;
  contributions: ShapContribution[];
}

export interface ShapAnalysisData {
  provenance: {
    generated_at: string;
    as_of: string;
    platform_code_revision: string;
    data_sha256: string;
    source_artifacts: string[];
    model: string;
    model_version: string;
    model_status: string;
    data_cutoff: string;
    n_training_rows: number;
    n_series: number;
    explainer: string;
    target: string;
    additivity_max_error: number;
  };
  features: string[];
  global_importance: ShapGlobalRow[];
  waterfalls: ShapWaterfall[];
}

export const shapAnalysis = shapAnalysisJson as unknown as ShapAnalysisData;

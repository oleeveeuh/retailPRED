import type { FC } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  ArrowRight,
  BarChart3,
  CheckCircle2,
  Clock,
  Cpu,
  Database,
  ExternalLink,
  GitBranch,
  Layers,
  ShieldAlert,
  Table2,
  XCircle,
} from 'lucide-react';

import {
  gateStory,
  hero,
  modelCards,
  pipelineTasks,
  rejections,
  tableauViews,
  type ModelCard,
  type PipelineTask,
  type TaskStatus,
} from '../content/showcase';
import { ExecutionTimeline } from '../components/ExecutionTimeline';
import { SkillsProofGrid } from '../components/SkillsProofGrid';
import { SectionEyebrow } from '../components/SectionEyebrow';
import { entrance, skipMountAnimation } from '../lib/entrance';

const statusChip = (status: TaskStatus) => {
  switch (status) {
    case 'executed':
      return {
        classes: 'border-emerald-300 bg-emerald-50 text-emerald-800',
        icon: <CheckCircle2 className="h-3.5 w-3.5" />,
        label: 'Executed',
      };
    case 'refused-by-gate':
      return {
        classes: 'border-amber-300 bg-amber-50 text-amber-800',
        icon: <ShieldAlert className="h-3.5 w-3.5" />,
        label: 'Refused by gate',
      };
    case 'awaiting':
      return {
        classes: 'border-sky-300 bg-sky-50 text-sky-800',
        icon: <Clock className="h-3.5 w-3.5" />,
        label: 'Awaiting input',
      };
    default:
      return {
        classes: 'border-stone-300 bg-stone-50 text-stone-600',
        icon: <Clock className="h-3.5 w-3.5" />,
        label: 'Dormant',
      };
  }
};

export const ShowcasePage: FC = () => (
  <div className="space-y-14">
    {/* Hero */}
    <motion.section
      initial={skipMountAnimation() ? false : { opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      className="relative overflow-hidden border-b border-[#d8d1c4] bg-[#f5f1e8] p-8 sm:p-12"
    >
      <p className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.13em] text-stone-500">
        <span className="inline-block h-[7px] w-[7px] rounded-full bg-[#a43b36]" aria-hidden="true" />
        retailpred · v2.0.0
      </p>
      <h1 className="mt-5 max-w-3xl font-serif text-4xl font-normal lowercase leading-[1.05] tracking-[-0.03em] text-[#25251f] sm:text-5xl">
        retail forecasting, with an{' '}
        <em className="italic text-[#89927a]">evidence gate.</em>
      </h1>
      <p className="mt-5 max-w-2xl text-[15px] leading-[1.8] text-stone-600">
        {hero.subtitle}
      </p>
      <div className="mt-6 flex flex-wrap gap-2">
        {hero.chips.map((chip) => (
          <span
            key={chip.label}
            className="rounded-[20px] bg-[#dfe1d5] px-3 py-1 font-mono text-[10px] text-[#555b4a]"
          >
            <span className="text-stone-500">{chip.label}:</span>{' '}
            <span className={chip.tone === 'amber' ? 'text-[#a43b36]' : 'text-[#25251f]'}>
              {chip.value}
            </span>
          </span>
        ))}
      </div>
      <div className="mt-8 flex flex-wrap items-center gap-6">
        <Link
          to="/dashboard/predictions"
          className="border-b border-[#a43b36] pb-1.5 text-sm text-[#25251f] transition hover:text-[#a43b36]"
        >
          Explore the forecast evidence <span className="text-[#a43b36]" aria-hidden="true">↗</span>
        </Link>
        <Link
          to="/story"
          className="border-b border-[#a43b36] pb-1.5 text-sm text-[#25251f] transition hover:text-[#a43b36]"
        >
          The research story, chapter by chapter{' '}
          <span className="text-[#a43b36]" aria-hidden="true">↗</span>
        </Link>
        <a
          href="https://github.com/oleeveeuh/retailPRED"
          target="_blank"
          rel="noopener noreferrer"
          className="border-b border-[#a43b36] pb-1.5 text-sm text-[#25251f] transition hover:text-[#a43b36]"
        >
          Source and evidence ledger <span className="text-[#a43b36]" aria-hidden="true">↗</span>
        </a>
        <a
          href="https://oleeveeuh.github.io/"
          className="border-b border-[#a43b36] pb-1.5 text-sm text-[#25251f] transition hover:text-[#a43b36]"
        >
          ← back to field notes
        </a>
      </div>
    </motion.section>

    {/* Pipeline */}
    <section className="space-y-6">
      <div className="flex items-center gap-3">
        <div className="flex h-9 w-9 items-center justify-center rounded-full bg-[#dfe1d5]">
          <Database className="h-5 w-5 text-[#89927a]" />
        </div>
        <div>
          <SectionEyebrow>The Databricks pull</SectionEyebrow>
          <h2 className="text-xl text-stone-900">Four governed tasks, one scheduled pipeline</h2>
        </div>
      </div>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-4">
        {pipelineTasks.map((task: PipelineTask, index) => {
          const chip = statusChip(task.status);
          return (
            <motion.div
              key={task.task}
              {...entrance(index * 0.06)}
              className="relative flex flex-col border border-[#d8d1c4] bg-[#fbf9f4] p-5 shadow-[0_4px_13px_rgba(59,49,34,0.08)] transition-transform duration-300 odd:rotate-[-0.8deg] even:rotate-[1.1deg] hover:rotate-0 hover:shadow-[0_10px_22px_rgba(59,49,34,0.14)]"
            >
              <div className="flex items-center justify-between">
                <p className="text-xs font-medium uppercase tracking-wide text-stone-400">
                  {task.task}
                </p>
                <span className={`flex items-center gap-1 border px-2 py-0.5 text-[11px] ${chip.classes}`}>
                  {chip.icon}
                  {chip.label}
                </span>
              </div>
              <h3 className="mt-3 text-sm font-medium text-stone-900">{task.headline}</h3>
              <p className="mt-2 flex-1 text-xs leading-relaxed text-stone-500">
                {task.description}
              </p>
              <p className="mt-3 border-t border-stone-100 pt-3 text-[11px] font-medium text-stone-700">
                {task.evidence}
              </p>
              {index < pipelineTasks.length - 1 && (
                <ArrowRight className="absolute -right-3 top-1/2 hidden h-4 w-4 -translate-y-1/2 text-stone-300 lg:block" />
              )}
            </motion.div>
          );
        })}
      </div>
      <p className="max-w-3xl text-xs leading-relaxed text-stone-500">
        A separate, evaluation-only harness validated the M5 competition track — the official
        12-level item/store hierarchy, WRMSSE machinery, and rolling stability windows. It has no
        live feed by design: that dataset is frozen at May 2016, so its story is measurement, not
        ingestion.
      </p>

      {/* Gate story */}
      <div className="border border-amber-300 bg-amber-50/70 p-6">
        <div className="flex items-center gap-3">
          <ShieldAlert className="h-5 w-5 shrink-0 text-amber-700" />
          <h3 className="text-sm font-semibold uppercase tracking-wide text-amber-900">
            {gateStory.title}
          </h3>
        </div>
        <p className="mt-3 max-w-3xl text-sm leading-relaxed text-amber-950">{gateStory.body}</p>
        <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
          {gateStory.metrics.map((metric) => (
            <div key={metric.label} className="border border-amber-200 bg-white p-3">
              <p className="text-[11px] uppercase tracking-wide text-amber-700">{metric.label}</p>
              <p className="mt-1 text-lg text-amber-950">{metric.value}</p>
            </div>
          ))}
        </div>
        <div className="mt-4 flex flex-wrap gap-4">
          {gateStory.links.map((link) => (
            <a
              key={link.label}
              href={link.href}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1 text-xs font-medium text-[#25251f] underline hover:no-underline"
            >
              {link.label} <ExternalLink className="h-3 w-3" />
            </a>
          ))}
        </div>
      </div>
    </section>

    {/* Execution timeline */}
    <ExecutionTimeline />

    {/* Models */}
    <section className="space-y-6">
      <div className="flex items-center gap-3">
        <div className="flex h-9 w-9 items-center justify-center rounded-full bg-[#dfe1d5]">
          <Cpu className="h-5 w-5 text-[#89927a]" />
        </div>
        <div>
          <SectionEyebrow>The models</SectionEyebrow>
          <h2 className="text-xl text-stone-900">Measured, gated, honestly ranked</h2>
        </div>
      </div>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {modelCards.map((card: ModelCard) => {
          const promoted = card.decision === 'promoted-demo';
          return (
            <motion.div
              key={card.dataset}
              {...entrance()}
              className={`flex flex-col border p-6 shadow-[0_4px_13px_rgba(59,49,34,0.08)] transition-transform duration-300 hover:rotate-0 hover:shadow-[0_10px_22px_rgba(59,49,34,0.14)] ${
                promoted
                  ? 'border-[#89927a] bg-[#fbf9f4] odd:rotate-[-0.7deg]'
                  : 'border-[#d8d1c4] bg-[#fbf9f4] even:rotate-[0.9deg]'
              }`}
            >
              <div className="flex items-start justify-between gap-3">
                <div>
                  <p className="text-xs font-medium uppercase tracking-wide text-stone-500">
                    {card.dataset}
                  </p>
                  <h3 className="mt-1 text-lg text-stone-900">{card.model}</h3>
                  <p className="mt-1 text-xs text-stone-500">{card.useCase}</p>
                </div>
                <span
                  className={`flex items-center gap-1 border px-2 py-1 text-[11px] font-medium ${
                    promoted
                      ? 'border-emerald-300 bg-white text-emerald-800'
                      : 'border-amber-300 bg-white text-amber-800'
                  }`}
                >
                  {promoted ? (
                    <CheckCircle2 className="h-3.5 w-3.5" />
                  ) : (
                    <XCircle className="h-3.5 w-3.5" />
                  )}
                  {promoted ? 'Promoted' : 'Not promoted'}
                </span>
              </div>
              <p className="mt-4 text-sm text-stone-600">{card.headline}</p>
              <div className="mt-4 grid grid-cols-2 gap-3">
                <div className="border border-stone-200 bg-white p-3">
                  <p className="text-[11px] uppercase tracking-wide text-stone-400">
                    {card.metricName}
                  </p>
                  <p className="mt-1 text-2xl text-[#25251f]">{card.metricValue}</p>
                </div>
                <div className="border border-stone-200 bg-white p-3">
                  <p className="text-[11px] uppercase tracking-wide text-stone-400">
                    {card.baselineName}
                  </p>
                  <p className="mt-1 text-2xl font-serif text-stone-500">{card.baselineValue}</p>
                </div>
              </div>
              <p className="mt-3 text-xs font-medium text-stone-700">{card.improvement}</p>
              <p
                className={`mt-3 border-t border-stone-200 pt-3 text-xs ${
                  promoted ? 'text-emerald-800' : 'text-amber-800'
                }`}
              >
                {card.decisionLine}
              </p>
              <a
                href={card.evidencePath}
                target="_blank"
                rel="noopener noreferrer"
                className="mt-3 flex items-center gap-1 text-xs text-[#25251f] underline hover:no-underline"
              >
                Full benchmark document <ExternalLink className="h-3 w-3" />
              </a>
            </motion.div>
          );
        })}
      </div>

      {/* Honest rejections */}
      <div className="border border-[#d8d1c4] bg-[#fbf9f4] p-6">
        <div className="flex items-center gap-2">
          <Layers className="h-4 w-4 text-stone-500" />
          <h3 className="text-sm font-medium uppercase tracking-wide text-stone-700">
            Also measured, also rejected — by predeclared gates
          </h3>
        </div>
        <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
          {rejections.map((rejection) => (
            <div key={rejection.name} className="border border-[#d8d1c4] bg-[#f5f1e8] p-4">
              <div className="flex items-center gap-2">
                <XCircle className="h-4 w-4 shrink-0 text-red-400" />
                <p className="text-sm font-medium text-stone-900">{rejection.name}</p>
              </div>
              <p className="mt-1 text-xs text-stone-500">{rejection.oneLine}</p>
              <p className="mt-2 text-xs font-medium text-stone-700">{rejection.outcome}</p>
            </div>
          ))}
        </div>
      </div>
    </section>

    {/* Skills / proof grid */}
    <SkillsProofGrid />

    {/* Tableau */}
    <section className="space-y-6">
      <div className="flex items-center gap-3">
        <div className="flex h-9 w-9 items-center justify-center rounded-full bg-[#dfe1d5]">
          <Table2 className="h-5 w-5 text-[#89927a]" />
        </div>
        <div>
          <SectionEyebrow>The Tableau layer</SectionEyebrow>
          <h2 className="text-xl text-stone-900">Governed extracts behind five views</h2>
        </div>
      </div>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-5">
        {tableauViews.map((view) => (
          <motion.div
            key={view.name}
            {...entrance()}
            className="border border-[#d8d1c4] bg-[#fbf9f4] p-4 shadow-[0_4px_13px_rgba(59,49,34,0.06)] odd:rotate-[-0.6deg] even:rotate-[0.7deg] transition-transform duration-300 hover:rotate-0"
          >
            <BarChart3 className="h-4 w-4 text-[#25251f]" />
            <h3 className="mt-2 text-sm font-medium text-stone-900">{view.name}</h3>
            <p className="mt-1 text-[11px] font-medium uppercase tracking-wide text-[#a43b36]">
              {view.marks}
            </p>
            <p className="mt-2 text-xs leading-relaxed text-stone-500">{view.oneLine}</p>
          </motion.div>
        ))}
      </div>
      <div className="flex flex-wrap items-center gap-x-6 gap-y-2 border border-[#d8d1c4] bg-[#fbf9f4] p-4 text-xs text-stone-600">
        <span className="flex items-center gap-1">
          <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600" />
          Verified in Tableau Public 2026.3 against a tracked aggregate-only Hyper extract
        </span>
        <span className="flex items-center gap-1">
          <GitBranch className="h-3.5 w-3.5 text-stone-400" />
          Schema-checked and fail-closed before publication — see the{' '}
          <a
            href="https://github.com/oleeveeuh/retailPRED/blob/main/bi/README.md"
            target="_blank"
            rel="noopener noreferrer"
            className="text-[#25251f] underline hover:no-underline"
          >
            BI contract
          </a>
        </span>
        <span className="text-stone-400">Public embed pending</span>
      </div>
    </section>

    {/* Evidence footer */}
    <section className="border border-[#d8d1c4] bg-[#fbf9f4] p-6">
      <div className="flex items-center gap-2">
        <ShieldAlert className="h-4 w-4 text-stone-500" />
        <h3 className="text-xs font-semibold uppercase tracking-wide text-stone-700">
          Claims are graded, not implied
        </h3>
      </div>
      <p className="mt-3 text-sm leading-relaxed text-stone-600">
        Every capability in this project sits on an explicit ladder — implemented → locally
        verified → executed on Databricks → scheduled → publicly deployed — and the serving
        contract keeps this site on historical evidence only. No live inference is claimed,
        because no live inference has been authorized.
      </p>
      <div className="mt-4 flex flex-wrap gap-x-6 gap-y-2 text-xs">
        {['DATABRICKS_RUN.md', 'PUBLICATION_AUDIT.md', 'M3_VINTAGE_BACKTEST.md'].map((doc) => (
          <a
            key={doc}
            href={`https://github.com/oleeveeuh/retailPRED/blob/main/docs/v2/${doc}`}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1 text-[#25251f] underline hover:no-underline"
          >
            {doc.replace('.md', '')} <ExternalLink className="h-3 w-3" />
          </a>
        ))}
      </div>
    </section>
  </div>
);

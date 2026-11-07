import { useState } from 'react';
import type { FC } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { ArrowLeft, ExternalLink } from 'lucide-react';

import {
  storyChapters,
  v1Chapters,
  type StoryChapter,
  type StoryFigure,
} from '../content/story';
import { entrance } from '../lib/entrance';
import { SectionEyebrow } from '../components/SectionEyebrow';

type StoryTab = 'v1' | 'v2';

const TABS: Array<{ id: StoryTab; label: string }> = [
  { id: 'v1', label: 'v1 · the pipeline that could not fail' },
  { id: 'v2', label: 'v2 · the rebuild, measured' },
];

const ChapterFigure: FC<{ figure: StoryFigure }> = ({ figure }) => (
  <figure className="mt-6 border border-[#d8d1c4] bg-[#fbf9f4] p-3 shadow-[0_4px_13px_rgba(59,49,34,0.08)]">
    <img src={figure.src} alt={figure.alt} loading="lazy" className="w-full" />
    <figcaption className="mt-2 border-t border-[#d8d1c4] pt-2 text-xs leading-relaxed text-stone-500">
      {figure.caption}
    </figcaption>
  </figure>
);

const ChapterEvidence: FC<{ chapter: StoryChapter }> = ({ chapter }) => (
  <div className="mt-5 flex flex-wrap items-center gap-x-4 border-t border-[#d8d1c4] pt-4">
    <span className="font-mono text-[10px] uppercase tracking-[0.13em] text-stone-400">
      evidence
    </span>
    <a
      href={chapter.evidence.href}
      target="_blank"
      rel="noopener noreferrer"
      className="flex items-center gap-1 text-xs text-[#25251f] underline hover:no-underline"
    >
      {chapter.evidence.label} <ExternalLink className="h-3 w-3" />
    </a>
  </div>
);

export const StoryPage: FC = () => {
  const [tab, setTab] = useState<StoryTab>('v2');
  const chapters = tab === 'v1' ? v1Chapters : storyChapters;

  return (
    <div className="space-y-12">
      {/* Story hero */}
      <section className="border-b border-[#d8d1c4] pb-2">
        <SectionEyebrow>field notes · the research story</SectionEyebrow>
        <h1 className="mt-5 max-w-3xl font-serif text-4xl font-normal lowercase leading-[1.05] tracking-[-0.03em] text-[#25251f] sm:text-5xl">
          a refused model, investigated <em className="italic text-[#89927a]">properly.</em>
        </h1>
        <p className="mt-5 max-w-2xl text-[15px] leading-[1.8] text-stone-600">
          Two tabs. v1 is the pipeline that could not fail — the undergraduate
          build whose numbers meant nothing. v2 is the rebuild: the vintage
          gate refused the champion, the question moved to a research
          satellite, and eight chapters follow, every number from a frozen
          artifact, negative results included on purpose.
        </p>
        <div
          className="mt-6 flex flex-wrap items-center gap-3"
          role="tablist"
          aria-label="story version"
        >
          {TABS.map((entry) => {
            const active = tab === entry.id;
            return (
              <button
                key={entry.id}
                type="button"
                role="tab"
                aria-selected={active}
                onClick={() => setTab(entry.id)}
                className={`rounded-full px-4 py-2 font-mono text-[10px] uppercase tracking-[0.13em] transition ${
                  active
                    ? 'bg-[#dfe1d5] text-[#25251f]'
                    : 'text-stone-500 hover:bg-[#dfe1d5]/50 hover:text-[#25251f]'
                }`}
              >
                {entry.label}
              </button>
            );
          })}
          <Link
            to="/"
            className="ml-2 flex items-center gap-1.5 border-b border-[#a43b36] pb-1.5 text-sm text-[#25251f] transition hover:text-[#a43b36]"
          >
            <ArrowLeft className="h-3.5 w-3.5" aria-hidden="true" /> back to the showcase
          </Link>
        </div>
      </section>

      {/* Chapters */}
      {chapters.map((chapter, index) => (
        <motion.article
          key={`${tab}-${chapter.id}`}
          id={chapter.id}
          {...entrance(Math.min(index * 0.04, 0.16))}
          className="max-w-3xl"
        >
          <p className="font-mono text-[10px] uppercase tracking-[0.13em] text-[#a43b36]">
            {chapter.kicker}
          </p>
          <h2 className="mt-2 font-serif text-2xl lowercase leading-snug text-[#25251f] sm:text-3xl">
            {chapter.title}
          </h2>
          <p className="mt-3 font-serif text-[17px] italic leading-[1.6] text-stone-500">
            {chapter.lede}
          </p>
          {chapter.blocks.map((block, blockIndex) => {
            if (block.kind === 'figure') {
              return <ChapterFigure key={blockIndex} figure={block.figure} />;
            }
            return (
              <p key={blockIndex} className="mt-4 text-[15px] leading-[1.85] text-stone-600">
                {block.text}
              </p>
            );
          })}
          <ChapterEvidence chapter={chapter} />
        </motion.article>
      ))}
    </div>
  );
};

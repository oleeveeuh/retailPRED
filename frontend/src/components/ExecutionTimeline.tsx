import type { FC } from 'react';
import { useEffect, useRef, useState } from 'react';
import { motion, useReducedMotion } from 'framer-motion';
import { ExternalLink, History } from 'lucide-react';

import { effectiveLinks, timeline } from '../content/timeline';
import { entrance } from '../lib/entrance';
import { SectionEyebrow } from './SectionEyebrow';

const RAIL_HEIGHT = 460;
const RAIL_INSET = 16;

export const ExecutionTimeline: FC = () => {
  const sectionRef = useRef<HTMLElement | null>(null);
  const entryRefs = useRef<(HTMLLIElement | null)[]>([]);
  const [activeIndex, setActiveIndex] = useState(0);
  const [progress, setProgress] = useState(0);
  const reducedMotion = useReducedMotion();

  useEffect(() => {
    const section = sectionRef.current;
    if (!section) return;

    // The app scrolls inside Layout's overflow-y main; walk up to find it so
    // tracking works no matter which ancestor actually scrolls.
    let scroller: HTMLElement | null = section.parentElement;
    while (scroller && scroller !== document.body) {
      const overflowY = window.getComputedStyle(scroller).overflowY;
      if (overflowY === 'auto' || overflowY === 'scroll') break;
      scroller = scroller.parentElement;
    }
    const target: HTMLElement | Window = scroller ?? window;

    const measure = () => {
      const bounds = section.getBoundingClientRect();
      const mid = window.innerHeight / 2;
      setProgress(Math.min(1, Math.max(0, (mid - bounds.top) / Math.max(1, bounds.height))));

      let best = 0;
      let bestDistance = Number.POSITIVE_INFINITY;
      entryRefs.current.forEach((el, i) => {
        if (!el) return;
        const rect = el.getBoundingClientRect();
        const distance = Math.abs(rect.top + rect.height / 2 - mid);
        if (distance < bestDistance) {
          bestDistance = distance;
          best = i;
        }
      });
      setActiveIndex(best);
    };

    let frame = 0;
    const onScroll = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(measure);
    };
    const initial = requestAnimationFrame(measure);
    target.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onScroll);
    return () => {
      cancelAnimationFrame(initial);
      cancelAnimationFrame(frame);
      target.removeEventListener('scroll', onScroll);
      window.removeEventListener('resize', onScroll);
    };
  }, []);

  const focusEntry = (index: number) => {
    const el = entryRefs.current[index];
    if (!el) return;
    el.scrollIntoView({
      behavior: reducedMotion ? 'auto' : 'smooth',
      block: 'center',
    });
  };

  return (
    <section ref={sectionRef} className="space-y-6">
      <div className="flex items-center gap-3">
        <div className="flex h-9 w-9 items-center justify-center rounded-full bg-[#dfe1d5]">
          <History className="h-5 w-5 text-[#89927a]" />
        </div>
        <div>
          <SectionEyebrow>The execution record</SectionEyebrow>
          <h2 className="text-xl text-stone-900">How the verdict actually unfolded</h2>
        </div>
      </div>
      <p className="max-w-3xl text-xs leading-relaxed text-stone-500">
        The full research story, gate refusal included — every turn below was
        measured first, and the predeclared gates did their job at each step.
      </p>

      <div className="grid grid-cols-1 gap-10 lg:grid-cols-[minmax(0,1fr)_170px]">
        {/* Thread + cards */}
        <ol className="relative space-y-8 before:absolute before:bottom-3 before:left-[7px] before:top-3 before:w-[2px] before:bg-[#a43b36]/70">
          {timeline.map((entry, index) => {
            const active = index === activeIndex;
            const tilt = index % 2 === 0 ? 'sm:rotate-[-0.6deg]' : 'sm:rotate-[0.8deg]';
            return (
              <li
                key={entry.id}
                ref={(el) => {
                  entryRefs.current[index] = el;
                }}
                className="relative pl-10"
              >
                <span
                  aria-hidden="true"
                  className={`absolute left-0 top-6 h-4 w-4 rounded-full border-2 bg-[#fbf9f4] transition-colors ${
                    active ? 'border-[#a43b36]' : 'border-[#89927a]'
                  }`}
                />
                <motion.article
                  {...entrance(Math.min(index * 0.03, 0.2))}
                  className={`border border-[#d8d1c4] bg-[#fbf9f4] p-5 shadow-[0_4px_13px_rgba(59,49,34,0.08)] transition-transform duration-300 hover:rotate-0 hover:shadow-[0_10px_22px_rgba(59,49,34,0.14)] ${tilt}`}
                >
                  <p className="font-mono text-[10px] uppercase tracking-[0.13em] text-[#a43b36]">
                    {entry.phase}
                  </p>
                  <h3 className="mt-1.5 font-serif text-lg lowercase leading-snug text-[#25251f]">
                    {entry.headline}
                  </h3>
                  <p className="mt-2 text-sm leading-relaxed text-stone-600">{entry.detail}</p>
                  {effectiveLinks(entry).length > 0 && (
                    <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1">
                      {effectiveLinks(entry).map((link) => (
                        <a
                          key={link.href}
                          href={link.href}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="flex items-center gap-1 text-xs text-[#25251f] underline hover:no-underline"
                        >
                          {link.label} <ExternalLink className="h-3 w-3" />
                        </a>
                      ))}
                    </div>
                  )}
                </motion.article>
              </li>
            );
          })}
        </ol>

        {/* Fixed side rail: tracks scroll, click a phase to scroll it into view */}
        <div className="hidden lg:block">
          <nav
            aria-label="Execution timeline phases"
            className="sticky top-8 rounded-sm border border-[#d8d1c4] bg-[#fbf9f4] p-4 shadow-[0_4px_13px_rgba(59,49,34,0.06)]"
          >
            <p className="font-mono text-[10px] uppercase tracking-[0.13em] text-stone-500">
              phases
            </p>
            <div className="relative mt-3" style={{ height: RAIL_HEIGHT }}>
              <span
                aria-hidden="true"
                className="absolute left-[3.5px] w-[2px] bg-[#a43b36]/70"
                style={{ top: RAIL_INSET, height: RAIL_HEIGHT - RAIL_INSET * 2 }}
              />
              <span
                aria-hidden="true"
                className="absolute left-[-2.5px] h-[9px] w-[9px] rotate-45 border border-[#a43b36] bg-[#fbf9f4]"
                style={{
                  top: RAIL_INSET + progress * (RAIL_HEIGHT - RAIL_INSET * 2) - 4,
                }}
              />
              <ol className="flex h-full flex-col justify-between">
                {timeline.map((entry, index) => {
                  const active = index === activeIndex;
                  return (
                    <li key={entry.id} className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={() => focusEntry(index)}
                        aria-current={active ? 'true' : undefined}
                        className="group flex items-center gap-2 text-left"
                      >
                        <span
                          aria-hidden="true"
                          className={`h-[9px] w-[9px] rounded-full border-2 bg-[#fbf9f4] transition-colors ${
                            active ? 'border-[#a43b36]' : 'border-[#89927a] group-hover:border-[#a43b36]'
                          }`}
                        />
                        <span
                          className={`font-mono text-[10px] leading-none transition-colors ${
                            active ? 'text-[#25251f]' : 'text-stone-400 group-hover:text-[#25251f]'
                          }`}
                        >
                          {entry.phase.replace('phase ', '')}
                        </span>
                      </button>
                    </li>
                  );
                })}
              </ol>
            </div>
          </nav>
        </div>
      </div>
    </section>
  );
};

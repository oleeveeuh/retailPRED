import type { FC } from 'react';
import { motion } from 'framer-motion';
import { Wrench } from 'lucide-react';

import { skills, type Skill } from '../content/skills';
import { entrance } from '../lib/entrance';
import { SectionEyebrow } from './SectionEyebrow';

const tile = (skill: Skill, index: number) => {
  const tilt = index % 2 === 0 ? 'sm:rotate-[-0.5deg]' : 'sm:rotate-[0.6deg]';
  return (
    <a
      key={skill.name}
      href={skill.proofHref}
      target="_blank"
      rel="noopener noreferrer"
      aria-label={`${skill.name} — proof: ${skill.proofLabel}`}
      title={`${skill.name} — ${skill.proofLabel}`}
      className={`group flex flex-col items-center gap-3 border border-[#d8d1c4] bg-[#fbf9f4] p-4 text-center shadow-[0_4px_13px_rgba(59,49,34,0.06)] transition-transform duration-300 hover:rotate-0 hover:shadow-[0_10px_22px_rgba(59,49,34,0.14)] ${tilt}`}
    >
      <span
        className="flex h-16 w-16 items-center justify-center bg-[#dfe1d5] text-[#a43b36] transition-colors [&>svg]:h-8 [&>svg]:w-8 group-hover:bg-[#d6d9c6]"
        dangerouslySetInnerHTML={{ __html: skill.icon }}
      />
      <span>
        <span className="block text-sm text-stone-600 transition-colors group-hover:text-[#25251f]">
          {skill.name}
        </span>
        <span className="mt-0.5 block font-mono text-[10px] uppercase tracking-[0.13em] text-stone-400 transition-colors group-hover:text-[#a43b36]">
          {skill.proofLabel}
        </span>
      </span>
    </a>
  );
};

export const SkillsProofGrid: FC = () => (
  <section className="space-y-6">
    <div className="flex items-center gap-3">
      <div className="flex h-9 w-9 items-center justify-center rounded-full bg-[#dfe1d5]">
        <Wrench className="h-5 w-5 text-[#89927a]" />
      </div>
      <div>
        <SectionEyebrow>The toolkit</SectionEyebrow>
        <h2 className="text-xl text-stone-900">Every skill, proven by an artifact</h2>
      </div>
    </div>
    <ul className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
      {skills.map((skill, index) => (
        <motion.li
          key={skill.name}
          {...entrance(Math.min(index * 0.04, 0.24))}
        >
          {tile(skill, index)}
        </motion.li>
      ))}
    </ul>
    <p className="max-w-3xl text-xs leading-relaxed text-stone-500">
      Every tile links into the repository — source files, the governed
      pipeline, and the CI contract are the proof, not claims about them.
    </p>
  </section>
);

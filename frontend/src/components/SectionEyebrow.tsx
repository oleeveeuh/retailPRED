import type { FC } from 'react';

/** Crimson-dot micro-label used to open each showcase section. */
export const SectionEyebrow: FC<{ children: React.ReactNode }> = ({ children }) => (
  <p className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.13em] text-stone-500">
    <span className="inline-block h-[7px] w-[7px] rounded-full bg-[#a43b36]" aria-hidden="true" />
    {children}
  </p>
);

/** Publication-safe header for the governed v2 evidence surface. */

import type { FC } from 'react';
import { ExternalLink, ShieldCheck } from 'lucide-react';

export const Header: FC = () => (
  <header className="border-b border-stone-200 bg-white">
    <div className="flex items-center justify-between gap-4 px-6 py-4">
      <div className="flex min-w-0 items-center gap-3">
        <ShieldCheck className="h-5 w-5 shrink-0 text-emerald-700" />
        <div>
          <p className="font-mono text-xs font-medium lowercase text-[#25251f]">governed historical evidence</p>
          <p className="lowercase text-xs text-stone-500">
            Deterministic fixture · explicit provenance · fail-closed serving
          </p>
        </div>
      </div>
      <a
        href="https://github.com/oleeveeuh/retailPRED/blob/main/docs/v2/M5_SERVING_CONTRACT.md"
        target="_blank"
        rel="noopener noreferrer"
        className="hidden items-center gap-2 lowercase text-xs text-[#25251f] hover:text-[#a43b36] hover:underline sm:flex"
      >
        Serving contract <ExternalLink className="h-4 w-4" />
      </a>
    </div>
  </header>
);

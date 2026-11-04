/** Publication-safe header for the governed v2 evidence surface. */

import type { FC } from 'react';
import { ExternalLink, ShieldCheck } from 'lucide-react';

export const Header: FC = () => (
  <header className="border-b border-gray-200 bg-white">
    <div className="flex items-center justify-between gap-4 px-6 py-4">
      <div className="flex min-w-0 items-center gap-3">
        <ShieldCheck className="h-5 w-5 shrink-0 text-emerald-700" />
        <div>
          <p className="text-sm font-medium text-gray-900">Governed historical evidence</p>
          <p className="text-xs text-gray-500">
            Deterministic fixture · explicit provenance · fail-closed serving
          </p>
        </div>
      </div>
      <a
        href="https://github.com/oleeveeuh/retailPRED/blob/main/docs/v2/M5_SERVING_CONTRACT.md"
        target="_blank"
        rel="noopener noreferrer"
        className="hidden items-center gap-2 text-sm text-[#3A3A6C] hover:underline sm:flex"
      >
        Serving contract <ExternalLink className="h-4 w-4" />
      </a>
    </div>
  </header>
);

/** Gate Referee — re-runs the predeclared criteria over frozen metrics. */

import { FC, useState } from 'react';
import { CheckCircle2, ExternalLink, XCircle } from 'lucide-react';

import { gateCandidates } from '../../content/lab';
import { evaluateCandidate, verdictLabel } from '../../lib/labGate';

const INK = '#25251f';
const SAGE = '#89927a';
const CRIMSON = '#a43b36';

export const GateReferee: FC = () => {
  const [selectedId, setSelectedId] = useState(gateCandidates[0].id);
  const candidate = gateCandidates.find((c) => c.id === selectedId) ?? gateCandidates[0];
  const result = evaluateCandidate(candidate);

  return (
    <div>
      <div className="mb-4 flex flex-wrap gap-2">
        {gateCandidates.map((c) => (
          <button
            key={c.id}
            onClick={() => setSelectedId(c.id)}
            className={`rounded-sm border px-3 py-1.5 font-mono text-xs lowercase transition-colors ${
              c.id === selectedId
                ? 'border-[#89927a] bg-[#dfe1d5] text-[#25251f]'
                : 'border-stone-200 bg-white text-stone-600 hover:border-[#89927a]'
            }`}
          >
            {c.model}
          </button>
        ))}
      </div>

      <p className="font-mono text-xs lowercase text-stone-500">
        {candidate.dataset} · {candidate.window}
      </p>

      <div className="mt-3 space-y-2">
        {result.results.map(({ criterion, passed }) => (
          <div
            key={criterion.name}
            className={`rounded-sm border px-4 py-3 ${
              passed ? 'border-[#c9c2b2] bg-[#fbf9f4]' : 'border-[#a43b36]/40 bg-[#a43b36]/5'
            }`}
          >
            <div className="flex items-start gap-3">
              {passed ? (
                <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-[#89927a]" />
              ) : (
                <XCircle className="mt-0.5 h-4 w-4 shrink-0 text-[#a43b36]" />
              )}
              <div>
                <p className="font-mono text-xs font-medium lowercase" style={{ color: INK }}>
                  {criterion.name}
                  <span className="ml-2 font-sans" style={{ color: passed ? SAGE : CRIMSON }}>
                    {passed ? 'pass' : 'fail'}
                  </span>
                </p>
                <p className="mt-1 text-sm font-light text-stone-600">{criterion.detail}</p>
                {criterion.kind === 'beats_references' && (
                  <ul className="mt-2 space-y-0.5">
                    {criterion.comparisons.map((c) => (
                      <li key={c.reference} className="font-mono text-xs text-stone-500">
                        {c.reference}: {(100 * c.referenceWape).toFixed(2)}% vs policy{' '}
                        {(100 * c.candidateWape).toFixed(2)}% →{' '}
                        <span style={{ color: c.candidateWape < c.referenceWape ? SAGE : CRIMSON }}>
                          {c.candidateWape < c.referenceWape ? 'beaten' : 'not beaten'}
                        </span>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-4 flex flex-wrap items-center justify-between gap-3 rounded-sm border border-stone-200 bg-white px-4 py-3">
        <div>
          <p className="font-mono text-xs lowercase text-stone-500">documented outcome</p>
          <p className="text-sm font-light text-stone-600">{candidate.documentedSummary}</p>
        </div>
        <span
          className="rounded-sm border-2 px-3 py-1 font-mono text-sm font-semibold lowercase"
          style={{ borderColor: CRIMSON, color: CRIMSON, transform: 'rotate(-2deg)' }}
        >
          {verdictLabel(candidate)}
        </span>
      </div>

      <a
        href={candidate.source.href}
        target="_blank"
        rel="noopener noreferrer"
        className="mt-3 inline-flex items-center gap-1 font-mono text-xs lowercase text-stone-500 hover:text-[#a43b36]"
      >
        evidence: {candidate.source.label} <ExternalLink className="h-3 w-3" />
      </a>
    </div>
  );
};

import type { HTMLMotionProps } from 'framer-motion';

/**
 * True when the page should render fully settled, with no entrance
 * animation. Headless screenshot tooling passes ?static so captures are
 * deterministic (framer still animates opacity under reduced motion, so
 * forced-preferences flags alone cannot produce a settled capture).
 */
const prefersStaticRender = (): boolean =>
  typeof window !== 'undefined' &&
  new URLSearchParams(window.location.search).has('static');

/** Shared whileInView entrance for showcase cards; empty when rendering static. */
export const entrance = (delay = 0): HTMLMotionProps<'div'> =>
  prefersStaticRender()
    ? {}
    : {
        initial: { opacity: 0, y: 10 },
        whileInView: { opacity: 1, y: 0 },
        viewport: { once: true },
        transition: { delay },
      };

/** True when mount animations should be skipped (renders straight at target). */
export const skipMountAnimation = (): boolean => prefersStaticRender();

import type { ReactNode } from "react";

/**
 * A short video, tap to play: native controls, inline on iPhone, muted, nothing loaded until played (the poster
 * shows first), never autoplayed, so a viewer's pause is respected. The server answers Range requests (206),
 * which Safari needs to play and seek.
 */
export function Clip({ src, poster, label, caption, source }: {
  src: string; poster: string; label: string; caption: ReactNode; source: string;
}) {
  return (
    <figure className="how-figure">
      <div className="how-media">
        <video src={src} poster={poster} controls playsInline muted preload="none" aria-label={label} />
      </div>
      <figcaption>
        <div className="how-caption">{caption}</div>
        <div className="how-source">Source: {source}</div>
      </figcaption>
    </figure>
  );
}

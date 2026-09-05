import React, { useMemo } from 'react';
import type { Manifest } from './types';
import { DoorCard } from './Catalogue';
import { useAppearance, type AppearanceRender } from './Appearance';
import { Icon, PageIntro } from './SiteUI';
import { isPetDoor } from './collections';
import './PetCollection.css';

export function PetCollection({manifest}:{manifest:Manifest}) {
  const appearance = useAppearance();
  const photos = useMemo(() => {
    const result = new Map<string, AppearanceRender>();
    for (const r of appearance?.renders ?? []) if (r.image && (!result.has(r.door_id) || r.quality === 'photo')) result.set(r.door_id, r);
    return result;
  }, [appearance]);
  const doors = manifest.doors.filter(d => isPetDoor(d) && !d.error);
  return <section className="page-shell pet-collection">
    <a className="pet-back" href="#/">← Standard door catalogue</a>
    <PageIntro eyebrow="Supplementary assets" title="Pet doors, in their own collection." aside={<span className="pet-count">{doors.length} downloadable doors</span>}>
      <p>Small swinging flaps for cats and dogs, with varied dimensions, materials and hardware. Inspect each asset in 3D and download its simulation files.</p>
    </PageIntro>
    <div className="pet-scope"><Icon name="door"/><p><strong>Downloadable, outside the benchmark.</strong> Standalone pet doors have no robot or human evaluation, baseline results, or reference-motion playback. Ordinary doors with a built-in pet flap remain in the standard collection.</p></div>
    <div className="grid">{doors.map(d => <DoorCard key={d.id} d={d} appearance={photos.get(d.id)}/>)}</div>
    <aside className="catalogue-note"><p>Choose a door for MJCF, URDF, USD and source JSON downloads. These supplementary assets retain their documented simulation approximations.</p><a href="#/">Browse standard doors <Icon name="arrow" size={16}/></a></aside>
  </section>;
}

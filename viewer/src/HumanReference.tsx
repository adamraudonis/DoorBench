import React, {useEffect, useRef, useState} from 'react';
import * as THREE from 'three';
import {GLTFLoader} from 'three/examples/jsm/loaders/GLTFLoader.js';
import {OrbitControls} from 'three/examples/jsm/controls/OrbitControls.js';
import {DOCS, Icon} from './SiteUI';
import {HUMAN_REFERENCE_BASE, sourceTime, validateAnimationClock, validateHumanReference, verifyHumanGLB, verifyHumanArtifact, verifyHumanAdjustment, humanPreviewLabel, type HumanReference as Metadata} from './humanCapture';
import './HumanReference.css';

function disposeObject(root: THREE.Object3D) {
  const textures = new Set<THREE.Texture>(), materials = new Set<THREE.Material>();
  root.traverse(obj => {
    if (obj instanceof THREE.Mesh) {
      obj.geometry.dispose();
      for (const mat of Array.isArray(obj.material) ? obj.material : [obj.material]) materials.add(mat);
      if (obj instanceof THREE.SkinnedMesh) obj.skeleton.dispose();
    }
  });
  for (const mat of materials) {
    for (const value of Object.values(mat)) if (value instanceof THREE.Texture) textures.add(value);
    mat.dispose();
  }
  for (const tex of textures) {tex.dispose(); const data = tex.source.data as {close?: () => void} | undefined; data?.close?.();}
}

type Player = {seek: (time: number) => void; fit: () => void; follow: () => void};
export function HumanReference() {
  const host = useRef<HTMLDivElement>(null), player = useRef<Player | null>(null);
  const timeRef = useRef(0), playingRef = useRef(false);
  const [mode, setMode] = useState<'render' | '3d'>('render');
  const [videoSrc, setVideoSrc] = useState(''), [following, setFollowing] = useState(true);
  const followRef = useRef(true);
  const [metadata, setMetadata] = useState<Metadata | null>(null);
  const [error, setError] = useState(''), [status, setStatus] = useState('Reading capture metadata…');
  const [time, setTime] = useState(0), [playing, setPlaying] = useState(false), [attempt, setAttempt] = useState(0);
  const local = import.meta.env.DEV;
  const play = (value: boolean) => {playingRef.current = value; setPlaying(value); if (!value) setTime(timeRef.current);};
  const seek = (value: number) => {play(false); player.current?.seek(value);};
  useEffect(() => {
    if (!local || !host.current) return;
    const controller = new AbortController();
    let root: THREE.Object3D | undefined, renderer: THREE.WebGLRenderer | undefined, videoURL = '';
    let mixer: THREE.AnimationMixer | undefined, controls: OrbitControls | undefined, grid: THREE.GridHelper | undefined;
    let observer: ResizeObserver | undefined, frame = 0;
    const container = host.current;
    setError(''); setMetadata(null); setVideoSrc(''); setStatus('Reading capture metadata…');
    timeRef.current = 0; playingRef.current = false; setPlaying(false); setTime(0);
    async function load() {
      const response = await fetch(`${HUMAN_REFERENCE_BASE}motion.json`, {signal: controller.signal, cache: 'no-store'});
      if (!response.ok) throw new Error(`Local capture metadata unavailable (${response.status}). Export motion.json and animation.glb, then try again.`);
      const meta = validateHumanReference(await response.json());
      if (meta.status === 'target_leg_contact_adjustment_candidate') {
        const report = await fetch(`${HUMAN_REFERENCE_BASE}contact-fit.json`, {signal: controller.signal, cache: 'no-store'});
        if (!report.ok) throw new Error(`Fitted-leg adjustment report unavailable (${report.status}).`);
        await verifyHumanAdjustment(await report.arrayBuffer(), meta);
        controller.signal.throwIfAborted();
      }
      if (mode === 'render') {
        setStatus('Loading the verified Blender render…');
        const response = await fetch(`${HUMAN_REFERENCE_BASE}normal-speed.mp4`, {signal: controller.signal, cache: 'no-store'});
        if (!response.ok) throw new Error(`Local Blender video unavailable (${response.status}).`);
        const bytes = await response.arrayBuffer();
        await verifyHumanArtifact(bytes, meta, 'normal-speed.mp4');
        controller.signal.throwIfAborted();
        videoURL = URL.createObjectURL(new Blob([bytes], {type: 'video/mp4'}));
        setVideoSrc(videoURL); setMetadata(meta); setStatus(''); return;
      }
      setStatus('Loading and verifying the captured human…');
      const glb = await fetch(`${HUMAN_REFERENCE_BASE}animation.glb`, {signal: controller.signal, cache: 'no-store'});
      if (!glb.ok) throw new Error(`Local human animation unavailable (${glb.status}).`);
      const bytes = await glb.arrayBuffer();
      await verifyHumanGLB(bytes, meta);
      controller.signal.throwIfAborted();
      const asset = await new GLTFLoader().parseAsync(bytes, '');
      if (controller.signal.aborted) {disposeObject(asset.scene); return;}
      root = asset.scene;
      if (asset.animations.length !== 1 || asset.animations[0].name !== meta.action) throw new Error('Expected exactly the metadata-bound capture action.');
      validateAnimationClock(asset.animations[0].duration, meta);
      mixer = new THREE.AnimationMixer(root);
      const action = mixer.clipAction(asset.animations[0]);
      action.setLoop(THREE.LoopOnce, 1); action.clampWhenFinished = true; action.play();
      const bones: THREE.Bone[] = [];
      root.traverse(obj => {if (obj instanceof THREE.Bone) bones.push(obj); if (obj instanceof THREE.Mesh) {obj.castShadow = true; obj.receiveShadow = true; obj.frustumCulled = false;}});
      if (!bones.length) throw new Error('The capture GLB has no articulated human skeleton.');
      const setPose = (value: number) => {
        // A completed LoopOnce action is paused by Three; reset before seeking back.
        action.paused = false; action.enabled = true;
        mixer!.setTime(value); root!.updateMatrixWorld(true);
      };
      const bounds = new THREE.Box3(), point = new THREE.Vector3();
      // Frame the whole path with skeletal positions; this never changes the motion.
      for (let i = 0; i <= 40; i++) {setPose(meta.duration_s * i / 40); for (const bone of bones) bounds.expandByPoint(bone.getWorldPosition(point));}
      bounds.expandByScalar(.2); setPose(0);
      const center = bounds.getCenter(new THREE.Vector3()), size = bounds.getSize(new THREE.Vector3());
      const scene = new THREE.Scene(); scene.background = new THREE.Color('#e9ede7'); scene.add(root);
      scene.add(new THREE.HemisphereLight(0xffffff, 0x929c8c, 2.4));
      const key = new THREE.DirectionalLight(0xfff3e1, 3.5); key.position.copy(center).add(new THREE.Vector3(-3, 5, -4)); scene.add(key);
      const fill = new THREE.DirectionalLight(0xdce9ff, 1.7); fill.position.copy(center).add(new THREE.Vector3(4, 3, 2)); scene.add(fill);
      grid = new THREE.GridHelper(14, 28, 0xb8c3b4, 0xd3dcd0); grid.position.set(center.x, 0, center.z); scene.add(grid);
      const camera = new THREE.PerspectiveCamera(38, 1, .02, 100);
      renderer = new THREE.WebGLRenderer({antialias: true, alpha: false});
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.outputColorSpace = THREE.SRGBColorSpace; renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.1;
      renderer.domElement.setAttribute('aria-label', 'Captured human animation. Drag to orbit, scroll to zoom.');
      container.appendChild(renderer.domElement);
      controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = true;
      controls.minDistance = .7; controls.maxDistance = 25; controls.maxPolarAngle = Math.PI * .94;
      const fit = () => {
        followRef.current = false; setFollowing(false);
        const span = Math.max(size.y, Math.hypot(size.x, size.z) / Math.max(.5, camera.aspect));
        const distance = span / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2))) * 1.23;
        controls!.target.copy(center);
        camera.position.copy(center).add(new THREE.Vector3(1, .55, -1.3).normalize().multiplyScalar(distance));
        controls!.update();
      };
      const anchor = bones.find(b => b.name === 'root') ?? bones[0];
      const previousAnchor = new THREE.Vector3();
      const follow = () => {
        followRef.current = true; setFollowing(true);
        const poseBounds = new THREE.Box3();
        for (const bone of bones) poseBounds.expandByPoint(bone.getWorldPosition(point));
        const focus = poseBounds.getCenter(new THREE.Vector3());
        const distance = Math.max(2.1, 2 / Math.max(.7, camera.aspect)) / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2))) * 1.2;
        controls!.target.copy(focus); camera.position.copy(focus).add(new THREE.Vector3(1, .35, -1.3).normalize().multiplyScalar(distance));
        anchor.getWorldPosition(previousAnchor); controls!.update();
      };
      const resize = () => {const w = container.clientWidth, h = container.clientHeight; camera.aspect = w / Math.max(1, h); camera.updateProjectionMatrix(); renderer!.setSize(w, h);};
      resize(); follow(); observer = new ResizeObserver(resize); observer.observe(container);
      player.current = {fit, follow, seek(value) {timeRef.current = Math.min(meta.duration_s, Math.max(0, value)); setPose(timeRef.current); setTime(timeRef.current);}};
      setMetadata(meta); setStatus('');
      let previous = performance.now(), displayed = previous;
      const draw = (now: number) => {
        // The wall clock advances at 1×. Pause when the tab is hidden instead of jumping on return.
        const elapsed = (now - previous) / 1000; previous = now;
        if (playingRef.current) {
          timeRef.current = Math.min(meta.duration_s, timeRef.current + elapsed);
          setPose(timeRef.current);
          if (timeRef.current >= meta.duration_s) {playingRef.current = false; setPlaying(false); setTime(meta.duration_s);}
        }
        if (now - displayed >= 60) {setTime(timeRef.current); displayed = now;}
        if (followRef.current) {
          anchor.getWorldPosition(point); const delta = point.clone().sub(previousAnchor);
          controls!.target.add(delta); camera.position.add(delta); previousAnchor.copy(point);
        }
        controls!.update(); renderer!.render(scene, camera); frame = requestAnimationFrame(draw);
      };
      frame = requestAnimationFrame(draw);
    }
    const visibility = () => {if (document.hidden) {playingRef.current = false; setPlaying(false);}};
    document.addEventListener('visibilitychange', visibility);
    load().catch(e => {if (!controller.signal.aborted) {setStatus(''); setError(e instanceof Error ? e.message : String(e)); if (root) {disposeObject(root); root = undefined;}}});
    return () => {
      controller.abort(); if (videoURL) URL.revokeObjectURL(videoURL); cancelAnimationFrame(frame); observer?.disconnect(); controls?.dispose();
      document.removeEventListener('visibilitychange', visibility); player.current = null;
      mixer?.stopAllAction(); if (root) {mixer?.uncacheRoot(root); disposeObject(root);}
      grid?.geometry.dispose(); if (grid) for (const mat of Array.isArray(grid.material) ? grid.material : [grid.material]) mat.dispose();
      renderer?.dispose(); renderer?.domElement.remove();
    };
  }, [attempt, local, mode]);
  const fitted = metadata?.status === 'target_leg_contact_adjustment_candidate';
  const correctionCm = metadata?.adjustment_report ? (100 * Math.max(metadata.adjustment_report.L.maximum_ankle_correction_m, metadata.adjustment_report.R.maximum_ankle_correction_m)).toFixed(2) : '';
  const restart = () => {player.current?.seek(0); play(true);};
  return <div className="human-reference page-shell">
    <header className="human-reference-intro"><div><div className="eyebrow">HUMAN MOTION / LOCAL PREVIEW</div><h1>A human, in motion.</h1><p>{humanPreviewLabel(metadata)}</p></div><span className="human-reference-stage">Door not fitted</span></header>
    {!local ? <section className="human-reference-unavailable"><h2>This preview runs locally.</h2><p>The captured human is still being inspected. Its animation is not part of this public deployment.</p><p>In a DoorBench checkout, export the human capture and open this route with the Vite development server.</p><a href={`${DOCS}/HUMAN_REFERENCE.md`}>Read the methodology <Icon name="external" size={14}/></a></section> : <>
      <div className="human-reference-tabs" role="tablist" aria-label="Preview format"><button role="tab" aria-selected={mode === 'render'} onClick={() => setMode('render')}>Blender render</button><button role="tab" aria-selected={mode === '3d'} onClick={() => setMode('3d')}>3D inspection</button><span>{mode === 'render' ? 'Original timing · rendered at 30 fps' : 'Interactive glTF export · original capture clock'}</span></div>
      <section className="human-reference-player" aria-label="Human capture player">
        <div className={`human-reference-viewport ${mode === 'render' ? 'human-reference-video' : ''}`} ref={host}>
          {mode === 'render' && videoSrc && <video src={videoSrc} controls playsInline preload="auto" aria-label="Blender render of captured human motion"/>}
          {(status || error) && <div className="human-reference-overlay" role={error ? 'alert' : 'status'}>{error ? <><h2>Preview unavailable</h2><p>{error}</p><button onClick={() => setAttempt(v => v + 1)}>Try again</button></> : <><span className="loading-dot"/><p>{status}</p></>}</div>}
          {metadata && mode === '3d' && <><div className="human-reference-badge">CeTI / d02 / o03 / run 01</div><div className="human-reference-orbit">Drag to orbit <span>·</span> Scroll to zoom</div><div className="human-reference-camera"><button aria-pressed={following} onClick={() => player.current?.follow()}>Follow human</button><button aria-pressed={!following} onClick={() => player.current?.fit()}>Fit whole path</button></div></>}
        </div>
        {mode === '3d' ? <div className="human-reference-controls">
          <div className="human-reference-control-row"><div className="human-reference-transport"><button className="human-reference-play" disabled={!metadata} onClick={() => playing ? play(false) : timeRef.current >= metadata!.duration_s ? restart() : play(true)}>{playing ? 'Pause' : 'Play'}</button><button disabled={!metadata} onClick={restart}>Restart</button><span className="human-reference-speed">1× <span>Original speed</span></span></div><output aria-live="off" data-testid="human-source-time">Source {metadata ? sourceTime(time, metadata).toFixed(2) : '0.00'} s <span>/ {metadata ? sourceTime(metadata.duration_s, metadata).toFixed(2) : '—'} s</span></output></div>
          <input aria-label="Source time" type="range" min={metadata?.source_clock_offset_s ?? 0} max={metadata ? sourceTime(metadata.duration_s, metadata) : 1} step={metadata?.source_frame_time_s ?? .01} value={metadata ? sourceTime(time, metadata) : 0} disabled={!metadata} onChange={e => seek(Number(e.target.value) - metadata!.source_clock_offset_s)}/>
          <div className="human-reference-timeline-label"><span>{metadata?.duration_s.toFixed(2) ?? '—'} s retained motion</span><span>Two leading calibration frames removed</span></div>
          <p className="human-reference-export-note">Interactive glTF materials and skinning can differ from the Blender render.</p>
        </div> : <div className="human-reference-video-note"><strong>Blender render · 1× motion</strong><p>Video controls show decoder time. This 30 fps render samples the retained 7.61 s clock; the final displayed sample is at 7.60 s. Skin and clothes use the Blender materials.</p></div>}
      </section>
      <section className="human-reference-context">
        <div><div className="eyebrow">WHAT YOU ARE SEEING</div><h2>{fitted ? 'Captured timing. Fitted legs.' : 'Human timing. A new character.'}</h2><p>Real full-body IMU motion from the CeTI door-opening task, transferred to an articulated, dressed MakeHuman character. The capture clock is preserved; the character has different body proportions.</p>{fitted && <p className="human-reference-adaptation">The target legs use authored fitting adjustments: ankles move by up to {correctionCm} cm and thigh/shin rotations change. The original clock, pelvis and upper-body transfer stay unchanged. Support phases are inferred, not measured contacts.</p>}<p>The door and room have not been registered to this capture. The grid is a visual reference. A complete human–door interaction, forces, balance and physical feasibility have not been verified.</p></div>
        <div className="human-reference-provenance"><div className="eyebrow">SOURCE & METHOD</div><dl><div><dt>Motion</dt><dd><a href="https://doi.org/10.6084/m9.figshare.26983645.v2" target="_blank" rel="noreferrer">CeTI-Age-Kinematics v2 ↗</a><span>Pogrzeba et al. · <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank" rel="noreferrer">CC BY 4.0</a></span></dd></div><div><dt>Character</dt><dd>MakeHuman / MPFB<span>CC0 assets · authored skin and clothes</span></dd></div><div><dt>Capture</dt><dd>{metadata ? Math.round(1 / metadata.source_frame_time_s) : 100} Hz · right-hand door task<span>Source frames {metadata?.source_frame_start ?? 2}–{metadata ? metadata.source_frame_start + metadata.retained_frames - 1 : 763} · original rhythm</span></dd></div>{fitted && <div><dt>Adaptation</dt><dd>Target legs only<span>Authored adjustments · door interaction not fitted</span></dd></div>}</dl><a className="human-reference-method" href={`${HUMAN_REFERENCE_BASE}methodology.md`} target="_blank" rel="noreferrer">Methodology & limitations <Icon name="external" size={14}/></a>{metadata && <details><summary>Verified local artifact</summary><p>SHA-256 verified before playback. This binds the preview file, not interaction correctness.</p><code>{metadata.artifacts.find(a => a.path === (mode === 'render' ? 'normal-speed.mp4' : 'animation.glb'))!.sha256}</code><a href={`${HUMAN_REFERENCE_BASE}motion.json`} target="_blank" rel="noreferrer">View transfer metadata</a>{fitted && <a className="human-reference-adjustment-link" href={`${HUMAN_REFERENCE_BASE}contact-fit.json`} target="_blank" rel="noreferrer">View bound leg-adjustment report</a>}</details>}</div>
      </section>
    </>}
  </div>;
}

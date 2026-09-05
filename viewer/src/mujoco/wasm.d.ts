// Vite `?url` imports of the MuJoCo WebAssembly binary (bundled as a hashed static asset, fetched only on the
// playground route).  The npm package ships `mujoco.wasm` next to `mujoco.js`; we hand its URL to the emscripten
// module through `locateFile` so the same code works in dev, in the built Pages site and under bun.
declare module "@mujoco/mujoco/mujoco.wasm?url" {
  const url: string;
  export default url;
}

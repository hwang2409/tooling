export type Chimy2Exports = {
  memory: WebAssembly.Memory;
  init: (width: number, height: number) => number;
  render_frame: (time: number, yaw: number, pitch: number, mode: number) => number;
  framebuffer_ptr: () => number;
  framebuffer_len: () => number;
  framebuffer_width: () => number;
  framebuffer_height: () => number;
};

export type Chimy2Instance = {
  exports: Chimy2Exports;
};

export async function loadWasm(url: string): Promise<Chimy2Instance> {
  if (WebAssembly.instantiateStreaming) {
    try {
      const response = await fetch(url);
      const result = await WebAssembly.instantiateStreaming(response, {});
      return { exports: result.instance.exports as unknown as Chimy2Exports };
    } catch (error) {
      // Streaming instantiation can fail if the server sends the wrong MIME
      // type; fall through to the array-buffer path so local previews work.
      console.info("streaming wasm load failed; using array-buffer fallback", error);
    }
  }
  const response = await fetch(url);
  const bytes = await response.arrayBuffer();
  const result = await WebAssembly.instantiate(bytes, {});
  return { exports: result.instance.exports as unknown as Chimy2Exports };
}

# chimy2 browser demo

This is a plain ES module demo. It uses the same CPU renderer as the native
programs. The wasm boundary exports a linear-memory RGBA8 framebuffer.

## build

From the repository root:

```sh
./chimy2/web/build.sh
```

The script checks for `wasm32-unknown-unknown`, builds the library, and copies
the output to `chimy2/web/chimy2.wasm`.

## serve

The browser blocks local wasm fetches. Serve the directory over HTTP:

```sh
cd chimy2/web
python3 -m http.server 8000
```

Open <http://localhost:8000/> in a browser with WebAssembly support. Drag the
canvas to orbit. Use the shader menu to switch modes. The animated arm samples
the embedded glTF animation from the current frame time.

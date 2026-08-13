// Tiny JSON tokenizer for HUD-style syntax colouring. Keeps the emitted markup
// safe by escaping first, then wrapping literal string / number / key regions.
export function highlightJson(source: string): string {
  const escape = (s: string) =>
    s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  let out = escape(source);
  out = out.replace(
    /(&quot;[^&]*?&quot;)(\s*:)/g,
    (_, key: string, tail: string) => `<span class="k">${key}</span>${tail}`,
  );
  out = out.replace(
    /:\s*(&quot;[^&]*?&quot;)/g,
    (_, str: string) => `: <span class="s">${str}</span>`,
  );
  out = out.replace(
    /(?<![\w"])(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)/gi,
    '<span class="n">$1</span>',
  );
  return out;
}

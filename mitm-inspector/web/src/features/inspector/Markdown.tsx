import ReactMarkdown, { defaultUrlTransform } from "react-markdown";
import type { Components, UrlTransform } from "react-markdown";
import remarkGfm from "remark-gfm";

/**
 * Markdown rendering for message prose. Safety model: react-markdown emits
 * React elements only — no innerHTML anywhere. Captured HTML/XML-ish tag
 * content is escaped BEFORE the markdown parser sees it: tag-shaped
 * `<foo>`/`</foo>`/`<foo attr>` sequences outside fenced code blocks and
 * inline code spans get their angle brackets rewritten to `&lt;`/`&gt;`
 * character references. This preserves the visible tag text and — crucially
 * — lets any markdown syntax INSIDE a multi-line captured tag render
 * normally instead of being swallowed into an opaque HTML block. Fenced
 * code and inline code stay untouched so real code samples still display
 * angle brackets verbatim. Links keep the library's default urlTransform
 * (javascript: etc. stripped). Images are never rendered at all: captured
 * traffic is attacker-controlled, and an <img> would fire an automatic
 * outbound request — an exfiltration channel in a local-only tool. Image
 * markdown renders as inert code instead, and BECAUSE it is text-only, the
 * image src bypasses the default URL transform so the captured URL and title
 * stay visible as evidence (no information loss, still zero network fetch;
 * the CSP in index.html backstops this). GFM (tables, strikethrough, task
 * lists, autolinks) is enabled so captured chat markdown renders faithfully.
 */
const TAG_LIKE = /<(\/?[a-zA-Z][a-zA-Z0-9-]*(?:\s+[^>]*)?\/?)>/g;
const FENCE = /(^|\n)(```|~~~)[^\n]*\n[\s\S]*?\n\2(?=\n|$)/g;
const INLINE_CODE = /(`+)[\s\S]*?\1/g;

function escapeTagsInText(input: string): string {
  return input.replace(TAG_LIKE, (_match, inner: string) => `&lt;${inner}&gt;`);
}

function escapeOutsideInlineCode(input: string): string {
  const parts: string[] = [];
  let cursor = 0;
  INLINE_CODE.lastIndex = 0;
  let match: RegExpExecArray | null;
  while ((match = INLINE_CODE.exec(input)) !== null) {
    parts.push(escapeTagsInText(input.slice(cursor, match.index)));
    parts.push(match[0]);
    cursor = match.index + match[0].length;
  }
  parts.push(escapeTagsInText(input.slice(cursor)));
  return parts.join("");
}

export function escapeCapturedTags(input: string): string {
  const parts: string[] = [];
  let cursor = 0;
  FENCE.lastIndex = 0;
  let match: RegExpExecArray | null;
  while ((match = FENCE.exec(input)) !== null) {
    parts.push(escapeOutsideInlineCode(input.slice(cursor, match.index)));
    parts.push(match[0]);
    cursor = match.index + match[0].length;
  }
  parts.push(escapeOutsideInlineCode(input.slice(cursor)));
  return parts.join("");
}

const remarkPlugins = [remarkGfm];
const urlTransform: UrlTransform = (url, key) => {
  // Images render as inert text, never as an element with a src — keep the
  // captured URL intact for display. Everything else gets the default
  // sanitization.
  if (key === "src") return url;
  return defaultUrlTransform(url);
};

const components: Components = {
  a(props) {
    const { node, children, ...rest } = props;
    void node;
    return (
      <a {...rest} target="_blank" rel="noopener noreferrer">
        {children}
      </a>
    );
  },
  img(props) {
    const { node, src, alt, title } = props;
    void node;
    const parts = [
      typeof alt === "string" && alt.length > 0 ? alt : null,
      typeof src === "string" && src.length > 0 ? src : null,
      typeof title === "string" && title.length > 0 ? `“${title}”` : null,
    ].filter((part): part is string => part !== null);
    return <code className="md-img">image: {parts.join(" — ")}</code>;
  },
  table(props) {
    const { node, children, ...rest } = props;
    void node;
    return (
      <div className="md-table-wrap" tabIndex={0} role="group" aria-label="table">
        <table {...rest}>{children}</table>
      </div>
    );
  },
};

export function MarkdownProse({ text }: { text: string }) {
  const escaped = escapeCapturedTags(text);
  return (
    <div className="md" data-testid="markdown-prose">
      <ReactMarkdown components={components} urlTransform={urlTransform} remarkPlugins={remarkPlugins}>{escaped}</ReactMarkdown>
    </div>
  );
}

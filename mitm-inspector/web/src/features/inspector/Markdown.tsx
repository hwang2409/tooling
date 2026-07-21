import ReactMarkdown, { defaultUrlTransform } from "react-markdown";
import type { Components, UrlTransform } from "react-markdown";

/**
 * Markdown rendering for message prose. Safety model: react-markdown emits
 * React elements only — no innerHTML anywhere. Raw HTML in the source is NOT
 * parsed as HTML (no rehype-raw), and links keep the library's default
 * urlTransform (javascript: etc. stripped). Images are never rendered at
 * all: captured traffic is attacker-controlled, and an <img> would fire an
 * automatic outbound request — an exfiltration channel in a local-only
 * tool. Image markdown renders as inert code instead, and BECAUSE it is
 * text-only, the image src bypasses the default URL transform so the
 * captured URL and title stay visible as evidence (no information loss,
 * still zero network fetch; the CSP in index.html backstops this). The
 * plain-text and raw-JSON views stay one toggle away in the conversation
 * UI, so rendering is presentation only, never a reduction.
 */
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
};

export function MarkdownProse({ text }: { text: string }) {
  return (
    <div className="md" data-testid="markdown-prose">
      <ReactMarkdown components={components} urlTransform={urlTransform}>{text}</ReactMarkdown>
    </div>
  );
}

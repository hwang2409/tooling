import ReactMarkdown from "react-markdown";
import type { Components } from "react-markdown";

/**
 * Markdown rendering for message prose. Safety model: react-markdown emits
 * React elements only — no innerHTML anywhere. Raw HTML in the source is NOT
 * parsed as HTML (no rehype-raw), and the library's default urlTransform
 * strips unsafe protocols such as javascript: from links. Images are never
 * rendered at all: captured traffic is attacker-controlled, and an <img>
 * would fire an automatic outbound request — an exfiltration channel in a
 * local-only tool. Image markdown renders as inert code instead (the CSP in
 * index.html backstops this). The plain-text and raw-JSON views stay one
 * toggle away in the conversation UI, so rendering is presentation only,
 * never a reduction.
 */
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
    const { node, src, alt } = props;
    void node;
    return <code className="md-img">image: {typeof alt === "string" && alt.length > 0 ? `${alt} — ` : ""}{typeof src === "string" ? src : ""}</code>;
  },
};

export function MarkdownProse({ text }: { text: string }) {
  return (
    <div className="md" data-testid="markdown-prose">
      <ReactMarkdown components={components}>{text}</ReactMarkdown>
    </div>
  );
}

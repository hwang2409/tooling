import ReactMarkdown from "react-markdown";
import type { Components } from "react-markdown";

/**
 * Markdown rendering for message prose. Safety model: react-markdown emits
 * React elements only — no innerHTML anywhere. Raw HTML in the source is NOT
 * parsed as HTML (no rehype-raw), and the library's default urlTransform
 * strips unsafe protocols such as javascript: from links and images. The
 * plain-text and raw-JSON views stay one toggle away in the conversation UI,
 * so rendering is presentation only, never a reduction.
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
};

export function MarkdownProse({ text }: { text: string }) {
  return (
    <div className="md" data-testid="markdown-prose">
      <ReactMarkdown components={components}>{text}</ReactMarkdown>
    </div>
  );
}

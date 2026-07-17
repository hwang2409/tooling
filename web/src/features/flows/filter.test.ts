import { describe, expect, it } from "vitest";

import { parseFilter } from "./filter";
import type { FilterableFlow } from "./filter";

function flow(overrides: Partial<FilterableFlow> = {}): FilterableFlow {
  return {
    method: "POST",
    scheme: "https",
    host: "api.example.test",
    port: "443",
    path: "/v1/messages",
    url: "https://api.example.test:443/v1/messages",
    requestHeaders: [{ name: "content-type", value: "application/json" }, { name: "x-request-id", value: "abc-123" }],
    responseHeaders: [{ name: "content-type", value: "text/event-stream" }],
    requestContentType: "application/json",
    responseContentType: "text/event-stream",
    hasResponse: true,
    errored: false,
    ...overrides,
  };
}

function matches(query: string, value: FilterableFlow): boolean {
  const parsed = parseFilter(query);
  if (!parsed.ok) throw new Error(`expected filter to parse: ${parsed.error}`);
  return parsed.matches(value);
}

function parseError(query: string): string {
  const parsed = parseFilter(query);
  if (parsed.ok) throw new Error("expected filter to fail parsing");
  return parsed.error;
}

describe("filter parsing", () => {
  it("treats an empty or blank query as match-all and reports it as empty", () => {
    const parsed = parseFilter("   ");
    expect(parsed.ok && parsed.empty).toBe(true);
    expect(matches("", flow())).toBe(true);
  });

  it("matches bare words as case-insensitive regexes over the URL", () => {
    expect(matches("MESSAGES", flow())).toBe(true);
    expect(matches("v1/mess", flow())).toBe(true);
    expect(matches("unrelated", flow())).toBe(false);
  });

  it("matches quoted patterns against the URL even when they start with a tilde", () => {
    expect(matches('"~odd"', flow({ url: "https://h:1/~odd" }))).toBe(true);
    expect(matches("'messages'", flow())).toBe(true);
  });

  it("supports escaped quotes inside quoted patterns", () => {
    expect(matches(String.raw`"a\"b"`, flow({ url: 'https://h:1/a"b' }))).toBe(true);
  });

  it("rejects unterminated quoted patterns", () => {
    expect(parseError('"unterminated')).toContain("Unterminated");
  });

  it("filters methods with ~m", () => {
    expect(matches("~m post", flow())).toBe(true);
    expect(matches("~m ^get$", flow())).toBe(false);
  });

  it("filters domains with ~d and urls with ~u", () => {
    expect(matches("~d example", flow())).toBe(true);
    expect(matches("~d ^api\\.", flow())).toBe(true);
    expect(matches("~d nowhere", flow())).toBe(false);
    expect(matches("~u /v1/messages", flow())).toBe(true);
  });

  it("matches headers as name-value lines for ~h, ~hq, and ~hs", () => {
    expect(matches("~h x-request-id", flow())).toBe(true);
    expect(matches("~hq 'x-request-id: abc-123'", flow())).toBe(true);
    expect(matches("~hs event-stream", flow())).toBe(true);
    expect(matches("~hq event-stream", flow())).toBe(false);
    expect(matches("~hs x-request-id", flow())).toBe(false);
  });

  it("matches content types on either or specific sides", () => {
    expect(matches("~t json", flow())).toBe(true);
    expect(matches("~t event-stream", flow())).toBe(true);
    expect(matches("~tq json", flow())).toBe(true);
    expect(matches("~tq event-stream", flow())).toBe(false);
    expect(matches("~ts event-stream", flow())).toBe(true);
    expect(matches("~t json", flow({ requestContentType: null, responseContentType: null }))).toBe(false);
  });

  it("evaluates ~q, ~s, and ~e presence filters", () => {
    expect(matches("~q", flow({ hasResponse: false }))).toBe(true);
    expect(matches("~q", flow())).toBe(false);
    expect(matches("~s", flow())).toBe(true);
    expect(matches("~e", flow({ errored: true }))).toBe(true);
    expect(matches("~e", flow())).toBe(false);
  });

  it("accepts ~http and ~all as always-true for captured flows", () => {
    expect(matches("~http", flow())).toBe(true);
    expect(matches("~all", flow())).toBe(true);
  });

  it("combines adjacent terms with implicit AND", () => {
    expect(matches("~m post ~d example", flow())).toBe(true);
    expect(matches("~m post ~d nowhere", flow())).toBe(false);
  });

  it("honours explicit &, |, parentheses, and precedence", () => {
    expect(matches("~m post & ~s", flow())).toBe(true);
    expect(matches("~m get | ~m post", flow())).toBe(true);
    expect(matches("~m get | ~m put", flow())).toBe(false);
    expect(matches("(~m get | ~m post) ~d example", flow())).toBe(true);
    expect(matches("~m get | ~m post ~d nowhere", flow())).toBe(false);
  });

  it("negates terms and groups with !", () => {
    expect(matches("!~e", flow())).toBe(true);
    expect(matches("!~m post", flow())).toBe(false);
    expect(matches("!(~m get | ~m put)", flow())).toBe(true);
    expect(matches("!!~m post", flow())).toBe(true);
  });

  it("rejects invalid regular expressions with the offending operator", () => {
    expect(parseError("~m [")).toContain("~m");
    expect(parseError("(~m post")).toContain("parenthesis");
    expect(parseError(")")).toContain("parenthesis");
  });

  it("requires pattern arguments for regex operators", () => {
    expect(parseError("~m")).toContain("~m needs a pattern");
    expect(parseError("~d ~m post")).toContain("~d needs a pattern");
  });

  it("explains unsupported mitmproxy operators instead of silently ignoring them", () => {
    expect(parseError("~c 200")).toContain("status code");
    expect(parseError("~b secret")).toContain("selecting a flow");
    expect(parseError("~tcp")).toContain("HTTP");
    expect(parseError("~marked")).toContain("marking");
  });

  it("rejects unknown tilde operators", () => {
    expect(parseError("~zz")).toContain("Unknown filter operator");
  });

  it("rejects dangling boolean structure", () => {
    expect(parseError("~m post |")).toContain("Expected a filter expression");
    expect(parseError("!")).toContain("Expected a filter expression");
  });
});

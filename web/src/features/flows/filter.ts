/* eslint-disable no-unused-vars */

export interface FilterHeader {
  readonly name: string;
  readonly value: string;
}

export interface FilterableFlow {
  readonly method: string;
  readonly scheme: string;
  readonly host: string;
  readonly port: string;
  readonly path: string;
  readonly url: string;
  readonly requestHeaders: readonly FilterHeader[];
  readonly responseHeaders: readonly FilterHeader[];
  readonly requestContentType: string | null;
  readonly responseContentType: string | null;
  readonly hasResponse: boolean;
  readonly errored: boolean;
}

export type FilterPredicate = (flow: FilterableFlow) => boolean;

export type ParsedFilter =
  | { readonly ok: true; readonly empty: boolean; readonly matches: FilterPredicate }
  | { readonly ok: false; readonly error: string };

type Token =
  | { readonly kind: "(" | ")" | "&" | "|" | "!" }
  | { readonly kind: "word"; readonly value: string; readonly quoted: boolean };

class FilterError extends Error {}

const STRUCTURAL = new Set(["(", ")", "&", "|", "!"]);

function tokenize(input: string): Token[] {
  const tokens: Token[] = [];
  let index = 0;
  while (index < input.length) {
    const char = input[index];
    if (/\s/.test(char)) {
      index += 1;
      continue;
    }
    if (STRUCTURAL.has(char)) {
      tokens.push({ kind: char as "(" | ")" | "&" | "|" | "!" });
      index += 1;
      continue;
    }
    if (char === '"' || char === "'") {
      const quote = char;
      let value = "";
      index += 1;
      let closed = false;
      while (index < input.length) {
        const current = input[index];
        if (current === "\\" && input[index + 1] === quote) {
          value += quote;
          index += 2;
          continue;
        }
        if (current === quote) {
          closed = true;
          index += 1;
          break;
        }
        value += current;
        index += 1;
      }
      if (!closed) throw new FilterError("Unterminated quoted pattern.");
      tokens.push({ kind: "word", value, quoted: true });
      continue;
    }
    let value = "";
    while (index < input.length && !/\s/.test(input[index]) && !STRUCTURAL.has(input[index]) && input[index] !== '"' && input[index] !== "'") {
      value += input[index];
      index += 1;
    }
    tokens.push({ kind: "word", value, quoted: false });
  }
  return tokens;
}

function compilePattern(pattern: string, operator: string): RegExp {
  try {
    return new RegExp(pattern, "i");
  } catch {
    throw new FilterError(`Invalid regular expression for ${operator}: ${pattern}`);
  }
}

function headerMatch(headers: readonly FilterHeader[], pattern: RegExp): boolean {
  return headers.some((header) => pattern.test(`${header.name}: ${header.value}`));
}

function contentTypeMatch(types: readonly (string | null)[], pattern: RegExp): boolean {
  return types.some((value) => value !== null && pattern.test(value));
}

const UNSUPPORTED: Record<string, string> = {
  "~c": "protocol v1 does not carry a status code",
  "~b": "body content is only decoded after selecting a flow",
  "~bq": "body content is only decoded after selecting a flow",
  "~bs": "body content is only decoded after selecting a flow",
  "~a": "asset detection needs response bodies",
  "~replay": "replay is out of scope for the read-only MVP",
  "~replayq": "replay is out of scope for the read-only MVP",
  "~replays": "replay is out of scope for the read-only MVP",
  "~marked": "flow marking is not part of protocol v1",
  "~marker": "flow marking is not part of protocol v1",
  "~comment": "flow comments are not part of protocol v1",
  "~meta": "flow metadata annotations are not part of protocol v1",
  "~websocket": "only HTTP(S) flows are captured",
  "~tcp": "only HTTP(S) flows are captured",
  "~udp": "only HTTP(S) flows are captured",
  "~dns": "only HTTP(S) flows are captured",
  "~src": "connection addresses are not part of protocol v1",
  "~dst": "connection addresses are not part of protocol v1",
};

const NO_ARGUMENT = new Set(["~q", "~s", "~e", "~http", "~all"]);
const REGEX_ARGUMENT = new Set(["~m", "~d", "~u", "~h", "~hq", "~hs", "~t", "~tq", "~ts"]);

class Parser {
  private position = 0;

  constructor(private readonly tokens: readonly Token[]) {}

  parse(): FilterPredicate {
    const predicate = this.parseOr();
    if (this.position < this.tokens.length) {
      throw new FilterError(this.peek()?.kind === ")" ? "Unbalanced closing parenthesis." : "Unexpected trailing filter input.");
    }
    return predicate;
  }

  private peek(): Token | undefined {
    return this.tokens[this.position];
  }

  private next(): Token | undefined {
    return this.tokens[this.position++];
  }

  private parseOr(): FilterPredicate {
    const branches = [this.parseAnd()];
    while (this.peek()?.kind === "|") {
      this.next();
      branches.push(this.parseAnd());
    }
    if (branches.length === 1) return branches[0];
    return (flow) => branches.some((branch) => branch(flow));
  }

  private parseAnd(): FilterPredicate {
    const branches = [this.parseNot()];
    for (;;) {
      const token = this.peek();
      if (token === undefined || token.kind === ")" || token.kind === "|") break;
      if (token.kind === "&") {
        this.next();
        continue;
      }
      branches.push(this.parseNot());
    }
    if (branches.length === 1) return branches[0];
    return (flow) => branches.every((branch) => branch(flow));
  }

  private parseNot(): FilterPredicate {
    if (this.peek()?.kind === "!") {
      this.next();
      const inner = this.parseNot();
      return (flow) => !inner(flow);
    }
    return this.parseAtom();
  }

  private parseAtom(): FilterPredicate {
    const token = this.next();
    if (token === undefined) throw new FilterError("Expected a filter expression.");
    if (token.kind === "(") {
      const inner = this.parseOr();
      if (this.next()?.kind !== ")") throw new FilterError("Unbalanced opening parenthesis.");
      return inner;
    }
    if (token.kind === ")") throw new FilterError("Unbalanced closing parenthesis.");
    if (token.kind !== "word") throw new FilterError(`Unexpected "${token.kind}" in filter.`);
    if (token.quoted || !token.value.startsWith("~")) {
      const pattern = compilePattern(token.value, "the URL pattern");
      return (flow) => pattern.test(flow.url);
    }
    return this.parseOperator(token.value);
  }

  private parseOperator(operator: string): FilterPredicate {
    if (operator in UNSUPPORTED) throw new FilterError(`${operator} is not available here: ${UNSUPPORTED[operator]}.`);
    if (NO_ARGUMENT.has(operator)) {
      switch (operator) {
        case "~q": return (flow) => !flow.hasResponse;
        case "~s": return (flow) => flow.hasResponse;
        case "~e": return (flow) => flow.errored;
        default: return () => true;
      }
    }
    if (!REGEX_ARGUMENT.has(operator)) throw new FilterError(`Unknown filter operator ${operator}.`);
    const argument = this.next();
    if (argument === undefined || argument.kind !== "word" || (!argument.quoted && argument.value.startsWith("~"))) {
      throw new FilterError(`${operator} needs a pattern argument.`);
    }
    const pattern = compilePattern(argument.value, operator);
    switch (operator) {
      case "~m": return (flow) => pattern.test(flow.method);
      case "~d": return (flow) => pattern.test(flow.host);
      case "~u": return (flow) => pattern.test(flow.url);
      case "~h": return (flow) => headerMatch(flow.requestHeaders, pattern) || headerMatch(flow.responseHeaders, pattern);
      case "~hq": return (flow) => headerMatch(flow.requestHeaders, pattern);
      case "~hs": return (flow) => headerMatch(flow.responseHeaders, pattern);
      case "~t": return (flow) => contentTypeMatch([flow.requestContentType, flow.responseContentType], pattern);
      case "~tq": return (flow) => contentTypeMatch([flow.requestContentType], pattern);
      default: return (flow) => contentTypeMatch([flow.responseContentType], pattern);
    }
  }
}

export function parseFilter(input: string): ParsedFilter {
  const trimmed = input.trim();
  if (trimmed === "") {
    return { ok: true, empty: true, matches: () => true };
  }
  try {
    const matches = new Parser(tokenize(trimmed)).parse();
    return { ok: true, empty: false, matches };
  } catch (error) {
    if (error instanceof FilterError) return { ok: false, error: error.message };
    throw error;
  }
}

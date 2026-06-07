export function compactWhitespace(value: string) {
  return value.replace(/\r/g, "\n").replace(/[ \t]+/g, " ").replace(/\n{3,}/g, "\n\n").trim();
}

export function clipText(value: string, maxLength: number) {
  const normalized = compactWhitespace(value);
  if (normalized.length <= maxLength) {
    return normalized;
  }
  return `${normalized.slice(0, maxLength)}\n\n[内容已截断，以控制单次模型上下文长度]`;
}

export function countCjkLikeChars(value: string) {
  return value.replace(/\s/g, "").length;
}

export function trimModelText(value: string) {
  return compactWhitespace(value)
    .replace(/^["'“”]+|["'“”]+$/g, "")
    .replace(/^#+\s*/gm, "")
    .trim();
}

export function enforceMaxChars(value: string, maxChars: number) {
  const clean = trimModelText(value);
  if (countCjkLikeChars(clean) <= maxChars) {
    return clean;
  }

  let count = 0;
  let output = "";
  for (const char of clean) {
    if (!/\s/.test(char)) {
      count += 1;
    }
    if (count > maxChars) {
      break;
    }
    output += char;
  }
  return output.replace(/[，。；、,.!?！？：:]+$/u, "。");
}

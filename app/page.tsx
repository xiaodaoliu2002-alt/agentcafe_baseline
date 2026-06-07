"use client";

import {
  BookOpenText,
  Download,
  FileText,
  Highlighter,
  Loader2,
  Pause,
  Play,
  RotateCcw,
  Send,
  Square,
  Upload
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { CustomAgentInput, DiscussionMessage, Note, SubmitAnswers } from "@/lib/types";

type FixedAgentPreview = {
  id: string;
  name: string;
  kind: "skill";
  source: string;
  shortRole: string;
};

const CUSTOM_AGENT_COUNT = 8;

const defaultCustomAgents: CustomAgentInput[] = Array.from({ length: CUSTOM_AGENT_COUNT }, (_, index) => ({
  name: `自定义 Agent ${index + 1}`,
  role: ""
}));

const submitLabels: Array<{ key: keyof SubmitAnswers; label: string }> = [
  { key: "insight", label: "提交 1" },
  { key: "concept", label: "提交 2" },
  { key: "evidence", label: "提交 3" },
  { key: "nextStep", label: "提交 4" }
];

function normalizeSnippet(text: string) {
  return text.replace(/\s+/g, " ").trim();
}

function renderHighlightedText(text: string, snippets: string[]) {
  const uniqueSnippets = Array.from(new Set(snippets.map(normalizeSnippet).filter(Boolean))).sort(
    (left, right) => right.length - left.length
  );

  let segments: Array<{ text: string; highlighted: boolean }> = [{ text, highlighted: false }];
  for (const snippet of uniqueSnippets) {
    segments = segments.flatMap((segment) => {
      if (segment.highlighted || !segment.text.includes(snippet)) {
        return [segment];
      }

      const pieces = segment.text.split(snippet);
      return pieces.flatMap((piece, index) => {
        const next = [{ text: piece, highlighted: false }];
        if (index < pieces.length - 1) {
          next.push({ text: snippet, highlighted: true });
        }
        return next;
      });
    });
  }

  return segments.map((segment, index) =>
    segment.highlighted ? (
      <mark key={`${segment.text}-${index}`}>{segment.text}</mark>
    ) : (
      <span key={`${segment.text}-${index}`}>{segment.text}</span>
    )
  );
}

export default function Home() {
  const [fixedAgents, setFixedAgents] = useState<FixedAgentPreview[]>([]);
  const [customAgents, setCustomAgents] = useState<CustomAgentInput[]>(defaultCustomAgents);
  const [briefText, setBriefText] = useState("");
  const [briefFileName, setBriefFileName] = useState("");
  const [initialPrompt, setInitialPrompt] = useState("");
  const [messages, setMessages] = useState<DiscussionMessage[]>([]);
  const [notes, setNotes] = useState<Note[]>([]);
  const [highlights, setHighlights] = useState<Record<string, string[]>>({});
  const [summary, setSummary] = useState("");
  const [submissions, setSubmissions] = useState<SubmitAnswers>({
    insight: "",
    concept: "",
    evidence: "",
    nextStep: ""
  });
  const [currentTurn, setCurrentTurn] = useState(0);
  const [running, setRunning] = useState(false);
  const [paused, setPaused] = useState(false);
  const [awaitingTurn, setAwaitingTurn] = useState(false);
  const [summarizing, setSummarizing] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [parsing, setParsing] = useState(false);
  const [error, setError] = useState("");

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    let mounted = true;
    fetch("/api/agents/fixed")
      .then(async (response) => {
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || "读取固定 Agent 失败。");
        }
        if (mounted) {
          setFixedAgents(data.agents);
        }
      })
      .catch((fetchError: Error) => setError(fetchError.message));

    return () => {
      mounted = false;
    };
  }, []);

  const activeCustomAgents = useMemo(
    () =>
      customAgents
        .map((agent, index) => ({
          name: agent.name.trim() || `自定义 Agent ${index + 1}`,
          role: agent.role.trim()
        }))
        .filter((agent) => agent.role.length > 0),
    [customAgents]
  );

  const roster = useMemo(
    () => [
      ...fixedAgents.map((agent) => ({ id: agent.id, name: agent.name, kind: "skill" as const })),
      ...activeCustomAgents.map((agent, index) => ({
        id: `custom-agent-${index + 1}`,
        name: agent.name,
        kind: "custom" as const
      }))
    ],
    [activeCustomAgents, fixedAgents]
  );

  const totalTurns = roster.length * 3;
  const readyToStart = fixedAgents.length === 4 && briefText.trim().length > 0 && initialPrompt.trim().length > 0;

  const progress = totalTurns > 0 ? Math.round((messages.length / totalTurns) * 100) : 0;
  const activeAgent = roster.length > 0 ? roster[currentTurn % roster.length] : undefined;

  const parseFile = async (file: File) => {
    setError("");
    setParsing(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("/api/document/parse", {
        method: "POST",
        body: formData
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "解析文档失败。");
      }
      setBriefText(data.text);
      setBriefFileName(data.fileName);
    } catch (parseError) {
      setError(parseError instanceof Error ? parseError.message : "解析文档失败。");
    } finally {
      setParsing(false);
    }
  };

  const requestSummary = useCallback(
    async (history: DiscussionMessage[]) => {
      setSummarizing(true);
      setError("");
      try {
        const response = await fetch("/api/discussion/summary", {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            brief: briefText,
            prompt: initialPrompt,
            history,
            notes,
            expectedTurns: totalTurns
          })
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || "总结失败。");
        }
        setSummary(data.summary);
      } catch (summaryError) {
        setError(summaryError instanceof Error ? summaryError.message : "总结失败。");
      } finally {
        setSummarizing(false);
      }
    },
    [briefText, initialPrompt, notes, totalTurns]
  );

  const runNextTurn = useCallback(async () => {
    if (!running || paused || awaitingTurn || currentTurn >= totalTurns) {
      return;
    }

    setAwaitingTurn(true);
    setError("");
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const response = await fetch("/api/discussion/step", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        signal: controller.signal,
        body: JSON.stringify({
          customAgents: activeCustomAgents,
          brief: briefText,
          prompt: initialPrompt,
          history: messages,
          turnIndex: currentTurn
        })
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "生成发言失败。");
      }

      const updatedHistory = [...messages, data.message as DiscussionMessage];
      const nextTurn = currentTurn + 1;
      setMessages(updatedHistory);
      setCurrentTurn(nextTurn);

      if (nextTurn >= totalTurns) {
        setRunning(false);
        setPaused(false);
        await requestSummary(updatedHistory);
      }
    } catch (turnError) {
      if (turnError instanceof DOMException && turnError.name === "AbortError") {
        return;
      }
      setRunning(false);
      setError(turnError instanceof Error ? turnError.message : "生成发言失败。");
    } finally {
      setAwaitingTurn(false);
      abortRef.current = null;
    }
  }, [
    awaitingTurn,
    activeCustomAgents,
    briefText,
    currentTurn,
    initialPrompt,
    messages,
    paused,
    requestSummary,
    running,
    totalTurns
  ]);

  useEffect(() => {
    if (running && !paused && !awaitingTurn && currentTurn < totalTurns) {
      const timeoutId = window.setTimeout(() => {
        void runNextTurn();
      }, 0);

      return () => window.clearTimeout(timeoutId);
    }
  }, [awaitingTurn, currentTurn, paused, runNextTurn, running, totalTurns]);

  const startRun = () => {
    if (!readyToStart) {
      setError("请确认 4 个固定 Skill Agent 已读取，并完成 brief 文档和初始 prompt。");
      return;
    }
    abortRef.current?.abort();
    setMessages([]);
    setNotes([]);
    setHighlights({});
    setSummary("");
    setCurrentTurn(0);
    setError("");
    setPaused(false);
    setRunning(true);
  };

  const pauseRun = () => {
    abortRef.current?.abort();
    setPaused(true);
  };

  const stopRun = () => {
    abortRef.current?.abort();
    setRunning(false);
    setPaused(false);
    setAwaitingTurn(false);
  };

  const resetRun = () => {
    abortRef.current?.abort();
    setMessages([]);
    setNotes([]);
    setHighlights({});
    setSummary("");
    setCurrentTurn(0);
    setRunning(false);
    setPaused(false);
    setAwaitingTurn(false);
    setError("");
  };

  const captureSelection = () => {
    const selection = window.getSelection();
    if (!selection || selection.isCollapsed || selection.rangeCount === 0) {
      return;
    }

    const text = normalizeSnippet(selection.toString());
    if (!text) {
      return;
    }

    const range = selection.getRangeAt(0);
    const node = range.commonAncestorContainer;
    const element = node.nodeType === Node.ELEMENT_NODE ? (node as Element) : node.parentElement;
    const messageElement = element?.closest("[data-message-id]");
    const messageId = messageElement?.getAttribute("data-message-id");
    const sourceMessage = messages.find((message) => message.id === messageId);
    if (!messageId || !sourceMessage || !sourceMessage.content.includes(text)) {
      return;
    }

    setHighlights((current) => ({
      ...current,
      [messageId]: Array.from(new Set([...(current[messageId] ?? []), text]))
    }));
    setNotes((current) => [
      ...current,
      {
        id: `note-${Date.now()}`,
        text,
        messageId,
        agentName: sourceMessage.agentName,
        createdAt: new Date().toISOString()
      }
    ]);
    selection.removeAllRanges();
  };

  const exportResult = () => {
    const payload = {
      exportedAt: new Date().toISOString(),
      briefFileName,
      brief: briefText,
      prompt: initialPrompt,
      fixedAgents,
      customAgents,
      activeCustomAgents,
      participatingAgents: roster,
      messages,
      notes,
      summary,
      submissions
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: "application/json;charset=utf-8"
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `design-agent-run-${Date.now()}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">LangChain / LangGraph</p>
          <h1>Design Brief Agent Lab</h1>
        </div>
        <div className="status-strip">
          <span>{messages.length}/{totalTurns || 0}</span>
          <div className="progress-track" aria-label="讨论进度">
            <div style={{ width: `${progress}%` }} />
          </div>
          <span>{summarizing ? "总结中" : running ? (paused ? "已暂停" : "讨论中") : summary ? "已完成" : "待开始"}</span>
        </div>
      </header>

      <section className="workspace">
        <aside className="setup-panel" aria-label="实验设置">
          <div className="panel-heading">
            <h2>实验设置</h2>
            <button className="icon-button" type="button" title="重置实验" onClick={resetRun}>
              <RotateCcw size={18} />
            </button>
          </div>

          <section className="setup-section">
            <h3>固定 Skill Agent</h3>
            <div className="fixed-agent-list">
              {fixedAgents.map((agent) => (
                <div className="fixed-agent-row" key={agent.id}>
                  <strong>{agent.name}</strong>
                  <span>{agent.source}</span>
                </div>
              ))}
              {fixedAgents.length === 0 && <p className="muted">读取中</p>}
            </div>
          </section>

          <section className="setup-section">
            <h3>自定义 Agent</h3>
            <div className="custom-agent-list">
              {customAgents.map((agent, index) => (
                <div className="agent-editor" key={index}>
                  <input
                    value={agent.name}
                    aria-label={`自定义 Agent ${index + 1} 名称`}
                    onChange={(event) => {
                      const next = [...customAgents];
                      next[index] = { ...next[index], name: event.target.value };
                      setCustomAgents(next);
                    }}
                  />
                  <textarea
                    value={agent.role}
                    rows={3}
                    placeholder="角色设定"
                    aria-label={`自定义 Agent ${index + 1} 角色设定`}
                    onChange={(event) => {
                      const next = [...customAgents];
                      next[index] = { ...next[index], role: event.target.value };
                      setCustomAgents(next);
                    }}
                  />
                </div>
              ))}
            </div>
          </section>
        </aside>

        <section className="main-panel">
          <div className="input-band">
            <div
              className={`dropzone ${dragging ? "is-dragging" : ""}`}
              onDragOver={(event) => {
                event.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={(event) => {
                event.preventDefault();
                setDragging(false);
                const file = event.dataTransfer.files[0];
                if (file) {
                  void parseFile(file);
                }
              }}
              role="button"
              tabIndex={0}
              onClick={() => fileInputRef.current?.click()}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  fileInputRef.current?.click();
                }
              }}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".txt,.md,.pdf,.docx"
                hidden
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) {
                    void parseFile(file);
                  }
                }}
              />
              {parsing ? <Loader2 className="spin" size={22} /> : <Upload size={22} />}
              <div>
                <strong>{briefFileName || "拖入 Brief 文档"}</strong>
                <span>{briefText ? `${briefText.length} 字符` : ".txt / .md / .pdf / .docx"}</span>
              </div>
            </div>

            <label className="prompt-box">
              <span>初始 Prompt</span>
              <textarea
                value={initialPrompt}
                rows={4}
                placeholder="输入本次讨论的目标、关注点或输出偏好"
                onChange={(event) => setInitialPrompt(event.target.value)}
              />
            </label>
          </div>

          <div className="runbar">
            <div>
              <strong>{activeAgent ? activeAgent.name : "等待 Agent"}</strong>
              <span>第 {Math.min(currentTurn + 1, totalTurns || 1)} 条</span>
            </div>
            <div className="run-actions">
              {!running ? (
                <button className="primary-button" type="button" onClick={startRun} disabled={!readyToStart || parsing}>
                  <Play size={18} />
                  开始
                </button>
              ) : paused ? (
                <button className="primary-button" type="button" onClick={() => setPaused(false)}>
                  <Play size={18} />
                  继续
                </button>
              ) : (
                <button className="secondary-button" type="button" onClick={pauseRun}>
                  <Pause size={18} />
                  暂停
                </button>
              )}
              <button className="icon-button" type="button" title="停止" onClick={stopRun} disabled={!running && !awaitingTurn}>
                <Square size={17} />
              </button>
            </div>
          </div>

          {error && <div className="error-bar">{error}</div>}

          <section className="discussion-panel" onMouseUp={captureSelection} aria-label="讨论记录">
            <div className="section-title">
              <BookOpenText size={18} />
              <h2>共享讨论空间</h2>
            </div>

            <div className="message-list">
              {messages.map((message) => (
                <article className="message-card" key={message.id} data-message-id={message.id}>
                  <div className="message-meta">
                    <strong>{message.agentName}</strong>
                    <span>第 {message.round} 轮 · {message.turnIndex + 1}/{totalTurns}</span>
                  </div>
                  <p>{renderHighlightedText(message.content, highlights[message.id] ?? [])}</p>
                </article>
              ))}

              {awaitingTurn && (
                <article className="message-card is-loading">
                  <div className="message-meta">
                    <strong>{activeAgent?.name ?? "Agent"}</strong>
                    <span>生成中</span>
                  </div>
                  <p>
                    <Loader2 className="spin inline-icon" size={17} />
                    正在发言
                  </p>
                </article>
              )}

              {messages.length === 0 && !awaitingTurn && (
                <div className="empty-state">
                  <FileText size={24} />
                  <span>等待开始</span>
                </div>
              )}
            </div>
          </section>

          {(summary || summarizing) && (
            <section className="summary-panel">
              <div className="section-title">
                <Send size={18} />
                <h2>总结 Agent</h2>
              </div>
              {summarizing ? (
                <p className="muted">
                  <Loader2 className="spin inline-icon" size={17} />
                  正在总结
                </p>
              ) : (
                <p>{summary}</p>
              )}
            </section>
          )}

          {summary && (
            <section className="submission-panel">
              <div className="section-title">
                <Send size={18} />
                <h2>最终提交</h2>
              </div>
              <div className="submission-grid">
                {submitLabels.map((item) => (
                  <label key={item.key}>
                    <span>{item.label}</span>
                    <textarea
                      rows={4}
                      value={submissions[item.key]}
                      onChange={(event) =>
                        setSubmissions((current) => ({
                          ...current,
                          [item.key]: event.target.value
                        }))
                      }
                    />
                  </label>
                ))}
              </div>
              <button className="primary-button export-button" type="button" onClick={exportResult}>
                <Download size={18} />
                导出结果
              </button>
            </section>
          )}
        </section>

        <aside className="notes-panel" aria-label="笔记本">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">Notebook</p>
              <h2>笔记本</h2>
            </div>
            <Highlighter size={18} />
          </div>
          <div className="note-list">
            {notes.map((note) => (
              <article className="note-card" key={note.id}>
                <strong>{note.agentName}</strong>
                <p>{note.text}</p>
              </article>
            ))}
            {notes.length === 0 && <p className="muted">选中讨论文本后记录</p>}
          </div>
        </aside>
      </section>
    </main>
  );
}

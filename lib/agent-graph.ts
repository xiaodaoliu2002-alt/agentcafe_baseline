import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { HumanMessage, SystemMessage, type BaseMessage } from "@langchain/core/messages";
import { StateGraph, START, END, Annotation } from "@langchain/langgraph";
import { ChatAnthropic } from "@langchain/anthropic";
import { Anthropic } from "@anthropic-ai/sdk";
import { ChatOpenAI } from "@langchain/openai";
import type {
  AgentConfig,
  DiscussionMessage,
  DiscussionSettings,
  Note,
  SummaryMode
} from "@/lib/types";
import { clipText, countCjkLikeChars, enforceMaxChars, trimModelText } from "@/lib/text";

const DiscussionState = Annotation.Root({
  agents: Annotation<AgentConfig[]>({
    reducer: (_left, right) => right,
    default: () => []
  }),
  brief: Annotation<string>({
    reducer: (_left, right) => right,
    default: () => ""
  }),
  prompt: Annotation<string>({
    reducer: (_left, right) => right,
    default: () => ""
  }),
  settings: Annotation<DiscussionSettings>({
    reducer: (_left, right) => right,
    default: () => ({
      participantCount: 6,
      roundCount: 3,
      speechesPerAgentPerRound: 1
    })
  }),
  history: Annotation<DiscussionMessage[]>({
    reducer: (_left, right) => right,
    default: () => []
  }),
  turnIndex: Annotation<number>({
    reducer: (_left, right) => right,
    default: () => 0
  }),
  message: Annotation<DiscussionMessage | null>({
    reducer: (_left, right) => right,
    default: () => null
  })
});

const SummaryState = Annotation.Root({
  brief: Annotation<string>({
    reducer: (_left, right) => right,
    default: () => ""
  }),
  prompt: Annotation<string>({
    reducer: (_left, right) => right,
    default: () => ""
  }),
  history: Annotation<DiscussionMessage[]>({
    reducer: (_left, right) => right,
    default: () => []
  }),
  notes: Annotation<Note[]>({
    reducer: (_left, right) => right,
    default: () => []
  }),
  mode: Annotation<SummaryMode>({
    reducer: (_left, right) => right,
    default: () => "round"
  }),
  summary: Annotation<string>({
    reducer: (_left, right) => right,
    default: () => ""
  })
});

function getAnthropicModelName() {
  return (
    process.env.ANTHROPIC_MODEL ||
    process.env.ANTHROPIC_DEFAULT_SONNET_MODEL ||
    process.env.ANTHROPIC_DEFAULT_OPUS_MODEL ||
    process.env.ANTHROPIC_DEFAULT_HAIKU_MODEL ||
    "claude-sonnet-4-5-20250929"
  );
}

function normalizeCredential(value: string) {
  return value.replace(/^Bearer\s+/i, "").trim();
}

let openAiKeyCursor = 0;

function splitCredentialList(value: string | undefined) {
  return (value ?? "")
    .split(/[\n,;]+/)
    .map(normalizeCredential)
    .filter(Boolean);
}

function getNextOpenAIKey(keys = splitCredentialList(process.env.OPENAI_API_KEYS)) {
  if (keys.length === 0) {
    return process.env.OPENAI_API_KEY ? normalizeCredential(process.env.OPENAI_API_KEY) : "";
  }

  const key = keys[openAiKeyCursor % keys.length];
  openAiKeyCursor = (openAiKeyCursor + 1) % keys.length;
  return key;
}

function normalizeOpenAIBaseURL(value: string | undefined) {
  const clean = value?.replace(/\/+$/, "");
  if (!clean) {
    return undefined;
  }
  return clean.endsWith("/v1") ? clean : `${clean}/v1`;
}

function parseOpenAIApiKeys() {
  const raw = process.env.OPENAI_API_KEYS?.trim();
  if (!raw) {
    return [];
  }

  try {
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed)) {
      return parsed.map((item) => String(item).trim()).filter(Boolean);
    }
  } catch {
    // Plain delimiter-separated values are the expected local env format.
  }

  return splitCredentialList(raw);
}

function shouldUseOpenAICompatibleProxy(model: string) {
  const provider = (process.env.MODEL_PROVIDER || process.env.LLM_PROVIDER || process.env.ANTHROPIC_API_FORMAT || "")
    .toLowerCase()
    .trim();

  return provider === "openai-compatible" || provider === "openai" || /^gpt[-_]/i.test(model);
}

function getModel(options: { purpose?: "discussion" | "summary"; apiKeyIndex?: number } = {}): BaseChatModel {
  const openAiKeys = parseOpenAIApiKeys();
  const indexedOpenAiKey =
    typeof options.apiKeyIndex === "number" && options.apiKeyIndex >= 0 ? openAiKeys[options.apiKeyIndex] : undefined;
  const summaryOpenAiKey = options.purpose === "summary" ? openAiKeys[6] : undefined;
  const selectedOpenAiKey = indexedOpenAiKey || summaryOpenAiKey || getNextOpenAIKey(openAiKeys);
  const anthropicBaseUrl = process.env.ANTHROPIC_BASE_URL;
  const anthropicModel = getAnthropicModelName();
  const anthropicAuthToken = process.env.ANTHROPIC_AUTH_TOKEN
    ? normalizeCredential(process.env.ANTHROPIC_AUTH_TOKEN)
    : "";
  const anthropicApiKey = process.env.ANTHROPIC_API_KEY ? normalizeCredential(process.env.ANTHROPIC_API_KEY) : "";
  const anthropicCredential = anthropicApiKey || anthropicAuthToken;

  if (selectedOpenAiKey) {
    return new ChatOpenAI({
      apiKey: selectedOpenAiKey,
      model: process.env.OPENAI_MODEL || "gpt-4.1-mini",
      temperature: Number(process.env.OPENAI_TEMPERATURE ?? 0.7),
      configuration: process.env.OPENAI_BASE_URL
        ? {
            baseURL: normalizeOpenAIBaseURL(process.env.OPENAI_BASE_URL)
          }
        : undefined
    });
  }

  if (anthropicCredential && anthropicBaseUrl && shouldUseOpenAICompatibleProxy(anthropicModel)) {
    return new ChatOpenAI({
      apiKey: anthropicCredential,
      model: anthropicModel,
      temperature: Number(process.env.ANTHROPIC_TEMPERATURE ?? process.env.OPENAI_TEMPERATURE ?? 0.7),
      configuration: {
        baseURL: normalizeOpenAIBaseURL(anthropicBaseUrl)
      }
    });
  }

  if (anthropicApiKey) {
    return new ChatAnthropic({
      apiKey: anthropicApiKey,
      model: anthropicModel,
      temperature: Number(process.env.ANTHROPIC_TEMPERATURE ?? process.env.OPENAI_TEMPERATURE ?? 0.7),
      anthropicApiUrl: anthropicBaseUrl,
      maxTokens: Number(process.env.ANTHROPIC_MAX_TOKENS ?? 1400)
    });
  }

  if (anthropicAuthToken) {
    return new ChatAnthropic({
      model: anthropicModel,
      temperature: Number(process.env.ANTHROPIC_TEMPERATURE ?? process.env.OPENAI_TEMPERATURE ?? 0.7),
      maxTokens: Number(process.env.ANTHROPIC_MAX_TOKENS ?? 1400),
      createClient: (options) =>
        new Anthropic({
          ...options,
          apiKey: null,
          authToken: anthropicAuthToken,
          baseURL: anthropicBaseUrl ?? options.baseURL
        })
    });
  }

  if (!selectedOpenAiKey) {
    throw new Error(
      "缺少模型 API 配置。请在 .env.local 中配置 OPENAI_API_KEYS，或使用 OPENAI_API_KEY。"
    );
  }

  return new ChatOpenAI({
    apiKey: selectedOpenAiKey,
    model: process.env.OPENAI_MODEL || "gpt-4.1-mini",
    temperature: Number(process.env.OPENAI_TEMPERATURE ?? 0.7),
    configuration: process.env.OPENAI_BASE_URL
      ? {
          baseURL: normalizeOpenAIBaseURL(process.env.OPENAI_BASE_URL)
        }
      : undefined
  });
}

function formatHistory(history: DiscussionMessage[]) {
  if (history.length === 0) {
    return "暂无历史发言。";
  }

  return history
    .map((item, index) => {
      const order = item.kind === "summary" ? `总结${index + 1}` : `第${item.turnIndex + 1}条`;
      return `${order} / 第${item.round}轮 / ${item.agentName}：${item.content}`;
    })
    .join("\n");
}

function formatMessageList(messages: DiscussionMessage[], emptyText: string) {
  if (messages.length === 0) {
    return emptyText;
  }

  return messages
    .map((item, index) => `${index + 1}. 第${item.round}轮 / ${item.agentName}：${clipText(item.content, 260)}`)
    .join("\n");
}

function formatDeduplicationContext(history: DiscussionMessage[], speaker: AgentConfig) {
  const discussionMessages = history.filter((item) => item.kind !== "summary");
  const recentDiscussionMessages = discussionMessages.slice(-12);
  const ownRecentMessages = discussionMessages.filter((item) => item.agentId === speaker.id).slice(-2);
  const recentSummaryMessages = history.filter((item) => item.kind === "summary").slice(-3);

  return [
    "最近 12 条 Agent 发言（用于识别最近6-12条内容中已出现的观点，不要复述）：",
    formatMessageList(recentDiscussionMessages, "暂无近期 Agent 发言。"),
    "",
    "你自己最近 2 次发言（不要重复自己的表达和观点）：",
    formatMessageList(ownRecentMessages, "你此前还没有发言。"),
    "",
    "总结 Agent 表达（禁止复述总结 Agent 的表达，只能回应其留下的问题或张力）：",
    formatMessageList(recentSummaryMessages, "暂无总结 Agent 表达。")
  ].join("\n");
}

function isOpeningTurnOfRound(turnIndex: number, turnsPerRound: number) {
  return turnsPerRound > 0 && turnIndex % turnsPerRound === 0;
}

function getDiscussionAction(turnIndex: number, history: DiscussionMessage[], opensNewRound: boolean) {
  if (opensNewRound || history.filter((item) => item.kind !== "summary").length === 0) {
    return "本轮第一位 Agent：开启本轮新的观点，提出一个清晰、可被其他 Agent 回应的观察、张力或问题；不要回应上一轮的具体发言。";
  }

  const actions = [
    "支持并补充：回应目标发言的具体观点，说明你认同的原因，并补充一个新的原因、边界或使用情境。",
    "追问澄清：回应目标发言的具体观点，指出其中模糊、跳跃或需要证据的地方，并提出一个推进讨论的问题。",
    "反驳或提出边界：回应目标发言的具体假设，指出可能不成立的条件、风险、反例或被忽略的人群。",
    "补充场景或反例：回应目标发言的具体观点，给出一个新的使用场景、极端案例或反向例子来拓宽讨论。",
    "转译为设计机会：回应目标发言的具体观点，把它转化成一个可探索的设计机会、功能方向或验证假设。",
    "整合张力：回应目标发言的具体观点，把前文至少两个观点之间的共识或冲突整理成一个下一步问题。"
  ];

  return actions[(turnIndex - 1) % actions.length];
}

function selectResponseTarget(history: DiscussionMessage[], turnIndex: number, opensNewRound: boolean) {
  if (opensNewRound) {
    return undefined;
  }

  const candidates = history.filter((item) => item.kind !== "summary").slice(-12);
  if (candidates.length === 0) {
    return undefined;
  }

  const offsetPattern = [0, 1, 2, 0, 3, 1];
  const maxOffset = Math.min(candidates.length - 1, offsetPattern[turnIndex % offsetPattern.length]);
  return candidates[candidates.length - 1 - maxOffset];
}

function formatInteractionContext(history: DiscussionMessage[], turnIndex: number, turnsPerRound: number) {
  const opensNewRound = isOpeningTurnOfRound(turnIndex, turnsPerRound);
  const responseTarget = selectResponseTarget(history, turnIndex, opensNewRound);
  const targetText = responseTarget
    ? `本次回应对象：第${responseTarget.round}轮 / ${responseTarget.agentName}：${clipText(responseTarget.content, 360)}`
    : "本次回应对象：本轮第一位 Agent 不回应之前发言，请开启本轮新的观点，并让后续 Agent 有内容可接。";

  return [
    "回应对象选择规则：本次可回应上一条，也可回应更早的一条 Agent 发言；不要默认总是回应上一条。",
    targetText,
    `本次讨论动作：${getDiscussionAction(turnIndex, history, opensNewRound)}`
  ].join("\n");
}

function formatAgentRoster(agents: AgentConfig[]) {
  return agents.map((agent, index) => `${index + 1}. ${agent.name}（${agent.kind === "skill" ? "skill" : "自定义"}）`).join("\n");
}

function messageContentToString(message: BaseMessage) {
  if (typeof message.content === "string") {
    return message.content;
  }

  return message.content
    .map((part) => {
      if (typeof part === "string") {
        return part;
      }
      if ("text" in part && typeof part.text === "string") {
        return part.text;
      }
      return "";
    })
    .join("");
}

async function invokeAgentSpeech(params: {
  model: BaseChatModel;
  agents: AgentConfig[];
  speaker: AgentConfig;
  brief: string;
  prompt: string;
  history: DiscussionMessage[];
  turnIndex: number;
  settings: DiscussionSettings;
  retry?: boolean;
}) {
  const turnsPerRound = params.agents.length * params.settings.speechesPerAgentPerRound;
  const round = Math.floor(params.turnIndex / turnsPerRound) + 1;
  const speechPass = Math.floor((params.turnIndex % turnsPerRound) / params.agents.length) + 1;
  const totalTurns = turnsPerRound * params.settings.roundCount;
  const deduplicationContext = formatDeduplicationContext(params.history, params.speaker);
  const interactionContext = formatInteractionContext(params.history, params.turnIndex, turnsPerRound);
  const system = [
    `你是一个多 Agent 设计 brief 研讨系统中的发言者：${params.speaker.name}。`,
    `你与其他 ${Math.max(params.agents.length - 1, 0)} 个 Agent 位于同一个共享讨论空间，你能看到此前完整讨论记录。`,
    "你必须严格遵守：只输出一段中文发言；不要标题、编号、寒暄、自我介绍或 Markdown；长度不超过150个中文字符；不要重复已有观点；每次发言要推进设计洞察。",
    "除非本次互动要求你开启新一轮观点，否则你的发言必须基于此前完整讨论记录，承接、回应或推进前文至少一个具体观点。",
    "除非本次互动要求你作为本轮第一位 Agent 开启新观点，否则你必须回应指定对象，并执行指定讨论动作。请在发言中自然点名回应对象或明确承接其具体观点，不要只发表独立看法。",
    "发言前必须做去重检查：不要重复最近6-12条内容中已出现的观点；不要重复你自己前两次说过的内容；本次最好新增一个新角度，或者深入前面的观点；如果你同意前文，也必须补充新的原因或边界条件；禁止复述总结 Agent 的表达。",
    `当前是第 ${round}/${params.settings.roundCount} 轮，本轮你第 ${speechPass}/${params.settings.speechesPerAgentPerRound} 次发言，也是全局第 ${params.turnIndex + 1}/${totalTurns} 条 Agent 发言。`,
    params.retry ? "上一版长度不符合要求，请重写为150个中文字符以内。" : "",
    "",
    "你的角色/skill 设定如下：",
    clipText(params.speaker.role, 14000)
  ]
    .filter(Boolean)
    .join("\n");

  const human = [
    "初始设计 brief：",
    clipText(params.brief, 12000),
    "",
    "用户初始 prompt：",
    clipText(params.prompt, 3000),
    "",
    "参与 Agent 列表：",
    formatAgentRoster(params.agents),
    "",
    "此前完整讨论记录：",
    clipText(formatHistory(params.history), 18000),
    "",
    "本次发言去重上下文：",
    clipText(deduplicationContext, 6000),
    "",
    "本次互动要求：",
    clipText(interactionContext, 1600),
    "",
    `请以 ${params.speaker.name} 的身份继续发言。`
  ].join("\n");

  const response = await params.model.invoke([new SystemMessage(system), new HumanMessage(human)]);
  return trimModelText(messageContentToString(response));
}

export async function runDiscussionStep(input: {
  agents: AgentConfig[];
  brief: string;
  prompt: string;
  history: DiscussionMessage[];
  turnIndex: number;
  settings: DiscussionSettings;
}) {
  const workflow = new StateGraph(DiscussionState)
    .addNode("agent_turn", async (state) => {
      const turnsPerRound = state.agents.length * state.settings.speechesPerAgentPerRound;
      const speaker = state.agents[state.turnIndex % state.agents.length];
      const model = getModel({
        purpose: "discussion",
        apiKeyIndex: speaker.apiKeyIndex
      });
      const round = Math.floor(state.turnIndex / turnsPerRound) + 1;
      let content = await invokeAgentSpeech({
        model,
        agents: state.agents,
        speaker,
        brief: state.brief,
        prompt: state.prompt,
        history: state.history,
        turnIndex: state.turnIndex,
        settings: state.settings
      });

      const length = countCjkLikeChars(content);
      if (length > 150) {
        content = await invokeAgentSpeech({
          model,
          agents: state.agents,
          speaker,
          brief: state.brief,
          prompt: state.prompt,
          history: state.history,
          turnIndex: state.turnIndex,
          settings: state.settings,
          retry: true
        });
      }

      const message: DiscussionMessage = {
        id: `turn-${state.turnIndex + 1}-${Date.now()}`,
        agentId: speaker.id,
        agentName: speaker.name,
        kind: speaker.kind,
        round,
        turnIndex: state.turnIndex,
        content: enforceMaxChars(content, 150),
        createdAt: new Date().toISOString()
      };

      return {
        message,
        history: [...state.history, message]
      };
    })
    .addEdge(START, "agent_turn")
    .addEdge("agent_turn", END)
    .compile();

  const result = await workflow.invoke(input);
  if (!result.message) {
    throw new Error("Agent 没有生成发言。");
  }
  return result.message;
}

export async function runSummary(input: {
  brief: string;
  prompt: string;
  history: DiscussionMessage[];
  notes: Note[];
  mode: SummaryMode;
  round?: number;
}) {
  const workflow = new StateGraph(SummaryState)
    .addNode("summary_agent", async (state) => {
      const model = getModel({ purpose: "summary" });
      const system =
        state.mode === "final"
          ? [
              "你负责将目前讨论空间内的所有文本转化为可直接给用户看的结构化设计洞察。",
              "请严格使用三个小标题：用户需求、设计问题、设计方向。",
              "三个小标题下合计输出四条洞察：用户需求1条，设计问题1条，设计方向2条。",
              "四条洞察总字符数必须500字以内，可直接给用户看。",
              "不要写成长段落，不要复述每个人发言，不要展示额外推理过程。"
            ].join("\n")
          : [
              "以上是目前讨论的内容和用户高亮的内容，请阅读并从中总结结束语。",
              "结束语的任务不是完整总结，而是让参与者看见本轮的pattern：共识、转变、隐藏观点、张力。",
              "用关键词式短句，300字以内，最多4行；不要写成长段落，不要复述每个人发言。"
            ].join("\n");

      const notesText =
        state.notes.length === 0
          ? "暂无用户高亮笔记。"
          : state.notes.map((note, index) => `${index + 1}. ${note.agentName}：${note.text}`).join("\n");

      const human = [
        `总结类型：${state.mode === "final" ? "全部轮次最终总结" : "本轮结束语"}`,
        "",
        "初始设计 brief：",
        clipText(state.brief, 12000),
        "",
        "用户初始 prompt：",
        clipText(state.prompt, 3000),
        "",
        "Agent 讨论记录：",
        clipText(formatHistory(state.history), 24000),
        "",
        "用户高亮笔记：",
        clipText(notesText, 8000)
      ].join("\n");

      const response = await model.invoke([new SystemMessage(system), new HumanMessage(human)]);
      const maxChars = state.mode === "final" ? 800 : 300;
      return {
        summary: enforceMaxChars(messageContentToString(response), maxChars)
      };
    })
    .addEdge(START, "summary_agent")
    .addEdge("summary_agent", END)
    .compile();

  const result = await workflow.invoke(input);
  return result.summary;
}

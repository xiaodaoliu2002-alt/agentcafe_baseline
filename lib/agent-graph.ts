import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { HumanMessage, SystemMessage, type BaseMessage } from "@langchain/core/messages";
import { StateGraph, START, END, Annotation } from "@langchain/langgraph";
import { ChatAnthropic } from "@langchain/anthropic";
import { Anthropic } from "@anthropic-ai/sdk";
import { ChatOpenAI } from "@langchain/openai";
import type { AgentConfig, DiscussionMessage, Note } from "@/lib/types";
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

function normalizeOpenAIBaseURL(value: string | undefined) {
  const clean = value?.replace(/\/+$/, "");
  if (!clean) {
    return undefined;
  }
  return clean.endsWith("/v1") ? clean : `${clean}/v1`;
}

function shouldUseOpenAICompatibleProxy(model: string) {
  const provider = (process.env.MODEL_PROVIDER || process.env.LLM_PROVIDER || process.env.ANTHROPIC_API_FORMAT || "")
    .toLowerCase()
    .trim();

  return provider === "openai-compatible" || provider === "openai" || /^gpt[-_]/i.test(model);
}

function getModel(): BaseChatModel {
  const openAiKey = process.env.OPENAI_API_KEY;
  const anthropicBaseUrl = process.env.ANTHROPIC_BASE_URL;
  const anthropicModel = getAnthropicModelName();
  const anthropicAuthToken = process.env.ANTHROPIC_AUTH_TOKEN
    ? normalizeCredential(process.env.ANTHROPIC_AUTH_TOKEN)
    : "";
  const anthropicApiKey = process.env.ANTHROPIC_API_KEY ? normalizeCredential(process.env.ANTHROPIC_API_KEY) : "";
  const anthropicCredential = anthropicApiKey || anthropicAuthToken;

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

  if (!openAiKey) {
    throw new Error(
      "缺少模型 API 配置。请在 .env.local 中配置 ANTHROPIC_AUTH_TOKEN，或使用 OPENAI_API_KEY。"
    );
  }

  return new ChatOpenAI({
    apiKey: openAiKey,
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
    .map((item) => `第${item.turnIndex + 1}条 / 第${item.round}轮 / ${item.agentName}：${item.content}`)
    .join("\n");
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
  retry?: boolean;
}) {
  const round = Math.floor(params.turnIndex / params.agents.length) + 1;
  const totalTurns = params.agents.length * 3;
  const system = [
    `你是一个多 Agent 设计 brief 研讨系统中的发言者：${params.speaker.name}。`,
    `你与其他 ${Math.max(params.agents.length - 1, 0)} 个 Agent 位于同一个共享讨论空间，你能看到此前完整讨论记录。`,
    "你必须严格遵守：只输出一段中文发言；不要标题、编号、寒暄、自我介绍或 Markdown；长度控制在 200-250 个中文字符；不要重复已有观点；每次发言要推进设计洞察。",
    "你的发言可以包含：设计机会、用户矛盾、隐含假设、风险、判断依据、值得验证的问题或概念方向。",
    `当前是你第 ${round}/3 次发言，也是全局第 ${params.turnIndex + 1}/${totalTurns} 条 Agent 发言。`,
    params.retry ? "上一版长度不符合要求，请重写为 200-250 个中文字符。" : "",
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
}) {
  const workflow = new StateGraph(DiscussionState)
    .addNode("agent_turn", async (state) => {
      const model = getModel();
      const speaker = state.agents[state.turnIndex % state.agents.length];
      const round = Math.floor(state.turnIndex / state.agents.length) + 1;
      let content = await invokeAgentSpeech({
        model,
        agents: state.agents,
        speaker,
        brief: state.brief,
        prompt: state.prompt,
        history: state.history,
        turnIndex: state.turnIndex
      });

      const length = countCjkLikeChars(content);
      if (length < 190 || length > 260) {
        content = await invokeAgentSpeech({
          model,
          agents: state.agents,
          speaker,
          brief: state.brief,
          prompt: state.prompt,
          history: state.history,
          turnIndex: state.turnIndex,
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
        content: enforceMaxChars(content, 250),
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
}) {
  const workflow = new StateGraph(SummaryState)
    .addNode("summary_agent", async (state) => {
      const model = getModel();
      const system = [
        "你负责本次共享讨论的全局 harvest。",
        "请提炼出创新观点、独特洞察、关键张力和下一步可能性。",
        "不要抹平分歧，保留可继续研究的问题。",
        "总结总字数不超过500个中文字符。",
        "请基于完整讨论记录、初始 brief、初始 prompt 和用户笔记进行综合，不要编造讨论中没有出现的信息。"
      ].join("\n");

      const notesText =
        state.notes.length === 0
          ? "暂无用户高亮笔记。"
          : state.notes.map((note, index) => `${index + 1}. ${note.agentName}：${note.text}`).join("\n");

      const human = [
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
      return {
        summary: enforceMaxChars(messageContentToString(response), 500)
      };
    })
    .addEdge(START, "summary_agent")
    .addEdge("summary_agent", END)
    .compile();

  const result = await workflow.invoke(input);
  return result.summary;
}

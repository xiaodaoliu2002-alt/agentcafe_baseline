export type AgentKind = "skill" | "custom" | "summary";

export type AgentConfig = {
  id: string;
  name: string;
  kind: AgentKind;
  role: string;
  source?: string;
  shortRole?: string;
};

export type DiscussionMessage = {
  id: string;
  agentId: string;
  agentName: string;
  kind: AgentKind;
  round: number;
  turnIndex: number;
  content: string;
  createdAt: string;
};

export type Note = {
  id: string;
  text: string;
  messageId: string;
  agentName: string;
  createdAt: string;
};

export type CustomAgentInput = {
  name: string;
  role: string;
};

export type SubmitAnswers = {
  insight: string;
  concept: string;
  evidence: string;
  nextStep: string;
};

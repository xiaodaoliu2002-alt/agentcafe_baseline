export type AgentKind = "skill" | "custom" | "summary";

export type AgentConfig = {
  id: string;
  name: string;
  kind: AgentKind;
  role: string;
  source?: string;
  shortRole?: string;
  apiKeyIndex?: number;
};

export type DiscussionSettings = {
  participantCount: number;
  roundCount: number;
  speechesPerAgentPerRound: number;
};

export type SummaryMode = "round" | "final";

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
  userNeed: string;
  designProblem: string;
  designDirection: string;
};

export type UserBehaviorLog = {
  id: string;
  createdAt: string;
  round: number;
  turnIndex: number;
  action: string;
  details: Record<string, unknown>;
};

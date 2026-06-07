import fs from "node:fs/promises";
import path from "node:path";
import JSZip from "jszip";
import type { AgentConfig } from "@/lib/types";
import { clipText, compactWhitespace } from "@/lib/text";

let fixedAgentsCache: Promise<AgentConfig[]> | undefined;

const ZIP_AGENT_NAMES: Record<string, string> = {
  "liu-long-agent.zip": "刘胧视角 Agent",
  "wang-meng-agent.zip": "王萌视角 Agent",
  "wang-shouzhi-agent.zip": "王受之视角 Agent",
  "lou-yongqi-agent.zip": "娄永琪视角 Agent"
};

const FILE_PRIORITY = ["AGENT.md", "SKILL.md", "persona.md", "router.md"];

export function getFixedSkillAgents() {
  fixedAgentsCache ??= loadFixedSkillAgents();
  return fixedAgentsCache;
}

async function loadFixedSkillAgents(): Promise<AgentConfig[]> {
  const skillDir = path.join(process.cwd(), "skill");
  const names = await fs.readdir(skillDir);
  const zipNames = names.filter((name) => name.endsWith(".zip")).sort();

  const agents = await Promise.all(
    zipNames.map(async (zipName) => {
      const buffer = await fs.readFile(path.join(skillDir, zipName));
      const zip = await JSZip.loadAsync(buffer);
      const allFiles = Object.values(zip.files).filter(
        (file) => !file.dir && !file.name.startsWith("__MACOSX/") && !file.name.includes("/._")
      );

      const sections: string[] = [];
      for (const fileName of FILE_PRIORITY) {
        const file = allFiles.find((entry) => entry.name.endsWith(`/${fileName}`) || entry.name === fileName);
        if (file) {
          const text = await file.async("string");
          sections.push(`## ${fileName}\n${text}`);
        }
      }

      const rawRole = compactWhitespace(sections.join("\n\n"));
      const role = clipText(rawRole, 14000);
      const fallbackName = zipName.replace(/-agent\.zip$/i, "").replace(/-/g, " ");
      const name = ZIP_AGENT_NAMES[zipName] ?? fallbackName;

      return {
        id: zipName.replace(/\.zip$/i, ""),
        name,
        kind: "skill" as const,
        source: `skill/${zipName}`,
        role,
        shortRole: clipText(rawRole, 220)
      };
    })
  );

  return agents;
}

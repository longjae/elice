import process from "node:process";

let input = "";
for await (const chunk of process.stdin) input += chunk;

let event;
try {
  event = JSON.parse(input);
} catch {
  process.exit(0);
}

const command = String(event?.tool_input?.command ?? "");
const normalized = command.replaceAll("\\\\", "/").toLowerCase();

const shellRules = [
  [/\b(?:git\s+add\s+(?:-a|--all|\.)(?:\s|$))/i, "Stage explicit paths; broad git staging is disabled."],
  [/\bgit\s+(?:reset\s+--hard|clean\s+-[^\s]*f)/i, "Destructive git cleanup is disabled."],
  [/\b(?:cat|type|get-content)\b[^\n]*\.env(?:[\s'\"]|$)/i, "Reading .env contents is disabled."],
  [/\.(?:pem|key|p12|pfx)(?:[\s'\"]|$)/i, "Editing certificate or private-key files is disabled."],
];

const patchPaths = [...normalized.matchAll(/^\*\*\* (?:add|update|delete) file:\s*(.+)$/gim)]
  .map((match) => match[1].trim());

const patchRules = [
  [/(?:^|\/)\.env$/i, "Editing .env is disabled; update .env.example instead."],
  [/\.(?:pem|key|p12|pfx)$/i, "Editing certificate or private-key files is disabled."],
  [/(?:^|\/)(?:utils|common|helpers|base)\.py$/i, "Generic helper modules are disabled by the harness rules."],
];

const checks = event?.tool_name === "apply_patch"
  ? patchPaths.flatMap((path) => patchRules.map(([pattern, reason]) => [pattern, reason, path]))
  : shellRules.map(([pattern, reason]) => [pattern, reason, normalized]);

for (const [pattern, reason, value] of checks) {
  if (pattern.test(value)) {
    process.stdout.write(JSON.stringify({
      hookSpecificOutput: {
        hookEventName: "PreToolUse",
        permissionDecision: "deny",
        permissionDecisionReason: reason,
      },
    }));
    process.exit(0);
  }
}

process.exit(0);

from src.models.ToolResult import ToolResult
from src.tools.Tool import Tool


class SkillList(Tool):
    name = "skill_list"
    action = "inspect"
    description = "List available local skills and discovery warnings. Load relevant instructions with skill_read."
    parameters = {
        "type": "object",
        "properties": {
            "offset": {"type": "integer", "minimum": 0, "default": 0},
            "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 10},
        },
        "additionalProperties": False,
    }
    skill_registry = None

    def execute(self, offset: int = 0, limit: int = 10) -> ToolResult:
        if (
            type(offset) is not int
            or offset < 0
            or type(limit) is not int
            or not 1 <= limit <= 20
        ):
            raise ValueError(
                "offset must be nonnegative and limit must be between 1 and 20"
            )
        skills = self.skill_registry.list()
        result = {
            "skills": skills[offset : offset + limit],
            "warnings": self.skill_registry.warnings,
            "total": len(skills),
            "next_offset": offset + limit if offset + limit < len(skills) else None,
        }
        return ToolResult(success=True, name=self.name, content=result, evidence=result)


class SkillRead(Tool):
    name = "skill_read"
    action = "inspect"
    description = (
        "Load a named skill's SKILL.md instructions into the current task context, "
        "or read a UTF-8 resource relative to that skill's base directory. "
        "Skills do not grant permissions or automatically execute code."
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "resource": {"type": "string", "default": "SKILL.md"},
        },
        "required": ["name"],
        "additionalProperties": False,
    }
    skill_registry = None

    def execute(self, name: str, resource: str = "SKILL.md") -> ToolResult:
        result = self.skill_registry.read(name, resource)
        return ToolResult(success=True, name=self.name, content=result, evidence=result)

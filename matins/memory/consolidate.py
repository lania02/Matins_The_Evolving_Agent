"""SLOW consolidation: propose / approve taste-skill versions (DESIGN.md section 5).

The slow kernel periodically distils the event log into a proposed update to the
taste skill. Unless human approval is disabled, a proposal is parked unapproved
and the human is notified over the messaging channel; approval flips the flag and
mirrors the approved content into skills/taste.md so it is human-readable on disk.
Notification failures are swallowed: an advisory step must not crash the loop.
"""
from __future__ import annotations

from .kernels import compute_memory


def run_consolidation(cfg, store, llm, messaging, approve_version=None) -> dict:
    """Approve an existing proposal, or propose a new skill version from the slow kernel.

    Always returns a dict with a 'message' string (and a 'version' when applicable).
    """
    if approve_version is not None:
        sv = store.get_skill_version(approve_version)
        if sv is None:
            return {"message": "no skill version " + str(approve_version)}
        store.approve_skill(approve_version)
        _write_taste_md(cfg, sv.content)
        return {
            "message": "approved skill v" + str(approve_version) + " and updated skills/taste.md",
            "version": approve_version,
        }

    slow = cfg.slow_kernel
    if slow is None:
        return {"message": "no slow kernel configured"}
    proposal = compute_memory(slow, store, llm, cfg.prompts_dir())
    if not proposal.strip():
        return {"message": "not enough history to propose a skill update yet"}

    cur = store.active_skill()
    parent = cur.version if cur else None
    approved = 0 if cfg.consolidation.require_human_approval else 1
    version = store.insert_skill_version(proposal, parent, "auto-proposed from slow kernel", approved)

    if approved:
        _write_taste_md(cfg, proposal)
        return {"message": "committed skill v" + str(version) + " (auto-approved)", "version": version}

    if messaging is not None:
        try:
            messaging.send(
                "Proposed taste-skill update v" + str(version) + ":\n\n"
                + proposal[:3500]
                + "\n\nApprove with: matins consolidate --approve " + str(version)
            )
        except Exception:
            pass
    return {
        "message": "proposed skill v" + str(version)
        + "; awaiting approval (matins consolidate --approve " + str(version) + ")",
        "version": version,
    }


def _write_taste_md(cfg, content: str) -> None:
    skills_dir = cfg.skills_dir()
    skills_dir.mkdir(parents=True, exist_ok=True)
    (skills_dir / "taste.md").write_text(content, encoding="utf-8")

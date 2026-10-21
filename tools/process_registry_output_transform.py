"""Background-process output transform + secret redaction.

Split out of ``process_registry.py`` (facade line cap): the plugin output-transform seam and
the redaction that follows it are one pure pipeline over a result dict, and keeping them
beside the rest of the output surfaces (`process_registry_output_log.py`) is where a reader
looks for "what happens to this text before the model sees it".
"""

from typing import Optional


def transform_process_output(output: str, *, command: str, returncode: Optional[int], task_id: str = "") -> str:
    """``transform_terminal_output`` seam for background-process output — the poll/wait/log/kill
    results and the completion/heartbeat/watch notifications — so a plugin that rewrites terminal
    output sees the same command output whether it ran in the foreground or not (#70760).
    Same helper as the foreground path; callers redact AFTER it, never before, so a replacement
    the plugin returns is still masked. ``returncode`` is None while the process is running and
    ``env_type`` is not recorded per process, so it is passed empty."""
    from tools.terminal_tool_result import _apply_output_transform_hook
    return _apply_output_transform_hook(command, output, returncode, task_id or "", "")


def _redact_process_result(result: dict) -> dict:
    """Transform, then redact secrets from background-process output before it reaches the
    model, session.db and CLI, mirroring the foreground ``terminal`` pipeline (hook first,
    redaction after) so the two surfaces can't diverge. Respects ``security.redact_secrets``;
    ``redact_terminal_output`` picks ``code_file`` from the recorded command.

    The command string itself is also redacted in case it carried an inline credential. See #43025.
    """
    if not isinstance(result, dict):
        return result
    from agent.redact import redact_sensitive_text, redact_terminal_output

    # Late import: the registry singleton is defined at the end of the facade, which imports
    # this module — a module-level import would be circular.
    from tools.process_registry import process_registry

    command = result.get("command") or ""
    # The hook's task_id is the process OWNER's (poll/log/wait results carry only session_id).
    task_id = str(result.get("task_id") or "")
    if not task_id and (session := process_registry.get(str(result.get("session_id") or ""))) is not None:
        task_id = str(getattr(session, "task_id", "") or "")
    for key in ("output", "output_preview"):
        if isinstance(value := result.get(key), str) and value:
            value = transform_process_output(value, command=command, returncode=result.get("exit_code"), task_id=task_id)
            result[key] = redact_terminal_output(value, command)
    if isinstance(command, str) and command:
        result["command"] = redact_sensitive_text(command, code_file=True)
    return result

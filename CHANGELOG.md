# Changelog

## 0.1.2 - 2026-05-14

- Project persisted and replayed message content blocks down to the Anthropic
  Messages API schema, stripping SDK-only fields such as `parsed_output`.

## 0.1.1 - 2026-05-14

- Set the backend `cli_cmd` to the active Python executable so Takopi's
  onboarding CLI availability precheck passes for this pure-Python engine.

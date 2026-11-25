# Security

- **The test page is local only.** `streamclip serve` listens on `127.0.0.1` and has no
  sign-in. Don't put it behind a public address.
- **Your API key** is read from `ANTHROPIC_API_KEY` (or `JUDGE_ANTHROPIC_KEY`) and sent only
  to Anthropic. It is never written into a job folder.
- **What leaves your computer:** the broadcast is downloaded from Twitch; only the
  shortlisted moments' transcripts and frame sheets are sent to Anthropic.

## Reporting a problem

Please report security problems privately through GitHub's "Report a vulnerability" button
(the repository's Security tab) rather than a public issue. You should hear back within a week.

---
name: code-review
description: Review a code change for concrete correctness, security, compatibility and test coverage issues without making unsolicited edits.
---

Read the diff, affected implementation, callers and relevant tests. Prioritize reproducible defects over stylistic preferences.

For each finding, provide the affected file, triggering conditions, incorrect behavior, practical impact, and a focused correction. Check permission boundaries, resource cleanup, cancellation, races, malformed input, and compatibility where relevant.

Use permitted read tools and test commands. Do not modify files unless the user requests fixes. Distinguish verified failures from suspected risks. If there are no findings, say so and state the coverage limits rather than claiming the code is perfect.

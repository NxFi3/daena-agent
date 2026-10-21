---
name: systematic-debugging
description: Investigate failures, exceptions, incorrect output, flaky tests, and runtime regressions before choosing a fix.
---

1. Record the failure, expected behavior, reproduction command, and relevant environment versions.
2. Reduce the reproduction to the smallest case that still fails. Separate missing dependencies or configuration from defects in code.
3. Trace the failing value through its producers and consumers. Form a specific hypothesis supported by logs or inspected code.
4. Change one relevant cause at a time. Do not suppress exceptions, disable checks, or increase retries to hide the defect.
5. Add a regression test that fails for the original behavior when the change has meaningful runtime impact.
6. Run the reproduction and nearby tests. If the evidence contradicts the hypothesis, revise it before another edit.
7. Explain the cause, fix, and verified scope. State explicitly when a real external service was not tested.

# Live multi-agent coordination

Daena subagents run in separate threads with independent agent contexts and
LLM clients. They can now exchange **live, attributed messages** during a run.

## Parent to child
The parent calls `subagent(action="message", recipient="<agent_id>", text="...")`
(or uses `agent_id` instead of `recipient`). The manager queues the message
via the running child agent's coordination queue. It is read on that child's
next model iteration, without cancelling or restarting the task.

## Child to parent / another child
The child uses `agent_message(action="send", recipient="parent", text="...")`
or sets `recipient` to the peer child's agent ID. The recipient must exist
and still be running. The parent receives messages through its low-trust
runtime observation queue, and can retrieve ordered incoming messages using
`subagent(action="inbox")`. Children can read their own received messages
using `agent_message(action="inbox")`.

Messages do not replace a child's original task, do not grant access to other
tools, and do not bypass workspace isolation. They are presented as
**untrusted coordination data**, distinct from human steering commands.

The SubagentManager emits `message_sent` and `message_received` events for
Studio with agent IDs, message IDs and bounded message text. Up to 1,000
messages per run are retained and each recipient advances an independent
inbox cursor; each message is at most 4,000 characters.

## Permissions and scope
This uses existing `subagents.enabled`. Child messaging is available when
delegation is permitted by the parent's default security policy. When the
parent has an explicit `security.allowed_tools` list, include
`agent_message` as well as `subagent` to enable messaging. For a particular
child, an optional `allowed_tools` filter can exclude `agent_message`.

Coordination is event-driven, not guaranteed instantaneous. A child waiting
for a slow model response processes inbound messages at its next turn.
Messaging does not create additional LLM calls by itself; agents decide when
coordination is necessary. Parent and child cannot each recursively delegate:
children still have subagent creation disabled.

## Limitations
This is a local, in-process message bus, not network transport and not shared
across separate Daena processes. There is no priority-based interruption of
an in-flight model response, no persisted cross-run chat, no delivery to a
completed agent, and no autonomous multi-agent negotiation protocol.

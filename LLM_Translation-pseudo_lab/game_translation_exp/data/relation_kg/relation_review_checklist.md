# Relation Edge Review Checklist

Keep edge if all checks pass:
- Source/target are actual lore entities (not generic words).
- Relation type is semantically plausible in context.
- Evidence sentence directly supports relation.
- Direction is reasonable (A -> relation -> B).
- Confidence updated (high/medium/low) after review.

Drop edge if:
- Entity is a noisy token (e.g., generic noun phrase).
- Relation is only co-occurrence with no translation value.
- Evidence is ambiguous or contextless.

Recommended keep-priority for translation:
- commands
- mentor_of
- enemy_of
- ally_of
- trusts / distrusts
- speaks_to (when dialogue register matters)
